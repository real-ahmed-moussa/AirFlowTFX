import os
from typing import NamedTuple, Dict, Any, Text

import tensorflow as tf
import tensorflow_transform as tft
from tfx.components.trainer.fn_args_utils import FnArgs
from keras_tuner.engine import base_tuner
from keras_tuner import HyperParameters, RandomSearch


LABEL_KEY = "charges"

CATEGORICAL_FEATURES: Dict[str, int] = {
    "region": 4,
    "sex": 2,
    "smoker": 2,
}

NUMERICAL_FEATURES = ["age", "bmi", "children"]


def transformed_name(key: str) -> str:
    return key + "_xf"


def fill_in_missing(x):
    """Convert SparseTensor to Dense and squeeze the trailing size-1 dimension."""
    if not isinstance(x, tf.sparse.SparseTensor):
        return x
    default_value = "" if x.dtype == tf.string else 0
    x = tf.sparse.to_dense(
        tf.SparseTensor(x.indices, x.values, [x.dense_shape[0], 1]),
        default_value,
    )
    return tf.squeeze(x, axis=1)


def convert_num_to_one_hot(label_tensor: tf.Tensor, num_labels: int = 2) -> tf.Tensor:
    one_hot_tensor = tf.one_hot(label_tensor, num_labels)
    return tf.reshape(one_hot_tensor, [-1, num_labels])


def preprocessing_fn(inputs: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
    outputs = {}

    for key, dim in CATEGORICAL_FEATURES.items():
        int_value = tft.compute_and_apply_vocabulary(
            fill_in_missing(inputs[key]), top_k=dim
        )
        outputs[transformed_name(key)] = convert_num_to_one_hot(int_value, num_labels=dim)

    for key in NUMERICAL_FEATURES:
        outputs[transformed_name(key)] = tft.scale_to_0_1(fill_in_missing(inputs[key]))

    outputs[transformed_name(LABEL_KEY)] = fill_in_missing(inputs[LABEL_KEY])

    return outputs


def get_model(hp: HyperParameters, show_summary: bool = True) -> tf.keras.Model:
    categorical_inputs = [
        tf.keras.Input(shape=(dim,), name=transformed_name(key))
        for key, dim in CATEGORICAL_FEATURES.items()
    ]
    numerical_inputs = [
        tf.keras.Input(shape=(1,), name=transformed_name(key))
        for key in NUMERICAL_FEATURES
    ]

    # Wide & Deep: deep branch encodes categorical embeddings, wide branch passes scaled numerics
    deep = tf.keras.layers.concatenate(categorical_inputs)
    deep = tf.keras.layers.Dense(hp.Int("deep_units_1", 64, 256, step=64), activation="relu")(deep)
    deep = tf.keras.layers.Dense(hp.Int("deep_units_2", 32, 128, step=32), activation="relu")(deep)
    deep = tf.keras.layers.Dense(hp.Int("deep_units_3", 16, 64, step=16), activation="relu")(deep)

    wide = tf.keras.layers.concatenate(numerical_inputs)
    wide = tf.keras.layers.Dense(16, activation="relu")(wide)

    output = tf.keras.layers.Dense(1)(tf.keras.layers.concatenate([deep, wide]))

    model = tf.keras.Model(categorical_inputs + numerical_inputs, output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=hp.Float("learning_rate", 1e-4, 1e-2, sampling="log")
        ),
        loss="mean_squared_error",
        metrics=[
            tf.keras.metrics.MeanAbsoluteError(),
            tf.keras.metrics.MeanSquaredError(),
            tf.keras.metrics.RootMeanSquaredError(),
        ],
    )

    if show_summary:
        model.summary()

    return model


def get_serve_tf_examples_fn(model, tf_transform_output):
    """Return a serving function that applies the TFT graph before inference."""
    model.tft_layer = tf_transform_output.transform_features_layer()

    @tf.function
    def serve_tf_examples_fn(serialized_tf_examples):
        feature_spec = tf_transform_output.raw_feature_spec()
        feature_spec.pop(LABEL_KEY)
        parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)
        transformed_features = model.tft_layer(parsed_features)
        return {"outputs": model(transformed_features)}

    return serve_tf_examples_fn


def gzip_reader_fn(filenames):
    return tf.data.TFRecordDataset(filenames, compression_type="GZIP")


def input_fn(file_pattern, tf_transform_output, batch_size: int = 64):
    transformed_feature_spec = tf_transform_output.transformed_feature_spec().copy()
    return tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transformed_feature_spec,
        reader=gzip_reader_fn,
        label_key=transformed_name(LABEL_KEY),
    )


TunerFnResult = NamedTuple(
    "TunerFnResult",
    [("tuner", base_tuner.BaseTuner), ("fit_kwargs", Dict[Text, Any])],
)


def tuner_fn(fn_args) -> TunerFnResult:
    tft_output = tft.TFTransformOutput(fn_args.transform_graph_path)

    train_data = input_fn(fn_args.train_files, tft_output, batch_size=64)
    eval_data = input_fn(fn_args.eval_files, tft_output, batch_size=64)

    tuner = RandomSearch(
        hypermodel=get_model,
        objective="val_mean_absolute_error",
        max_trials=10,
        executions_per_trial=2,
        directory=fn_args.working_dir,
        project_name="model_tuning",
    )

    fit_kwargs = {
        "x": train_data,
        "validation_data": eval_data,
        "steps_per_epoch": fn_args.train_steps,
        "validation_steps": fn_args.eval_steps,
        "epochs": 10,
    }

    return TunerFnResult(tuner=tuner, fit_kwargs=fit_kwargs)


def run_fn(fn_args: FnArgs):
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

    train_dataset = input_fn(fn_args.train_files, tf_transform_output, batch_size=64)
    eval_dataset = input_fn(fn_args.eval_files, tf_transform_output, batch_size=64)

    hp = HyperParameters.from_config(fn_args.hyperparameters)
    model = get_model(hp=hp)

    log_dir = os.path.join(os.path.dirname(fn_args.serving_model_dir), "logs")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, update_freq="batch")

    model.fit(
        train_dataset,
        epochs=1,
        steps_per_epoch=fn_args.train_steps,
        validation_data=eval_dataset,
        validation_steps=fn_args.eval_steps,
        callbacks=[tensorboard_callback],
    )

    signatures = {
        "serving_default": get_serve_tf_examples_fn(
            model, tf_transform_output
        ).get_concrete_function(
            tf.TensorSpec(shape=[None], dtype=tf.string, name="examples")
        )
    }

    model.save(fn_args.serving_model_dir, save_format="tf", signatures=signatures)
