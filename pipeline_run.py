import os
from pathlib import Path

from tfx.orchestration import metadata, pipeline
from tfx.orchestration.local.local_dag_runner import LocalDagRunner

from base_pipeline import init_components


PIPELINE_NAME = "charges_pipeline_local"

# Resolve paths relative to this file so the script works from any working directory.
# Override DATA_DIR or MODULE_FILE via environment variables if needed.
PROJECT_ROOT = Path(__file__).resolve().parent
AIRFLOW_HOME = Path(os.environ.get("AIRFLOW_HOME", Path.home() / "airflow"))

DATA_DIR = os.environ.get("TFX_DATA_DIR", str(PROJECT_ROOT / "data"))
MODULE_FILE = os.environ.get("TFX_MODULE_FILE", str(PROJECT_ROOT / "module.py"))
PIPELINE_ROOT = str(AIRFLOW_HOME / "pipelines" / PIPELINE_NAME)
METADATA_PATH = str(AIRFLOW_HOME / "metadata" / PIPELINE_NAME / "metadata.sqlite")
SERVING_MODEL_DIR = str(AIRFLOW_HOME / "serving_model" / PIPELINE_NAME)


def init_pipeline(components, pipeline_root: str) -> pipeline.Pipeline:
    return pipeline.Pipeline(
        pipeline_name=PIPELINE_NAME,
        pipeline_root=pipeline_root,
        components=components,
        enable_cache=True,
        metadata_connection_config=metadata.sqlite_metadata_connection_config(METADATA_PATH),
    )


if __name__ == "__main__":
    components = init_components(
        DATA_DIR,
        MODULE_FILE,
        training_steps=1000,
        eval_steps=100,
        serving_model_dir=SERVING_MODEL_DIR,
    )
    tfx_pipeline = init_pipeline(components, PIPELINE_ROOT)

    print(f"Starting pipeline: {PIPELINE_NAME}")
    print(f"  Data dir      : {DATA_DIR}")
    print(f"  Pipeline root : {PIPELINE_ROOT}")
    print(f"  Serving dir   : {SERVING_MODEL_DIR}")
    LocalDagRunner().run(tfx_pipeline)
