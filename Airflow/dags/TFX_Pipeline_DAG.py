import os
from datetime import datetime

from airflow import DAG
from airflow.operators.bash import BashOperator


# TFX must run in its own venv due to dependency conflicts with Airflow.
# Set these via environment variables or update the defaults below.
TFX_VENV = os.environ.get("TFX_VENV_PATH", "/path/to/tfx_venv")
PIPELINE_SCRIPT = os.environ.get("TFX_PIPELINE_SCRIPT", "/path/to/pipeline_run.py")

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": datetime(2024, 12, 1),
    "retries": 1,
}

with DAG(
    dag_id="tfx_pipeline_dag",
    default_args=default_args,
    description="Orchestrate TFX medical insurance cost prediction pipeline",
    schedule_interval=None,
    catchup=False,
    tags=["tfx", "mlops", "regression"],
) as dag:
    run_tfx_pipeline = BashOperator(
        task_id="run_tfx_pipeline",
        bash_command=f"source {TFX_VENV}/bin/activate && python3 {PIPELINE_SCRIPT}",
    )
