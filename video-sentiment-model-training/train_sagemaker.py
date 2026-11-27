import os
from pathlib import Path
from dotenv import load_dotenv

from sagemaker.pytorch import PyTorch
from sagemaker.debugger import TensorBoardOutputConfig

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

def start_training():
    tensorboard_config = TensorBoardOutputConfig(
        s3_output_path=os.environ["SAGEMAKER_TENSORBOARD_S3_URI"],
        container_local_output_path="/opt/ml/output/tensorboard"
    )

    estimator = PyTorch(
        entry_point="train.py",
        source_dir=str(BASE_DIR / "training"),
        role=os.environ["SAGEMAKER_TRAINING_ROLE_ARN"],
        framework_version="2.5.1",
        py_version="py311",
        instance_count=1,
        instance_type="ml.g5.xlarge",
        hyperparameters={
            "batch-size": 32,
            "epochs": 25
        },
        tensorboard_config=tensorboard_config
    )

    dataset_uri = os.environ["SAGEMAKER_DATASET_S3_URI"].rstrip("/")
    # Start training
    estimator.fit({
        "training": f"{dataset_uri}/train",
        "validation": f"{dataset_uri}/dev",
        "test": f"{dataset_uri}/test"
    })


if __name__ == "__main__":
    start_training()
