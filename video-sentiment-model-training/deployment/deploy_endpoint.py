import os
from pathlib import Path
from dotenv import load_dotenv

from sagemaker.pytorch import PyTorchModel
import sagemaker

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR.parent / ".env")

def deploy_endpoint():
    sagemaker.Session()
    role = os.environ["SAGEMAKER_DEPLOYMENT_ROLE_ARN"]

    model_uri = os.environ["SAGEMAKER_MODEL_S3_URI"]

    model = PyTorchModel(
        model_data=model_uri,
        role=role,
        framework_version="2.5.1",
        py_version="py311",
        entry_point="inference.py",
        source_dir=str(BASE_DIR),
        name="sentiment-analysis-model",
    )

    predictor = model.deploy(
        initial_instance_count=1,
        instance_type="ml.g5.xlarge",
        endpoint_name=os.environ["SAGEMAKER_ENDPOINT_NAME"],
    )


if __name__ == "__main__":
    deploy_endpoint()
