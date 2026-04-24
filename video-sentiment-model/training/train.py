import os
import argparse
import torchaudio
import torch

# AWS Sagemaker

SM_MODEL_DIR = os.environ.get("SM_MODEL_DIR", ".")
SM_CHANNEL_TRAINING = os.environ.get("SM_CHANNEL_TRAINING", "./opt/ml/input/data/training")
SM_CHANNEL_VALIDATION = os.environ.get("SM_CHANNEL_VALIDATION", "./opt/ml/input/data/validation")
SM_CHANNEL_TEST = os.environ.get("SM_CHANNEL_TEST", "./opt/ml/input/data/test")

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = "expandable_segments:True"

def pargs_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=0.001)    

    # Data directories
    parser.add_argument('--train-dir', type=str, default=SM_CHANNEL_TRAINING)
    parser.add_argument('--val-dir', type=str, default=SM_CHANNEL_VALIDATION)
    parser.add_argument('--test-dir', type=str, default=SM_CHANNEL_TEST)
    parser.add_argument('--model-dir', type=str, default=SM_MODEL_DIR)

    return parser.parse_args()

def main():
    #install ffmpeg
    print("Available audio backends:")
    print(str(torchaudio.list_audio_backends()))

    args = pargs_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # track initial gpu memory if available
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        memory_used = torch.cuda.max_memory_allocated(device)
        

if __name__ == "__main__":
    args = pargs_args()
    print(f"Training with arguments: {args}")
    main()