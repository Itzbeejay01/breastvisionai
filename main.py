from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

# Import TensorFlow (Keras included)
import tensorflow as tf


def print_device_info() -> None:
    print("TensorFlow version:", tf.__version__)
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"Detected {len(gpus)} GPU device(s):")
        for idx, gpu in enumerate(gpus):
            print(f"  [{idx}] {gpu}")
    else:
        print("No GPU detected. Running on CPU. This is expected if CUDA-enabled NVIDIA GPU is not available.")


if __name__ == "__main__":
    print_device_info()
