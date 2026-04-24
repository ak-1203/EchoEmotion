import tensorflow as tf


def configure_gpu():
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        print("No GPU detected. Using CPU.")
        return False

    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU detected: {[gpu.name for gpu in gpus]}")
        return True
    except RuntimeError as exc:
        print(f"GPU configuration warning: {exc}")
        return False