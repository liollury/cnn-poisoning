import tensorflow as tf

print("TensorFlow version:", tf.__version__)
print("GPUs disponibles :", tf.config.list_physical_devices('GPU'))
print("CPUs disponibles :", tf.config.list_physical_devices('CPU'))