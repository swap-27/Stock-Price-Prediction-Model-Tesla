import tensorflow as tf

try:
    tf.keras.models.load_model("test_model.h5")
    print("HDF5 works!")
except ImportError as e:
    print("Error:", e)