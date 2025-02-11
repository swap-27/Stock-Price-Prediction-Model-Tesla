from keras.models import load_model

model = load_model("artifacts/model.keras")  # Replace with your actual model path
model.summary()