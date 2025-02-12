from keras.models import load_model

model_path = "artifacts/model.keras"  # Adjust if needed
try:
    model = load_model(model_path)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Model loading failed: {e}")
