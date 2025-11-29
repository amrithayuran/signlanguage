import os
import config
from keras.models import model_from_json
import numpy as np

def test_model_loading():
    print("Testing model loading...")
    
    # Check if model files exist
    if not os.path.exists(config.MODEL_BW_JSON) or not os.path.exists(config.MODEL_BW_H5):
        print(f"Model files not found in {config.MODEL_DIR}. Skipping loading test.")
        return

    try:
        with open(config.MODEL_BW_JSON, 'r') as jf:
            model_json = jf.read()
        model = model_from_json(model_json)
        model.load_weights(config.MODEL_BW_H5)
        print("Main model loaded successfully.")
        
        # Test prediction with dummy data
        dummy_input = np.zeros((1, config.IMG_SIZE, config.IMG_SIZE, 1))
        prediction = model.predict(dummy_input, verbose=0)
        print(f"Prediction shape: {prediction.shape}")
        print("Prediction test successful.")
        
    except Exception as e:
        print(f"Model loading failed: {e}")

if __name__ == "__main__":
    test_model_loading()
