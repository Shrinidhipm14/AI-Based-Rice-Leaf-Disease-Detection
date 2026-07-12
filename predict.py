import tensorflow as tf
import numpy as np
import cv2
import os
from gtts import gTTS
import pygame
import time

# 1. Load the Master Brain
# Make sure 'rice_disease_model.keras' is in the same folder as this script
model = tf.keras.models.load_model('rice_disease_model.keras')

def speak_kannada(text):
    """Converts text to Kannada speech and plays it."""
    try:
        tts = gTTS(text=text, lang='kn')
        filename = "advice.mp3"
        tts.save(filename)
        
        pygame.mixer.init()
        pygame.mixer.music.load(filename)
        pygame.mixer.music.play()
        
        while pygame.mixer.music.get_busy():
            time.sleep(1)
            
        pygame.mixer.quit()
        if os.path.exists(filename):
            os.remove(filename)
    except Exception as e:
        print(f"Audio Error: {e}")

def process_and_predict():
    # Input from user
    image_path = input("\nEnter the image path (or drag the photo here): ").strip().replace("'", "").replace('"', '')

    if not os.path.exists(image_path):
        print(f"ದೋಷ: '{image_path}' ಎಂಬ ಫೈಲ್ ಸಿಗುತ್ತಿಲ್ಲ!") 
        return

    # Image Preprocessing
    img = cv2.imread(image_path)
    img_resized = cv2.resize(img, (224, 224))
    img_normalized = img_resized / 255.0
    img_final = np.expand_dims(img_normalized, axis=0)

    # AI Prediction
    prediction = model.predict(img_final, verbose=0)
    index = np.argmax(prediction[0])
    confidence = prediction[0][index] * 100
    
    # The Database - Must match the order in your 'dataset' folder
    results_kn = [
        {"name": "Bacterial Leaf Blight", "advice": "ಬ್ಯಾಕ್ಟೀರಿಯಲ್ ಲೀಫ್ ಬ್ಲೈಟ್ ರೋಗ ತಗುಲಿದೆ. ಗದ್ದೆಯಲ್ಲಿ ನೀರನ್ನು ಹೊರಹಾಕಿ ಮತ್ತು ಸಾರಜನಕ ಗೊಬ್ಬರ ನಿಲ್ಲಿಸಿ."},
        {"name": "Brown Spot", "advice": "ಇದು ಕಂದು ಚುಕ್ಕೆ ರೋಗ. ಮಣ್ಣಿನ ಫಲವತ್ತತೆ ಹೆಚ್ಚಿಸಲು ಪೊಟ್ಯಾಶ್ ಗೊಬ್ಬರ ಬಳಸಿ."},
        {"name": "Healthy Rice Leaf", "advice": "ನಿಮ್ಮ ಬೆಳೆ ತುಂಬಾ ಆರೋಗ್ಯವಾಗಿದೆ! ಯಾವುದೇ ಔಷಧಿಯ ಅಗತ್ಯವಿಲ್ಲ, ಹೀಗೆ ಕಾಪಾಡಿಕೊಳ್ಳಿ."},
        {"name": "Leaf Blast", "advice": "ಇದು ಬ್ಲಾಸ್ಟ್ ರೋಗ. ಇದು ವೇಗವಾಗಿ ಹರಡುತ್ತದೆ, ತಕ್ಷಣ ಶಿಲೀಂಧ್ರನಾಶಕ ಸಿಂಪಡಿಸಿ."},
        {"name": "Leaf Scald", "advice": "ಇದು ಎಲೆ ಸ್ಕಾಲ್ಡ್ ರೋಗ. ಹೊಲದಲ್ಲಿ ನೀರು ನಿಲ್ಲದಂತೆ ನೋಡಿಕೊಳ್ಳಿ ಮತ್ತು ಪೌಷ್ಟಿಕಾಂಶದ ಕಡೆ ಗಮನ ಕೊಡಿ."},
        {"name": "Sheath Blight", "advice": "ಇದು ಶೀತ್ ಬ್ಲೈಟ್ ರೋಗ. ಸಸ್ಯಗಳ ನಡುವೆ ಗಾಳಿ ಆಡುವಂತೆ ನೋಡಿಕೊಳ್ಳಿ ಮತ್ತು ತಜ್ಞರ ಸಲಹೆ ಪಡೆಯಿರಿ."}
    ]

    print("\n" + "=" * 50)
    
    # 2. THE SAFETY FILTER (Crucial for impact)
    if confidence < 75:
        low_conf_msg = "ಕ್ಷಮಿಸಿ, ಫೋಟೋ ಸ್ಪಷ್ಟವಾಗಿಲ್ಲ. ದಯವಿಟ್ಟು ಹೆಚ್ಚಿನ ಬೆಳಕಿನಲ್ಲಿ ಎಲೆಯ ಹತ್ತಿರದಿಂದ ಮತ್ತೊಂದು ಫೋಟೋ ತೆಗೆಯಿರಿ."
        print(f"CONFIDENCE: {confidence:.2f}% (Too Low!)")
        print("-" * 50)
        print(low_conf_msg)
        speak_kannada(low_conf_msg)
    else:
        detected = results_kn[index]
        print(f"ಪತ್ತೆಯಾದ ರೋಗ (Detected): {detected['name']}")
        print(f"ನಂಬಿಕೆ (Confidence): {confidence:.2f}%")
        print("-" * 50)
        print(f"ಸಲಹೆ: {detected['advice']}")
        
        # Final voice output for the farmer
        full_audio_text = f"ಪತ್ತೆಯಾದ ರೋಗ {detected['name']}. {detected['advice']}"
        speak_kannada(full_audio_text)

    print("=" * 50 + "\n")

if __name__ == "__main__":
    process_and_predict()