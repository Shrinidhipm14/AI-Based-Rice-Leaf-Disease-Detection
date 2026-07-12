import tensorflow as tf
import numpy as np
import cv2
import os
import tkinter as tk
from tkinter import filedialog, Toplevel
from PIL import Image, ImageTk
from gtts import gTTS
import pygame
import threading
import time

# 1. LOAD MODEL
try:
    # Use compile=False to ensure it opens regardless of how it was trained
    model = tf.keras.models.load_model('rice_disease_model.keras', compile=False)
    print("✅ System Ready: Model Loaded Successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")

# 2. ALPHABETICAL LABELS (Ensures correct naming)
results_kn = [
    {"name": "Bacterial Leaf Blight", "kn": "ಬ್ಯಾಕ್ಟೀರಿಯಲ್ ಲೀಫ್ ಬ್ಲೈಟ್", "advice": "ಗದ್ದೆಯಲ್ಲಿ ನೀರನ್ನು ಹೊರಹಾಕಿ ಮತ್ತು ಸಾರಜನಕ ಗೊಬ್ಬರ ನಿಲ್ಲಿಸಿ."},
    {"name": "Brown Spot", "kn": "ಕಂದು ಚುಕ್ಕೆ ರೋಗ", "advice": "ಪೊಟ್ಯಾಶ್ ಗೊಬ್ಬರ ಬಳಸಿ ಮಣ್ಣಿನ ಪೌಷ್ಟಿಕಾಂಶ ಹೆಚ್ಚಿಸಿ."},
    {"name": "Healthy", "kn": "ಆರೋಗ್ಯಕರ ಪೈರು", "advice": "ನಿಮ್ಮ ಬೆಳೆ ಆರೋಗ್ಯವಾಗಿದೆ! ಯಾವುದೇ ಔಷಧಿಯ ಅಗತ್ಯವಿಲ್ಲ."},
    {"name": "Leaf Blast", "kn": "ಬ್ಲಾಸ್ಟ್ ರೋಗ", "advice": "ತಕ್ಷಣ ಶಿಲೀಂಧ್ರನಾಶಕ ಸಿಂಪಡಿಸಿ."},
    {"name": "Leaf Scald", "kn": "ಎಲೆ ಸ್ಕಾಲ್ಡ್ ರೋಗ", "advice": "ಹೊಲದಲ್ಲಿ ನೀರು ನಿಲ್ಲದಂತೆ ಎಚ್ಚರವಹಿಸಿ."},
    {"name": "Sheath Blight", "kn": "ಶೀತ್ ಬ್ಲೈಟ್ ರೋಗ", "advice": "ಸಸ್ಯಗಳ ನಡುವೆ ಗಾಳಿ ಆಡುವಂತೆ ನೋಡಿ."}
]

class FarmerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Smart Rice Doctor")
        self.root.geometry("600x850")
        self.root.configure(bg="#1b5e20")
        
        # Init pygame mixer
        pygame.mixer.init()

        # UI Design
        tk.Label(root, text="ರೈತ ಮಿತ್ರ", font=("Helvetica", 24, "bold"), fg="white", bg="#2d3436", pady=10).pack(pady=15, fill="x")

        self.img_frame = tk.Frame(root, width=450, height=300, bg="#bdc3c7", highlightthickness=2, highlightbackground="white")
        self.img_frame.pack_propagate(0) 
        self.img_frame.pack(pady=5)

        self.img_label = tk.Label(self.img_frame, text="ಚಿತ್ರವನ್ನು ಆಯ್ಕೆಮಾಡಿ", font=("Arial", 12), bg="#bdc3c7")
        self.img_label.pack(expand=True, fill="both")

        btn_container = tk.Frame(root, bg="#1b5e20")
        btn_container.pack(pady=30) 
        self.btn_upload = tk.Button(btn_container, text="📸 UPLOAD IMAGE", font=("Arial", 12, "bold"), 
                                    command=self.show_options, bg="#00b894", fg="white", padx=20, pady=10)
        self.btn_upload.pack()

        self.res_frame = tk.Frame(root, bg="#2d3436", padx=15, pady=20)
        self.res_frame.pack(fill="x", side="bottom", pady=0)
        
        self.acc_label = tk.Label(self.res_frame, text="", font=("Arial", 11, "bold"), bg="#2d3436", fg="#fdcb6e")
        self.acc_label.pack()

        self.result_title = tk.Label(self.res_frame, text="ಸಿದ್ಧವಾಗಿದೆ", font=("Arial", 18, "bold"), bg="#2d3436", fg="white")
        self.result_title.pack(pady=2)

        self.result_advice = tk.Label(self.res_frame, text="", font=("Arial", 13, "bold"), bg="#2d3436", fg="#55efc4", wraplength=550, justify="center")
        self.result_advice.pack(pady=5)

    def show_options(self):
        opt = Toplevel(self.root)
        opt.geometry("300x180")
        tk.Button(opt, text="📁 Gallery", width=15, command=lambda:[opt.destroy(), self.from_gallery()]).pack(pady=10)
        tk.Button(opt, text="📷 Camera", width=15, command=lambda:[opt.destroy(), self.from_camera()]).pack(pady=10)

    def from_gallery(self):
        path = filedialog.askopenfilename()
        if path: self.process_image(path)

    def from_camera(self):
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        if ret:
            cv2.imwrite("temp_img.jpg", frame)
            self.process_image("temp_img.jpg")
        cap.release()

    def process_image(self, path):
        try:
            # 1. Refresh UI
            img_ui = Image.open(path).resize((450, 300))
            photo = ImageTk.PhotoImage(img_ui)
            self.img_label.config(image=photo, text="")
            self.img_label.image = photo
            self.result_title.config(text="ವಿಶ್ಲೇಷಿಸಲಾಗುತ್ತಿದೆ...", fg="#fdcb6e")
            self.root.update()

            # 2. Image Pre-processing (Forcing 224x224 to match model)
            raw_img = cv2.imread(path)
            raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(raw_img, (224, 224))
            img_array = resized.astype(np.float32) / 255.0
            final_input = np.expand_dims(img_array, axis=0)
            
            # 3. Prediction
            pred = model.predict(final_input, verbose=0)
            idx = np.argmax(pred[0])
            accuracy_score = pred[0][idx] * 100

            # 4. Show Result
            res = results_kn[idx]
            self.acc_label.config(text=f"Accuracy: {accuracy_score:.2f}%")
            self.result_title.config(text=f"ರೋಗ: {res['kn']}", fg="white")
            self.result_advice.config(text=f"ಸಲಹೆ: {res['advice']}")
            
            # 5. Audio Output (New threading logic)
            audio_text = f"ಪತ್ತೆಯಾದ ರೋಗ {res['kn']}. {res['advice']}"
            threading.Thread(target=self.play_audio, args=(audio_text,)).start()

        except Exception as e:
            print(f"❌ Error: {e}")
            self.result_title.config(text="ವಿಶ್ಲೇಷಣೆ ವಿಫಲವಾಗಿದೆ", fg="red")

    def play_audio(self, text):
        try:
            # Stop and UNLOAD to release the file for the next upload
            if pygame.mixer.music.get_busy():
                pygame.mixer.music.stop()
            pygame.mixer.music.unload() 
            
            tts = gTTS(text=text, lang='kn')
            tts.save("voice.mp3")
            
            pygame.mixer.music.load("voice.mp3")
            pygame.mixer.music.play()
            
            # Keep alive while playing
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
        except Exception as e:
            print(f"🔊 Audio error: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = FarmerApp(root)
    root.mainloop()