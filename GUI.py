import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from keras.models import load_model
import numpy as np

class MaskPredictionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Mask Prediction App")

        self.model = load_model('model.h5')
        self.model.make_predict_function()

        self.upload_button = tk.Button(root, text="Upload Image", command=self.load_image)
        self.upload_button.pack()

        self.canvas = tk.Canvas(root, width=300, height=300)
        self.canvas.pack()

        self.prediction_label = tk.Label(root, text="")
        self.prediction_label.pack()

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png")])
        if file_path:
            prediction = self.predict_mask(file_path)
            self.show_image(file_path)
            self.prediction_label.config(text=prediction)

    def preprocess_image(self, image_path):
        img = Image.open(image_path)
        img = img.resize((150, 150))
        img = np.array(img) / 255.0
        img = np.expand_dims(img, axis=0)
        return img

    def predict_mask(self, image_path):
        img = self.preprocess_image(image_path)
        result = self.model.predict(img)
        if result[0][0] > 0.5:
            prediction = "Correctly wearing a mask"
            print(result)
        else:
            prediction = "Not correctly wearing a mask"
            print(result)
        return prediction

    def show_image(self, image_path):
        img = Image.open(image_path)
        img.thumbnail((300, 300))
        img = ImageTk.PhotoImage(img)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=img)
        self.canvas.image = img

if __name__ == "__main__":
    root = tk.Tk()
    app = MaskPredictionApp(root)
    root.mainloop()
