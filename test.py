import tkinter as tk
from tkinter import Canvas, Button, Label
from PIL import Image, ImageDraw, ImageTk
import numpy as np
import tensorflow as tf

# Load pretrained model
model = tf.keras.models.load_model('digit_recognition_model.h5')

# Function to convert PIL Image to numpy array and preprocess
def PILImageToNumpyImage(img):
    img = img.resize((90, 140))  # Resize image to model input size
    img = img.convert('RGB')  # Convert to RGB
    img = np.array(img)
    img = img / 255.0  # Normalize image
    img = np.expand_dims(img, axis=0)
    return img

# Function to preprocess image
def preprocess_image(img):
    img = img.astype('float32')
    return img

# Function to handle prediction
def predict_digit():
    img = PILImageToNumpyImage(drawing)
    img = preprocess_image(img)
    
    predictions = model.predict(img)
    print(predictions)
    
    predicted_class = np.argmax(predictions[0])
    percen_predicted_class = predictions[0][predicted_class]
    result_label.config(text=f'Predicted Digit: {predicted_class} ')

# Function to clear canvas
def clear_canvas():
    canvas.delete("all")
    draw.rectangle((0, 0, 300, 300), fill=(255, 255, 255))  # Màu trắng để dễ nhận diện chữ số

# Create main window
root = tk.Tk()
root.title("Digit Recognition")

# Create canvas for drawing
drawing = Image.new('RGB', (300, 300), (255, 255, 255))  # Màu trắng
draw = ImageDraw.Draw(drawing)

# Function to handle mouse motion
def paint(event):
    x1, y1 = (event.x - 1), (event.y - 1)
    x2, y2 = (event.x + 1), (event.y + 1)
    canvas.create_oval(x1, y1, x2, y2, fill="red", outline="red", width=10)  # Vẽ màu đỏ
    draw.line([x1, y1, x2, y2], fill="red", width=10)  # Vẽ màu đỏ

# Bind mouse motion to canvas
canvas = Canvas(root, width=300, height=300, bg="white")  # Màu trắng để dễ nhận diện chữ số
canvas.pack()
canvas.bind("<B1-Motion>", paint)

# Create predict button
predict_button = Button(root, text="Predict", command=predict_digit)
predict_button.pack(pady=10)

# Create clear button
clear_button = Button(root, text="Clear", command=clear_canvas)
clear_button.pack(pady=10)

# Create label for prediction result
result_label = Label(root, text="", font=("Helvetica", 18))
result_label.pack()

# Run the application
root.mainloop()
