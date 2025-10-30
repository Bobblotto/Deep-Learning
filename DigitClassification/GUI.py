import tkinter as tk
from PIL import Image, ImageDraw
import numpy as np
from tensorflow.keras.models import load_model

win = tk.Tk()
win.geometry('500x500')

label = tk.Label(text='Write a number')
label.pack()

canvas = tk.Canvas(width=280, height=280, bg='white') # place to draw on
canvas.pack(pady=5)

img = Image.new('L', (280, 280), 'black') # L is for grayscale mode, image size, white is colour - blank image
imgPen = ImageDraw.Draw(img) # tool to draw on blank image

def draw(event):
    canvas.create_oval(event.x, event.y, event.x+10, event.y+10, fill='black', outline='black')
    imgPen.ellipse([event.x, event.y, event.x+10, event.y+10], 'white')


canvas.bind('<B1-Motion>', draw) # draw when b1 motion (mouse movement)

model = load_model('Deep Learning/DigitClassification/digitClassifier.h5') # load model from file

predictionLabel = tk.Label(text='None')
predictionLabel.pack()

def preprocess():
    newImg = img.resize((28, 28)) # set image to size of data piece
    newImg = np.array(newImg) / 255.0 # normalise
    newImg = np.expand_dims(newImg, axis=0) # make 3 dimensional
    prediction = model.predict(newImg)
    prediction = np.argmax(prediction)
    predictionLabel.config(text=str(prediction))


predictButton = tk.Button(text='Predict', command=preprocess)
predictButton.pack()

def clear():
    canvas.delete('all')

clearButton = tk.Button(text='Clear', command=clear)
clearButton.pack(pady=5)


win.mainloop()