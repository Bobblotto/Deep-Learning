import tkinter as tk
from tkinter import filedialog # not loaded automatically
from PIL import Image, ImageTk
import numpy as np
from tensorflow.keras.models import load_model

# initialising variables ----------------------- v

model = load_model('Deep Learning/German Traffic Sign Recognition/model.h5')

classes = {
    1: 'Speed limit (20km/h)',
    2: 'Speed limit (30km/h)',
    3: 'Speed limit (50km/h)',
    4: 'Speed limit (60km/h)',
    5: 'Speed limit (70km/h)',
    6: 'Speed limit (80km/h)',
    7: 'End of speed limit (80km/h)',
    8: 'Speed limit (100km/h)',
    9: 'Speed limit (120km/h)',
    10: 'No passing',
    11: 'No passing veh over 3.5 tons',
    12: 'Right-of-way at intersection',
    13: 'Priority road',
    14: 'Yield',
    15: 'Stop',
    16: 'No vehicles',
    17: 'Veh > 3.5 tons prohibited',
    18: 'No entry',
    19: 'General caution',
    20: 'Dangerous curve left',
    21: 'Dangerous curve right',
    22: 'Double curve',
    23: 'Bumpy road',
    24: 'Slippery road',
    25: 'Road narrows on the right',
    26: 'Road work',
    27: 'Traffic signals',
    28: 'Pedestrians',
    29: 'Children crossing',
    30: 'Bicycles crossing',
    31: 'Beware of ice/snow',
    32: 'Wild animals crossing',
    33: 'End speed + passing limits',
    34: 'Turn right ahead',
    35: 'Turn left ahead',
    36: 'Ahead only',
    37: 'Go straight or right',
    38: 'Go straight or left',
    39: 'Keep right',
    40: 'Keep left',
    41: 'Roundabout mandatory',
    42: 'End of no passing',
    43: 'End no passing veh > 3.5 tons'
}

win = tk.Tk()
win.minsize(500, 500)

# functions ---------------------------- v

def uploadImg():
    filename = filedialog.askopenfilename()
    print(filename)
    img = Image.open(filename).convert('RGB')
    img.thumbnail((350, 350)) # resize
    tkImg = ImageTk.PhotoImage(img)

    label.configure(image=tkImg)
    label.image = tkImg

    addPredictButton(img)

def predict(img):
    img = img.resize((32, 32))
    print(img)
    img = np.array(img)
    img = img / 255
    print(img.shape)
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    prediction = np.argmax(prediction)+1
    prediction = classes[prediction]
    print(prediction)

    predictionLabel.configure(text=prediction)
    predictionLabel.text = prediction
    

# UI ----------------------------------- v

label = tk.Label(win)
label.pack()

uploadButton = tk.Button(win, text='Upload image', command=uploadImg)
uploadButton.pack()

predictionLabel = tk.Label(win, text='prediction here')
predictionLabel.pack()

# lambda 

def addPredictButton(img):
    predictButton = tk.Button(win, text='Predict', command=lambda: predict(img))
    predictButton.place(x=50, y=50)

tk.mainloop()