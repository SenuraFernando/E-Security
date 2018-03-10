
import numpy as np
import cv2
from keras.preprocessing import image

# -----------------------------
# opencv initialization

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
# -----------------------------
# face expression recognizer initialization
from keras.models import model_from_json

model = model_from_json(open("model.json", "r").read())
model.load_weights ('./models/ai_model.h5')  # load weights

# -----------------------------
names = {
    0: 'Isuru',
    1: 'Pasindu',
    2: 'Senura',
    3: 'Sheelu',

}
# classes = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'senura')


