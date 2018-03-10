
import numpy as np
import cv2
from keras.preprocessing import image

# -----------------------------
# opencv initialization

face_cascade = cv2.CascadeClassifier('./HaarCascade/haarcascade_frontalface_default.xml')

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

while (True):
    ret, img = cap.read()
    # img = cv2.imread('C:/Users/IS96273/Desktop/hababam.jpg')

    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    colour= cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    faces = face_cascade.detectMultiScale(colour, 1.3, 5)

        # print(faces) #locations of detected faces

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)  # draw rectangle to main image

        detected_face = img[int(y):int(y + h), int(x):int(x + w)]  # crop detected face
        detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY)  # transform to gray scale
        detected_face = cv2.resize(detected_face, (48, 48))  # resize to 48x48

        img_pixels = image.img_to_array(detected_face)
        img_pixels = np.expand_dims(img_pixels, axis=0)

        img_pixels /= 255  # pixels are in scale of [0, 255]. normalize all pixels in scale of [0, 1]

        predictions = model.predict(img_pixels)  # store probabilities of 7 expressions

        # find max indexed array 0: angry, 1:disgust, 2:fear, 3:happy, 4:sad, 5:surprise, 6:neutral
        max_index = np.argmax(predictions[0])

        names = names[max_index]

        # write emotion text above rectangle
        cv2.putText(img, names, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # process on detected face end
    # -------------------------

    cv2.imshow('img', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # press q to quit
        break

# kill open cv things
cap.release()
cv2.destroyAllWindows()
