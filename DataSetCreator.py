
# Import OpenCV2 for image processing
import cv2

# Load prebuilt model for Frontal Face
detector = cv2.CascadeClassifier('HaarCascade/haarcascade_frontalface_default.xml')

# Initialize and start the video frame capture
camera = cv2.VideoCapture(0)

# Input ID to number Face ID
Id = input('enter your id')

# Initialize sample face image
sampleNum = 0

#Loop
while (True):

    # Read the video frame
    ret, img = camera.read()

    # Colour:   cv2.COLOR_BGR2HSV
    # Gray:     cv2.COLOR_BGR2GRAY
    # Bluish:   cv2.COLOR_BGR2RGB
    # Bluish:   cv2.COLOR_BGR2HLS
    #Full   :   cv2.COLOR_BGR2BGRA
    #cv2.COLOR_RGBA2RGB
    #:COLOR_COLORCVT_MAX =
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

    # Detect frames of different sizes, list of faces rectangles
    faces = detector.detectMultiScale(gray, 1.3, 5)

    # Loops for each faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Incrementing sample number
        sampleNum = sampleNum + 1

        # Saving the captured face in the dataset folder
        cv2.imwrite("abc/User." + Id + '.' + str(sampleNum) + ".jpg", gray[y:y + h, x:x + w])

        # Display the video frame with the boun1
        # ded rectangle
        cv2.imshow('frame', img)

    # wait for 100 miliseconds & If 'q' is pressed, close program
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

    # break if the sample number is morethan 20
    elif sampleNum > 30:
        break

# Stop the camera
camera.release()
# Close all windows
cv2.destroyAllWindows()