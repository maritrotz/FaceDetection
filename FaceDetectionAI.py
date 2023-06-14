import cv2
import pathlib

# load the required trained XML classifier
cascade_path = pathlib.Path(cv2.__file__).parent.absolute( ) /"data/haarcascade_frontalface_default.xml"

# Classfies cascade path
clf = cv2.CascadeClassifier(str(cascade_path))

# capture frames from desktop camera
camera = cv2.VideoCapture(0)

# loop runs if capturing has been initialized.
while True:
    #reads frames from the camera
    _ ,frame = camera.read()
    #Converts frames to grayscale
    gray = cv2.cvtColor(frame ,cv2.COLOR_BGR2GRAY)
    #Detects faces of different sizes in the camera
    faces = clf.detectMultiScale(gray,scaleFactor=1.1,minNeighbors =5,minSize=(30 ,30),flags =cv2.CASCADE_SCALE_IMAGE)

    for (x, y, width, height) in faces:
        #Draws a rectangle on the face
        cv2.rectangle(frame ,(x, y), ( x +width, y+ height), (0, 0, 255), 4)

#Displays camera image in frame
    cv2.imshow('faces', frame)
    #waiting for 'q' key to stop
    if cv2.waitKey(1) == ord("q"):
        break
#closes camera
camera.release()
#Destroy any associated memory usage
cv2.destroyAllWindows()