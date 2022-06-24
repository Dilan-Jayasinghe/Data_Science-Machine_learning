#importing the open cv library
import cv2
from random import randrange
#loadding trained  data from opencv
trained_face_data = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
#taking a file in from anywhere to read
#img = cv2.imread("RDJ.jpg")
webcam = cv2.VideoCapture(0)
#this is to iterate over each frome of the webcam until ended
while True:
    #this enables the webcam and then stores it as a single image
    susessful_frame_read, frame = webcam.read()
    #need to make the image black and white so it can be anyalysed by the algorihm
    grayscaleimg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #detect faces and face coordinates
    face_coordinates = trained_face_data.detectMultiScale(grayscaleimg)
    #this is a loop that repeats the face detecion for multiple people
    for (x,y,w,h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x+w,y+h), (0, 0,255,20))
    #showing the file
    cv2.imshow("picture test",frame)
    #these are the coorinates of the analysed face (boundry)
    print(face_coordinates)
    #without this line the program will just open then close immediatly the integer indicates the ms refresh rate
    cv2.waitKey(1)
