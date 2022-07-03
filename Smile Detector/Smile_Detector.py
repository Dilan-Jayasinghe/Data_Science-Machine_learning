#importing the Open CV library
import cv2
import numpy as np
from random import randrange
#loading trained data from Open CV
trained_face_data = cv2.CascadeClassifier("Smile Detector\haarcascade_frontalface_default.xml")
trained_smile_data = cv2.CascadeClassifier("Smile Detector\haarcascade_smile.xml")
trained_eye_data = cv2.CascadeClassifier("Smile Detector\haarcascade_eye.xml")
#taking a file in from anywhere to read
webcam = cv2.VideoCapture(0)
#this is to iterate over each frome of the webcam until a break
while True:
    #this enables the webcam and then stores it as a single image to be fed into the alogorithm
    susessful_frame_read, frame = webcam.read()
    #change the scale factor of fx and fy to increase window size
    frame  = cv2.resize(frame, (0, 0), fx=1.5, fy=1.5)
    #need to make the image grayscale so it can be anyalysed by the algorithm
    grayscaleimg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #detect faces and face coordinates
    face_coordinates = trained_face_data.detectMultiScale(grayscaleimg)
    #this is a loop that repeats the face detecion for multiple instances
    for (x,y,w,h) in face_coordinates:
        #within the frame of the face draw a rectangle using the fram coordinates
        cv2.rectangle(frame, (x, y), (x+w,y+h), (0, 0,255,20))
        #splcing the whole image to just focus on the face for the smile algorithm
        face_boundaryimg= frame[y:y+h, x:x+w]
        #need to make it greyscale first
        face_grayscaleimg = cv2.cvtColor(face_boundaryimg, cv2.COLOR_BGR2GRAY)
        #applying the trained smile data to the localised image
        smile = trained_smile_data.detectMultiScale(face_grayscaleimg, scaleFactor=1.7 , minNeighbors= 50)
        eye = trained_eye_data.detectMultiScale(face_grayscaleimg, scaleFactor=1.1, minNeighbors= 5)
        #search within boundry of face for smile
        for (x_,y_,w_,h_) in smile:
            cv2.rectangle(    face_boundaryimg, (x_ ,y_ ), (x_ + w_, y_ + h_ ),(0,250,0),2)
        for (x_,y_,w_,h_) in eye:
            cv2.rectangle(    face_boundaryimg, (x_ ,y_ ), (x_ + w_, y_ + h_ ),(250,0,0),1)
        #if the length of the smile array is greater than 0 (smile present) add text
        if len(smile) > 0:
            cv2.putText(frame,"smiling",(x, y+h+40),fontScale=1,fontFace=cv2.FONT_HERSHEY_PLAIN,color =(0,250,0), thickness = 2)
    #showing the file
    cv2.imshow("picture test",frame )
    #these are the coorinates of the analysed face (boundry)
    print(face_coordinates)
    #without this line the program will just open then close immediatly the integer indicates the ms refresh rate
    cv2.waitKey(1)
