#importing the open cv library
import cv2
#loading trained  data from opencv
trained_car_data = cv2.CascadeClassifier("car_detector.xml")
trained_pedestrian_data =cv2.CascadeClassifier("pedestrian_detector.xml")
trained_bus_data = cv2.CascadeClassifier("bus_detector.xml")
trained_bike_data = cv2.CascadeClassifier("bike_detector.xml")
#taking a file in from anywhere to read
dashcam = cv2.VideoCapture("dashcam.mp4")
#this is to iterate over each frome of the webcam until a break
while True:
    #this enables the webcam and then stores it as a single image to be fed into the alogorithm
    susessful_frame_read, frame = dashcam.read()
    #need to make the image grayscale so it can be anyalysed by the algorithm
    grayscaleimg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #detect the instances coordinates using trained data
    car_coordinates = trained_car_data.detectMultiScale(grayscaleimg)
    pedestrian_coordinates = trained_pedestrian_data.detectMultiScale(grayscaleimg)
    bus_coordinates = trained_bus_data.detectMultiScale(grayscaleimg)
    bike_coordinates = trained_bike_data.detectMultiScale(grayscaleimg)
    #this is a loop that repeats the face detecion for multiple detected instances
    for (x,y,w,h) in car_coordinates:
        cv2.rectangle(frame, (x, y), (x+w,y+h), (0, 0,255,70))
    for (x,y,w,h) in pedestrian_coordinates:
        cv2.rectangle(frame, (x, y), (x+w,y+h), (0,255,0,70))
    for (x,y,w,h) in bus_coordinates:
        cv2.rectangle(frame, (x, y), (x+w,y+h), (255,0,0,70))
    for (x,y,w,h) in bike_coordinates:
        cv2.rectangle(frame, (x, y), (x+w,y+h), (0,255,255,70))
    #showing the file
    cv2.imshow("picture test",frame)
    #these are the coorinates of the analysed cars and pedestrians (boundry)
    #without this line the program will just open then close immediatly the integer indicates the ms refresh rate
    cv2.waitKey(1)
