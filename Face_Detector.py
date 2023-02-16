import cv2
from random import randrange

trained_face_data=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#img = cv2.imread('FB.jpg')
cam = cv2.VideoCapture('Barcelona Football.mp4')
#cam = cv2.VideoCapture(0)


while True:
    
    frame_read,frame=cam.read()

    grayscale_img=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    face_coorinates=trained_face_data.detectMultiScale(grayscale_img)
#print(face_coorinates)

    for (x,y,w,h) in face_coorinates:
#(x, y, w, h) = face_coorinates[2]
        cv2.rectangle(frame,(x,y),(x+w,y+h),(randrange(256),randrange(256),randrange(256)),3)

    cv2.imshow('SC',frame)
    key=cv2.waitKey(1)
    
    if key==113 or key==81:
        break

cam.release()

print('complete!')