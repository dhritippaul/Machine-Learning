import cv2
import numpy as np
import face_recognition

elon = face_recognition.load_image_file('C:/Users/Asus/Zero to Hero/open-cv/face-detection/face_recog+attendance/elon.jpg')
elon = cv2.cvtColor(elon,cv2.COLOR_BGR2RGB)
Test = face_recognition.load_image_file('C:/Users/Asus/Zero to Hero/open-cv/face-detection/face_recog+attendance/tesla.jpg')
Test = cv2.cvtColor(Test,cv2.COLOR_BGR2RGB)

facLoc = face_recognition.face_locations(elon)[0]
encodeElon = face_recognition.face_encodings(elon)[0]
cv2.rectangle(elon,(facLoc[3],facLoc[0]),(facLoc[1],facLoc[2]),(255,0,255),2)

facLoc = face_recognition.face_locations(Test)[0]
encodeTest= face_recognition.face_encodings(Test)[0]
cv2.rectangle(Test,(facLoc[3],facLoc[0]),(facLoc[1],facLoc[2]),(255,0,255),2)

results = face_recognition.compare_faces([encodeElon],encodeTest)
faceDis = face_recognition.face_distance([encodeElon],encodeTest)
print(results,faceDis)

cv2.putText(Test,f'{results}{round(faceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)

cv2.imshow('elon',elon)
cv2.imshow('tesla',Test)
cv2.waitKey(0)

