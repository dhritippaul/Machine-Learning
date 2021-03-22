import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime


path = 'C:/Users/Asus/Zero to Hero/open-cv/face-detection/face_recog+attendance/attendance'
images = []
classNames = []
mylist = os.listdir(path)
# print(mylist)
for cl in mylist:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)

    return encodeList

def markAttendance(name):
    with open('C:/Users/Asus/Zero to Hero/open-cv/face-detection/face_recog+attendance/attend.csv','r+') as f:
        myDataList = f.readlines()
        print(myDataList)
        nameList = []
        for line in myDataList:
            entry=line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtstring = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtstring}')








encodeListKnown = findEncodings(images)
print('Encoding Complete')

cap = cv2.VideoCapture(0)

while True:
    success,img = cap.read()
    imgS = cv2.resize(img,(0,0),None,0.25,0.25)
    imgS = cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB)
    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS,facesCurFrame)


    for encodeFace,faceLoc in zip(encodesCurFrame , facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
        print(faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print(name)
            y1,x2,y2,x1 = faceLoc
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y1-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            markAttendance(name)


    cv2.imshow('Webcam',img)
    cv2.waitKey(1)
 


    key = cv2.waitKey(30)
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()















# facLoc = face_recognition.face_locations(elon)[0]
# encodeElon = face_recognition.face_encodings(elon)[0]
# cv2.rectangle(elon,(facLoc[3],facLoc[0]),(facLoc[1],facLoc[2]),(255,0,255),2)

# facLoc = face_recognition.face_locations(Test)[0]
# encodeTest= face_recognition.face_encodings(Test)[0]
# cv2.rectangle(Test,(facLoc[3],facLoc[0]),(facLoc[1],facLoc[2]),(255,0,255),2)

# results = face_recognition.compare_faces([encodeElon],encodeTest)
# faceDis = face_recognition.face_distance([encodeElon],encodeTest)
# print(results,faceDis)