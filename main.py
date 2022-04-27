import face_recognition
import cv2
import numpy as np

imgElon = face_recognition.load_image_file('img/yourimage.jpg')  # insert your image and path
imgElon = cv2.cvtColor(imgElon,cv2.COLOR_BGR2RGB)
imgTest = face_recognition.load_image_file('img/testingimage.jpg') # insert testing image and path
imgTest = cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)



faceLoc = face_recognition.face_locations(imgElon)[0]
encodeElon = face_recognition.face_encodings(imgElon)[0]
# print(faceLoc)
cv2.rectangle(imgElon,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2) # top, right, bottom, left

faceLocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
# print(faceLoc)
cv2.rectangle(imgTest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,255),2) # top, right, bottom, left



results = face_recognition.compare_faces([encodeElon], encodeTest)
faceDis = face_recognition.face_distance([encodeElon], encodeTest)
print(results,faceDis)
cv2.putText(imgTest,f'{results} {round(faceDis[0],2)} ',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)


cv2.imshow('yourname', imgElon)
cv2.imshow('testname', imgTest)
cv2.waitKey(0)
 
# faceLocTest = face_recognition.face_locations(imgTest)&#91;0]
# encodeTest = face_recognition.face_encodings(imgTest)&#91;0]
# cv2.rectangle(imgTest,(faceLocTest&#91;3],faceLocTest&#91;0]),(faceLocTest&#91;1],faceLocTest&#91;2]),(255,0,255),2)