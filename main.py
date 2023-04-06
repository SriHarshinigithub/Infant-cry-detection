from keras.models import load_model
from time import sleep
from tensorflow.keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import random
import pywhatkit
import numpy as np
import pyttsx3
import speech_recognition as sr
listner = sr.Recognizer()
s = pyttsx3.init()
def talk(x):
   s.say(x)
   s.runAndWait()

face_classifier = cv2.CascadeClassifier(r'D:\cryingdetection\Emotion_Detection_CNN-main\haarcascade_frontalface_default.xml')
classifier =load_model(r'D:\cryingdetection\Emotion_Detection_CNN-main\model.h5')

emotion_labels = ['angry','disgust','fear','happy','neutral','sad','surprise']

cap = cv2.VideoCapture(0)



while True:
    _, frame = cap.read()
    labels = []
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
        roi_gray = gray[y:y+h,x:x+w]
        roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)



        if np.sum([roi_gray])!=0:
            roi = roi_gray.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi,axis=0)

            prediction = classifier.predict(roi)[0]
            
            label=emotion_labels[prediction.argmax()]
            talk(label)
            if label=="sad":
                talk("your sad let i try to change your mood")
                a=["mrbean","cartoon","tom and jerry"]
                pywhatkit.playonyt(str(random.choice(a)))
                

            
            label_position = (x,y)
            cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        else:
            cv2.putText(frame,'No Faces',(30,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    cv2.imshow('Emotion Detector',frame)
    if cv2.waitKey(1) & 0xFF ==ord('q'):
        break