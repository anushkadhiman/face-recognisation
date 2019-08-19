import cv2
import matplotlib.pyplot as plt
from skimage.feature import hog
from sklearn.externals import joblib

num_of_sample=200

vid = cv2.VideoCapture(0)
face_cascade=cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')

model=joblib.load('images.pkl')
iter1=0

while(iter1<num_of_sample):
    r,frame=vid.read();
    frame=cv2.resize(frame,(640,480)) #resizing the frame
    im1=cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY) #gray scale conversion of color image
    face=face_cascade.detectMultiScale(im1)
    for x,y,w,h in(face):
        cv2.rectangle(frame,(x,y),(x+w,y+h),[255,0,0],4)
        iter1+=1
        im_f=im1[y:y+h,x:x+w]
        im_f=cv2.resize(im_f,(112,92))
        feat=hog(im_f)
        name=model.predict(feat.reshape(1,-1))
        print(name)
        cv2.putText(frame,'face No. '+str(iter1),(x,y), cv2.FONT_ITALIC, 1,
                    (255,0,255),2,cv2.LINE_AA)
        cv2.imshow('frame',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
vid.release()
cv2.destroyAllWindows()
