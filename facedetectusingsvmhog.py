import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mimg
from sklearn import svm
from sklearn.metrics import confusion_matrix,classification_report
from skimage.feature import hog
from sklearn.externals import joblib

samp=7
train_data=np.zeros((7*40,8748))
train_label=np.zeros((7*40))
count=-1
plt.figure(1)
plt.ion()
for i in range(1,41):
    for j in range(1,samp+1):
        plt.cla()
        count=count+1
        path='C:/Users/AD/Documents/My documents/Downloads/orl_face1/orl_face/u%d/%d.png'%(i,j)
        im=mimg.imread(path)
        feat=hog(im)
        train_data[count,:]=feat
        train_label[count]=i

test_data=np.zeros(((10-samp)*40,8748))
test_label=np.zeros((3*40))
count=-1
for i in range(1,41):
    for j in range(samp+1,11):
        plt.cla()
        count+=1
        path='C:/Users/AD/Documents/My documents/Downloads/orl_face1/orl_face/u%d/%d.png'%(i,j)
        im=mimg.imread(path)
        feat=hog(im)
        test_data[count,:]=feat
        test_label[count]=i

svm1=svm.SVC(kernel='linear',C=1)
svm2=svm.SVC(kernel='rbf',C=1)
svm3=svm.SVC(kernel='poly',C=1)

y_pred_svm1=svm1.fit(train_data,train_label).predict(test_data)
cnf_matrix_svm1=confusion_matrix(test_label,y_pred_svm1)
print(cnf_matrix_svm1)
acc1=svm1.score(test_data,test_label)

joblib.dump(svm1,'images.pkl')

y_pred_svm2=svm2.fit(train_data,train_label).predict(test_data)
cnf_matrix_svm2=confusion_matrix(test_label,y_pred_svm2)
print(cnf_matrix_svm2)
acc2=svm2.score(test_data,test_label)

y_pred_svm3=svm3.fit(train_data,train_label).predict(test_data)
cnf_matrix_svm3=confusion_matrix(test_label,y_pred_svm3)
print(cnf_matrix_svm3)
acc3=svm3.score(test_data,test_label)

print(acc1)
print(acc2)
print(acc3)

'''
y1=[acc1]
y2=[acc2]
y3=[acc3]
x1=['a1']
x2=['a2']
x3=['a3']
tl=['linear','rbf','poly']
plt.bar(x1,y1,width=0.2,color=['red'])
plt.bar(x2,y2,width=0.2,color=['green'])
plt.bar(x3,y3,width=0.2,color=['blue'])
plt.xlabel('x - axis') 
plt.ylabel('y - axis') 
plt.title('Accuracy bar chart!') 
plt.tight_layout()
plt.legend(('linear','rbf','poly'))
plt.show() 
'''