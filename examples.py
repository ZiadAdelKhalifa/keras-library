


#EX:3.2.1
"""
import tensorflow as tf
minst=tf.keras.datasets.mnist
(xtrain,ytrain),(xtest,ytest)=minst.load_data()

xtrain=xtrain/255.0
xtest=xtest/255.0

model=tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512,activation=tf.nn.relu),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10,activation=tf.nn.sigmoid),
])
model.compile(optimizer="adam",
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

model.fit(xtrain,ytrain,epochs=3)
model.evaluate(xtest,ytest)
"""

#EX:3.2.1:


"""
import tensorflow as tf
minst=tf.keras.datasets.mnist

(xtrain,ytrain),(xtest,ytest)=minst.load_data()

print(xtrain[0])

import matplotlib.pyplot as plt

plt.imshow(xtrain[0],cmap=plt.cm.binary)
plt.show()

print(ytrain[0])

#عمل تسوية للبيانات و عرض الصورة و البيانات بعد التسوية
xtrain=tf.keras.utils.normalize(xtrain,axis=1)

xtest=tf.keras.utils.normalize(xtest,axis=1)

plt.imshow(xtrain[0],cmap=plt.cm.binary)
plt.show()

print(xtrain[0])

#built the model
model=tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10,activation=tf.nn.softmax))

model.compile(optimizer="adam",
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])


model.fit(xtrain,ytrain,epochs=3)

val_loss,val_acc=model.evaluate(xtest,ytest)

print("the loss is :",val_loss,"\n","the accuracy is :",val_acc)

#to save the model

model.save('ex2.model')
#load it
new_model=tf.keras.models.load_model('ex2.model')
prediction=new_model.predict(xtest)
print(prediction)

#show some numbers

import numpy as np
print(np.argmax(prediction[0]))
plt.imshow(xtest[0],cmap=plt.cm.binary)
plt.show()
"""


#EX:3.2.3
"""
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

fastion_minst=keras.datasets.fashion_mnist

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']   


(xtrain_images,ytrain_lables),(xtest_images,ytest_lables)=fastion_minst.load_data()

print(xtrain_images.shape)
print(len(ytrain_lables))


print(xtest_images.shape)
print(len(ytest_lables))

#normalization
xtrain_images=xtrain_images/255.0
xtest_images=xtest_images/255.0

#show the plot of some values

plt.imshow(xtest_images[15],cmap='gray')
plt.colorbar()
plt.grid()
plt.show()



#show 25 photoes
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid()
    plt.imshow(xtrain_images[i],cmap=plt.cm.binary)
    plt.xlabel(class_names[ytrain_lables[i]])
plt.show()

#bulit the neural network

model=keras.Sequential([
        keras.layers.Flatten(),
        keras.layers.Dense(128,activation=tf.nn.relu),
        keras.layers.Dense(10,activation=tf.nn.sigmoid)
])
model.compile(optimizer='adam',
                metrics=['accuracy'],
                loss='sparse_categorical_crossentropy')
model.fit(xtrain_images,ytrain_lables,epochs=5)

test_loss,test_acc=model.evaluate(xtest_images,ytest_lables)

print("the accuracy is :",test_acc)
print("the loss is :",test_loss)

prediction=model.predict(xtest_images)
print(prediction[0])
print(np.argmax(prediction[0]))
print(ytest_lables[0])

#plot image
def plot_image(i,prediction,true_val,img):
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(img[i],cmap=plt.cm.binary)
    if np.argmax(prediction[i])==true_val[i]:
        color='blue'
    else:
        color='red'
    plt.xlabel("{} {:2.0f}% ({})".format(class_names[np.argmax(prediction[i])],
                                            100*np.max(prediction[i]),
                                            class_names[true_val[i]]),color=color)
    
def plot_array(i,pred,truee):
    thisplot=plt.bar(range(10),pred[i],color='#777777')
    plt.xticks([])
    plt.xticks(range(10), class_names, rotation=45)
    plt.yticks([])
    plt.grid(False)
    plt.ylim([0, 1])
    pred_lable=np.argmax(pred[i])
    thisplot[pred_lable].set_color('red')
    thisplot[truee[i]].set_color('blue')

i=0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i,prediction,ytest_lables,xtest_images)
plt.subplot(1,2,2)
plot_array(i,prediction,ytest_lables)
plt.show()

i=120
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i,prediction,ytest_lables,xtest_images)
plt.subplot(1,2,2)
plot_array(i,prediction,ytest_lables)
plt.show()

i=1245
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i,prediction,ytest_lables,xtest_images)
plt.subplot(1,2,2)
plot_array(i,prediction,ytest_lables)
plt.show()


i=2000
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i,prediction,ytest_lables,xtest_images)
plt.subplot(1,2,2)
plot_array(i,prediction,ytest_lables)
plt.show()


i=500
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i,prediction,ytest_lables,xtest_images)
plt.subplot(1,2,2)
plot_array(i,prediction,ytest_lables)
plt.show()

i=200
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i,prediction,ytest_lables,xtest_images)
plt.subplot(1,2,2)
plot_array(i,prediction,ytest_lables)
plt.show()


"""
#EX:3.2.4



from tabnanny import verbose
import pandas as pd
import numpy as np # linear algebra
import matplotlib.pyplot as plt 
import seaborn as sns
sns.set(style='whitegrid')
import os
import cv2
import glob as gp
import tensorflow as tf
from tensorflow import keras


#for kaggle path
trainpath='../input/intel-image-classification/seg_train/'
testpath='../input/intel-image-classification/seg_test/'
predpath='../input/intel-image-classification/seg_pred/'


#enter each folder and know how many images it have in the training 
for folder in os.listdir(trainpath + 'seg_train'): #os.listdir:"not save the path but only arrive to the folders that ended this path"
    #note i write seg-train again as "لانه الفولدر مكرر في الداتا "and so with rest and pred
    files=gp.glob(pathname=str(trainpath + 'seg_train//'+folder+'/*.jpg')) #gp.glop :i told him to get all images ended with jpg
    print(f'for training data found : {len(files)} in folder :{folder}')
    
#enter each folder and know how many images it have in the testing 
for folder in os.listdir(testpath + 'seg_test'): #note i wrire seg-train again as "لانه الفولدر مكرر في الداتا "and so with rest and pred
    files=gp.glob(pathname=str(testpath + 'seg_test//'+folder+'/*.jpg')) #gp.glop :i told him to get all images ended with jpg
    print(f'for testpath data found : {len(files)} in folder :{folder}')
    
    
#enter each folder and know how many images it have in the predect 
#this no folder in the predect it only some random images

files=gp.glob(pathname=str(predpath + 'seg_pred//*.jpg')) #gp.glop :i told him to get all images ended with jpg
print(f'for predpath data found : {len(files)} photos')
    

#prepare data for classification
code={'buildings':0,'forest':1,'glacier':2,'mountain':3,'sea':4,'street':5}
def getcode(n):
    for x,y in code.items():
        if n==y:
            return x


#to find image size of each image in each folder in the train data 
size=[]
for folder in os.listdir(trainpath+'seg_train'):
    files=gp.glob(pathname=str(trainpath +'seg_train//' +folder +'/*.jpg')) #"هنا يعتبر مسكت اول فولدر"
    for file in files: #"هنا هتمسك الصور صورة صورة"
        image=cv2.imread(file)
        size.append(image.shape)
#print(pd.Series(size).value_counts())  #will result number of images in teh same size


#to find image size of each image in each folder in the test data 
size=[]
for folder in os.listdir(testpath+'seg_test'):
    files=gp.glob(pathname=str(testpath +'seg_test//' +folder +'/*.jpg')) #"هنا يعتبر مسكت اول فولدر"
    for file in files: #"هنا هتمسك الصور صورة صورة"
        image=cv2.imread(file)
        size.append(image.shape)
#print(pd.Series(size).value_counts())  #will result number of images in teh same size



#to find image size of each image in each folder in the pred data 
size=[]
files=gp.glob(pathname=str(predpath + 'seg_pred//*.jpg')) #gp.glop :i told him to get all images ended with jpg
for file in files: #"هنا هتمسك الصور صورة صورة"
        image=cv2.imread(file)
        size.append(image.shape)
#print(pd.Series(size).value_counts()) 


#remember folders are seprated to train and test and predect

#resize images and append it to an array
s=100
xtrain=[]
ytrain=[]

for folder in os.listdir(trainpath+'seg_train'):
    files=gp.glob(pathname=str(trainpath +'seg_train//' +folder +'/*.jpg')) #"هنا يعتبر مسكت اول فولدر"
    for file in files: #"هنا هتمسك الصور صورة صورة"
        image=cv2.imread(file)
        image_array=cv2.resize(image,(s,s))
        xtrain.append(list(image_array))
        ytrain.append(code[folder])

print(f'we have {len(xtrain)} items in xtrain')
"""
#plot some figures 
plt.figure(figsize=(20,20))
for n ,i in enumerate (list(np.random.randint(0,len(xtrain),36))):
    plt.subplot(6,6,n+1)
    plt.imshow(xtrain[i])
    plt.axis('off')
    plt.title(getcode(ytrain[i]))
"""
xtest=[]
ytest=[]

for folder in os.listdir(testpath+'seg_test'):
    files=gp.glob(pathname=str(testpath +'seg_test//' +folder +'/*.jpg')) #"هنا يعتبر مسكت اول فولدر"
    for file in files: #"هنا هتمسك الصور صورة صورة"
        image=cv2.imread(file)
        image_array=cv2.resize(image,(s,s))
        xtest.append(list(image_array))
        ytest.append(code[folder])

print(f'we have {len(xtest)} items in x test')

#plot some figures 
"""
plt.figure(figsize=(20,20))
for n ,i in enumerate (list(np.random.randint(0,len(xtest),36))):
    plt.subplot(6,6,n+1)
    plt.imshow(xtest[i])
    plt.axis('off')
    plt.title(getcode(ytest[i]))
"""
xpred=[]
files=gp.glob(pathname=str(predpath + 'seg_pred//*.jpg')) #"هنا يعتبر مسكت اول فولدر"
for file in files: #"هنا هتمسك الصور صورة صورة"
    image=cv2.imread(file)
    image_array=cv2.resize(image,(s,s))
    xpred.append(list(image_array))

print(f'we have {len(xpred)} items in xpredect')

"""
#plot some figures 
plt.figure(figsize=(20,20))
for n ,i in enumerate (list(np.random.randint(0,len(xpred),36))):
    plt.subplot(6,6,n+1)
    plt.imshow(xpred[i])
    plt.axis('off')
    
"""

xtrain=np.array(xtrain)
ytrain=np.array(ytrain)
xtest=np.array(xtest)
ytest=np.array(ytest)
xpred=np.array(xpred)

model=tf.keras.models.Sequential([
    #keras.layers.conv2D(num of filters,filter size,activation function,input_shape=(s,s,3 as it is rgb))
    keras.layers.Conv2D(200,kernel_size=(3,3),activation='relu',input_shape=(s,s,3)),
    #to calculate num of parametaras of a layer=[(wedith of filter size *height *num of dimentions which is 3 as it is rgb)+1] *num of filters
    # [(3*3*3)+1]*200 
    keras.layers.Conv2D(150,kernel_size=(3,3),activation='relu'),
    keras.layers.MaxPool2D(4,4), #size of filter used in max pool 
    keras.layers.Conv2D(120,kernel_size=(3,3),activation='relu'),
    keras.layers.Conv2D(80,kernel_size=(3,3),activation='relu'),
    keras.layers.Conv2D(50,kernel_size=(3,3),activation='relu'),
    keras.layers.MaxPool2D(4,4),
    #here we finich CNN convolotion neural network and start a normal neural network
    keras.layers.Flatten(),
    keras.layers.Dense(120,activation='relu'),
    keras.layers.Dense(80,activation='relu'),
    keras.layers.Dense(50,activation='relu'),
    keras.layers.Dropout(rate=0.5),
    keras.layers.Dense(6,activation='softmax'),

])

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy')


print('model detales are:',model.summary())

this_model=model.fit(xtrain,ytrain,epochs=5,batch_size=64,verbose=1)
#batch_size=64 :means each epochs work with 64 image not all images in one epochs,verbose:contro; how much detal will be shown


modelloss,modelaccuracy=model.evaluate(xtest,ytest)
print('loss of the test is : {}'.format(modelloss))
print('accuracy of the test is : {}'.format(modelaccuracy))

yresult=model.predict(xpred)

print('predection shape is {}'.format(yresult.shape))

plt.figure(figsize=(20,20))
for n ,i in enumerate (list(np.random.randint(0,len(xpred),36))):
    plt.subplot(6,6,n+1)
    plt.imshow(xpred[i])
    plt.axis('off')
    plt.title(getcode(np.argmax(yresult[i])))




