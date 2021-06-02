import os    
import warnings 
warnings.filterwarnings('always') 
warnings.filterwarnings('ignore') 

import cv2 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import LabelEncoder 
from sklearn.model_selection import train_test_split 

from keras.models import Sequential 
from keras.layers import Dense 
from keras.optimizers import Adam 
from keras.utils import to_categorical 
from keras.preprocessing.image import ImageDataGenerator 

import tensorflow as tf  
from tqdm import tqdm 
from keras.layers import Flatten, Activation 
from keras.layers import Conv2D, MaxPooling2D 


X=[] 
Z=[] 

#Reading data (image, label) from input file and save it in lists (X,Z) 
def  savedata(ftype, directory): 
  for i in tqdm(os.listdir(directory)): 
    path = os.path.join(directory,i) 
    i = cv2.imread(path, cv2.IMREAD_COLOR) 
    if i is not None: 
      i = cv2.resize(i, (150, 150)) 
      X.append(np.array(i)) 
      Z.append(ftype) 

flowers=["daisy", "dandelion", "rose", "tulip", "sunflower"] 
for i in flowers: 
  savedata(i,'../input/flowers/flowers/'+i) 

#One-hot-encoding 
encoder=LabelEncoder() 
Y=encoder.fit_transform(Z) 
Y=to_categorical(Y,5) 
X=np.array(X) 
X=X/255 

#Splitting data as training and test sets 
x_train, x_test, y_train, y_test=train_test_split(X, Y, test_size=0.2) 

#Building a CNN 
model = Sequential() 
model.add(Conv2D(filters = 32, kernel_size = (7,7),padding = 'Same', activation ='relu', input_shape = (150,150,3))) 
model.add(MaxPooling2D(pool_size=(2,2))) 
model.add(Conv2D(filters = 64, kernel_size = (5,5),padding = 'Same', activation ='relu'))  
model.add(Conv2D(filters =96, kernel_size = (3,3),padding = 'Same',activation ='relu')) 
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2))) 
model.add(Conv2D(filters = 128, kernel_size = (3,3),padding = 'Same',activation ='relu')) 
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2))) 
model.add(Flatten()) 
model.add(Dense(512)) 
model.add(Activation('relu')) 
model.add(Dense(5, activation = "softmax")) 

b_size=100 
epochs=30 

#Data generation 
datagen = ImageDataGenerator( 
        rotation_range =20,  
        zoom_range = 0.1,  
        width_shift_range=0.2,  
        height_shift_range=0.2,  
        horizontal_flip=True )  
datagen.fit( x_train ) 

#Compile model 
model.compile(optimizer=Adam(lr=0.0005), loss='categorical_crossentropy', metrics=['accuracy']) 

# Train model on data 
History = model.fit_generator (datagen.flow (x_train, y_train, batch_size=b_size), 			
          epochs = epochs, validation_data = (x_test,y_test), verbose = 0, steps_per_epoch=int(x_train.shape[0]/ batch_size)) 

# Loss curve 
plt.plot(History.history['loss']) 
plt.plot(History.history['val_loss']) 
plt.title('Loss Curve') 
plt.ylabel('Loss') 
plt.xlabel('No. of Epochs') 
plt.legend(['train', 'test']) 
plt.show() 

#Accuracy curve 
plt.plot(History.history['acc']) 
plt.plot(History.history['val_acc']) 
plt.title('Accuracy Curve') 
plt.ylabel('Accuracy') 
plt.xlabel('No. of Epochs') 
plt.legend(['train', 'test'])
plt.show() 

#Predicting classes on test set 
x_pred = model.predict(x_test) 
predicted=np.argmax(x_pred, axis=1) 

# separate images into correct and incorrect predictions 
hits=[] 
miss=[] 
for i in range(len(y_test)): 
  if(np.argmax(y_test[i])==predicted[i]): 
    hits.append(i) 
  if(len(hits)==8): 
    break 

for i in range(len(y_test)): 
  if(not np.argmax(y_test[i])==predicted[i]): 
    miss.append(i) 
  if(len(miss)==8): 
    break 

#Correctly classified images 
count=0 
fig,ax=plt.subplots(2,2) 
fig.set_size_inches(14,14) 
for i in range (2): 
  for j in range (2): 
    ax[i,j].imshow(x_test[hits[count]]) 
    ax[i,j].set_title("Model:"+str(encoder.inverse_transform([predicted[hits[count]]]))+
                "\n"+"Actual:"+str(encoder.inverse_transform(np.argmax([y_test[hits[count]]])))) 
    plt.tight_layout() 
    count+=1 

#Incorrectly classified images 
count=0 
fig,ax=plt.subplots(2,2) 
fig.set_size_inches(14,14) 
for i in range (2): 
  for j in range (2): 
    ax[i,j].imshow(x_test[miss[count]]) 
    ax[i,j].set_title("Model:"+str(encoder.inverse_transform([predicted[miss[count]]]))              +"\n Actual:"+str(encoder.inverse_transform(np.argmax([y_test[miss[count]]])))) 
    plt.tight_layout() 
    count+=1 