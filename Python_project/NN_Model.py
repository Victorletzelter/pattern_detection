###This document consists on a neural network model whose aim is to detect ZF patterns in the time series of electrostatic potential
#in the plasma. For executing this file, the package tensorflow has to be installed in the user's system. 

#As decribed in the internship's report, the hyperparameters of the model were optimized using a random grid, for which the code 
#is not presented. 

### Imports

import numpy as np
import pandas as pd
import tensorflow as tf
import warnings
import os
import time
import random

os.chdir("/Users/victorletzelter/Documents/GitHub/pattern_detection/Python_project/Data_files")
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

#Loading the data

data=pd.read_csv("/Users/victorletzelter/Documents/GitHub/pattern_detection/Python_project/Data_files/Data_WE_train_big",delimiter='\t')
data_test=pd.read_csv("/Users/victorletzelter/Documents/GitHub/pattern_detection/Python_project/Data_files/Data_WE_test_big",delimiter='\t')
h=50 #h is the size of each example in the input. 
Nr=1 #Nr is the recoverage parameter. The parameter is such that the starting points of two successive training examples of the time 
#series will be separated from the distance int(h/Nr).

def to_supervised_with_recoverage(data,h=50,Nr=2) : #data is a data frame containing the Signal, and the labelled values of the ZF
#h in the length of the input vectors
#Nr is the recoverage parameters (a divider of h)

#It returns two tensors, that will be the inputs and outputs of the training part of the NN

  X=[]
  Y=[]

  for i in range (0,len(data['S']),int(h/Nr)) :
    if len(data['S'][i:i+h])==h :
      X.append(data['S'][i:i+h])
      Y.append(data['zf'][i:i+h])
  n=np.shape(X)[0]
  M=np.zeros((n-1,h))
  Sortie=np.zeros((n-1,h))
  for i in range (n-1) :
    M[i,:]=np.asmatrix(X[i])
    Sortie[i,:]=np.asmatrix(Y[i])
  In=tf.convert_to_tensor(M)
  Out=tf.convert_to_tensor(Sortie)

  return(In,Out)

def to_time_series_with_recoverage(Ys,h=50,Nr=5) : #Ys is the output of the NN

#The function returns of temporal series, which has the same length as the signal data['S'] initially provided

  n,p=np.shape(Ys)
  Yt=[0]*len(data['S'])
  m=1

  for j in range (0,int(h/Nr)) :
    Yt[j]=Ys[0][j]

  for l in range (1,len(Ys)) :

    m+=1

    for j in range (0,int(h/Nr)) :

      liste=[Ys[l][j]]

      k=1
      while j+k*int(h/Nr)<h and l-k<n and l-k>=0 :

        liste.append(Ys[l-k][j+k*int(h/Nr)])
        k+=1

      Yt[l*int(h/Nr)+j]=np.max(liste)

  for x in range (1,Nr) :

    for j in range (x*int(h/Nr),(x+1)*int(h/Nr)) :

      liste=[Ys[len(Ys)-1][j]]
      k=1

      while j+k*int(h/Nr)<h and l-k<n and l-k>=0 :

        liste.append(Ys[l-k][j+k*int(h/Nr)])
        k+=1

      Yt[(len(Ys)-1)*int(h/Nr)+j]=np.max(liste)

  return(Yt)

###Functions

def save(var,name,namefile) :
    file = open(namefile, "w")
    file.write("%s = %s\n" %(name, var))
    file.close()

h=100 #h is the size of each example in the input. 
Nr=5 #Nr is the recoverage parameter. The parameter is such that the starting points of two successive training examples of the time 
#series will be separated from the distance int(h/Nr).

In,Out=to_supervised_with_recoverage(data,h,Nr)
In2,Out2=to_supervised_with_recoverage(data_test,h,Nr)

Couche1=128
Couche2=256

activation1='selu'
activation2='selu'
kernel1='lecun_normal'
kernel2='lecun_normal'
Dropout=0.004 

model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(units=Couche1, activation=activation1,kernel_initializer=kernel1),
  tf.keras.layers.Dropout(Dropout),
  tf.keras.layers.Dense(units=Couche2, activation=activation2,kernel_initializer=kernel2),
  tf.keras.layers.Dropout(Dropout),
  tf.keras.layers.Dense(h,activation='sigmoid')
])

#A time-decreasing learning rate was also used, but the results were not particularly satisfactory
#lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
#    initial_learning_rate=initial_learning_rate,
#    decay_steps=33046,
#    decay_rate=Decay,staircase=True) #initial_learning_rate * decay_rate ^ (step / decay_steps)
#initial_learning_rate=0.001

earlystopping = tf.keras.callbacks.EarlyStopping(monitor ="val_loss",mode ="min", patience = 100,restore_best_weights = True)

model.compile(
        optimizer='Adam',
        loss= tf.keras.losses.MeanSquaredError(),
        metrics=[tf.keras.metrics.MeanSquaredError()],
    )

checkpoint_path = "/Users/victorletzelter/Desktop/Projet_python/Weights"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback =tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,save_weights_only=True,verbose=1)

hist=model.fit(In, Out,batch_size=256,epochs=300,validation_split=0.8,validation_data=(In2,Out2), callbacks =[earlystopping])

name='Hist'
save(hist.history,'x',name)

Res=model(In2).numpy() #Applying the model on the test set
save(Res,'Res','Res')

n,p=np.shape(Res)
print(list(Res[1]))

ZF=to_time_series_with_recoverage(Res,h,Nr) #converting the output on the NN
name='NRmax{}'.format(Nr) #The value of the recoverage number is kept 
save(ZF,'ZF',name) #The file is saved

###This part is dedicated to the plot of the result of the NN, using the result of the previous algorithm:

import os 
os.chdir("/Users/victorletzelter/Documents/GitHub/pattern_detection/Python_project")
exec(open('Imports.py').read())
data_test=pd.read_csv("/Users/victorletzelter/Documents/GitHub/pattern_detection/Python_project/Data_files/Data_WE_test_big",delimiter='\t')

Nr=5
ListNr=[Nr] 
Listerr=[] #Here, only one error will be added to the list. 

exec(open("/Users/victorletzelter/Documents/GitHub/pattern_detection/Python_project/Data_files/NRMax{}".format(Nr)).read())
ZF=pd.DataFrame(ZF)[0]
ZFp=(ZF>0.5)
Listerr.append(np.sqrt(np.mean((ZFp-data_test['zf'])**2)))

plt.plot(data_test['S'],label='Simulated data')
plt.plot(data_test['zf'],label='Presence of ZF (Truth)')
plt.plot(ZF,label='Presence of ZF (Deduced by the NN)')
plt.grid()
plt.legend()
plt.title('ZF detection with artificial data')
plt.xlabel('Point Number')
plt.ylabel('Potential(V)')
plt.show()