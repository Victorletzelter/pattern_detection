### Paragraph 4

plt.plot(data1['Time(ms)'],data1['Potential(V)'],label='Probe B')
plt.plot(data2['Time(ms)'],data2['Potential(V)'],label='Probe D')
plt.axvline(1063,color='red',linestyle='--')
plt.axvline(1150,color='red',linestyle='--')
plt.axvline(1222,color='red',linestyle='--')
plt.xlabel('Time(ms)')
plt.ylabel('Floating Potential (V)')
plt.title('Evolution of the floating potential during the experiment')
plt.legend()
plt.grid()
plt.show()
plt.close()

### Paragraph 6

#Portion of the data
plt.plot(D['Time(ms)'],D['Potential(V)'],label='Probe B')
plt.plot(D2['Time(ms)'],D2['Potential(V)'],label='Probe D')
#plt.axvline(1063,color='red',linestyle='--')
#plt.axvline(1150,color='red',linestyle='--')
#plt.axvline(1222,color='red',linestyle='--')
plt.xlabel('Time(ms)')
plt.ylabel('Floating Potential (V)')
plt.title('Portion of the data at stake')
plt.legend()
plt.grid()
plt.show()
plt.close()

### Paragraph 6.1

D=split(data1,x1,x2)
D2=split(data2,x1,x2)

Da_s=smooth2(D,D2,x1,x2) #in RMS_gestion
D_s=Da_s[0]
D2_s=Da_s[1]

plt.plot(D['Time(ms)'],D['Potential(V)'],label='Orginal')
plt.plot(D_s['Time(ms)'],D_s['Potential(V)'],color='blue',label='Smoothed')
#plt.axvline(1063,color='red',linestyle='--')
#plt.axvline(1150,color='red',linestyle='--')
#plt.axvline(1222,color='red',linestyle='--')
plt.xlabel('Time(ms)')
plt.ylabel('Floating Potential (V)')
plt.title('Evolution of the floating potential during the experiment')
plt.legend()
plt.grid()
plt.show()
plt.close()

Da=smooth(D,D2,x1,x2) #in RMS_gestion
D=Da[0]
D2=Da[1]

plt.plot(D['Time(ms)'],D['Potential(V)'],label='Probe B?')
plt.plot(D2['Time(ms)'],D2['Potential(V)'],label='Probe D?')
#plt.axvline(1063,color='red',linestyle='--')
#plt.axvline(1150,color='red',linestyle='--')
#plt.axvline(1222,color='red',linestyle='--')
plt.xlabel('Time(ms)')
plt.ylabel('Floating Potential (V)')
plt.title('Evolution of the floating potential during the experiment')
plt.legend()
plt.grid()
plt.show()
plt.close()

### Paragraph 6.2

Da=smooth(D,D2,x1,x2) #in RMS_gestion
D=Da[0]
D2=Da[1]

Fs=2e6
f, t, Sxx = signal.spectrogram(D['Potential(V)'], Fs, window=('gaussian',F))
n=N

f=f[0:n]
Sxx=Sxx[0:n,:]
fig, ax = plt.subplots(4, figsize=(8, 7))

im=ax[0].pcolormesh(t, f, Sxx,shading='auto')
ax[0].set_ylabel('Frequency [Hz]')
ax[0].set_xlabel('Time [sec]')
#fig.colorbar(im, ax=ax[0])

f2, t2, Sxx2 = signal.spectrogram(D2['Potential(V)'], Fs,window=('gaussian',F))
f2=f2[0:n]
Sxx2=Sxx2[0:n,:]
im2=ax[1].pcolormesh(t2, f2, Sxx2,shading='auto')
#ax[1].colorbar(ax=ax[0])
ax[1].set_ylabel('Frequency [Hz]')
ax[1].set_xlabel('Time [sec]')
#fig.colorbar(im2, ax=ax[1])

ax[0].get_shared_x_axes().join(ax[0], ax[1])
ax[0].get_shared_y_axes().join(ax[0], ax[1])

t3=np.linspace(0,t[len(t)-1],len(D['Potential(V)']))
ax[2].plot(t3,D['Potential(V)'])
ax[2].plot(t3,D2['Potential(V)'])
ax[2].grid()
ax[2].set_ylabel('Potiential[V]')
ax[2].set_xlabel('Time [sec]')
ax[2].get_shared_x_axes().join(ax[2], ax[1])

Mins_6=pd.DataFrame([0]*len(D2['Potential(V)']),index=D.index)

for e in indMin_6 :
    Mins_6[0][e]=D2['Potential(V)'][e]

ax[2].plot(t3,Mins_6[0],'bx')

Max_6=pd.DataFrame([0]*len(D2['Potential(V)']),index=D.index)

for e in indMax_6 :
    Max_6[0][e]=D2['Potential(V)'][e]

ax[2].plot(t3,Max_6[0],'bx')

#for i in range (0,len(filtre_max[0])) :
#    x=indMax_6[i]
#    x=((x-450000)/49999)*t3[len(t3)-1]
#    y=min(Sorties_max[i],Sorties_max_2[i])
#    X=np.arange(x,x+10*(t3[1]-t3[0])*(y+1),t3[1]-t3[0])
#    ax[2].fill_between(X,-30,30, color='blue', alpha=0.1)

#for i in range (0,len(filtre_min[0])) :
#    x=indMin_6[i]
#    x=((x-450000)/49999)*t3[len(t3)-1]
#    y=min(Sorties_min[i],Sorties_min_2[i])
#    X=np.arange(x,x+10*(t3[1]-t3[0])*(y+1),t3[1]-t3[0])
#    ax[2].fill_between(X,-30,30, color='red', alpha=0.1)

SXX=[Sxx[:,t] for t in range (0,len(t))]
SYY=[Sxx2[:,t] for t in range (0,len(t))]

C=[0]*len(t)

for j in range (len(t)) :
    C[j]=np.dot(SXX[j],SYY[j])


t4=np.linspace(0,t[len(t)-1],len(t))
ax[3].plot(t4,C)
ax[3].grid()
ax[3].set_ylabel('Spectral coherence')
ax[3].set_xlabel('Time [sec]')
#ax[3].plot(t4,[np.mean(C)+2*np.std(C)]*len(t4),linestyle='--',color='red')
ax[3].plot(t4,[np.mean(C)]*len(t4),linestyle='--',color='red')
ax[3].get_shared_x_axes().join(ax[3], ax[2])

plt.show()

### Paragraph 6.3

indMin, _=find_peaks(-D['Potential(V)']+max(abs(D['Potential(V)'])),height=0)    #indMin : indices des minimums locaux
Mins=[D['Potential(V)'][e+x1] for e in indMin]
Mins_2=[D['Potential(V)'][e] for e in indMin_2]
Mins_3=[D['Potential(V)'][e] for e in indMin_3]

Mins=pd.DataFrame(Mins)
Mins_2=pd.DataFrame(Mins_2)
Mins_3=pd.DataFrame(Mins_3)

Mins=Mins.set_index(indMin+x1)
Mins_2=Mins_2.set_index(np.array(indMin_2))
Mins_3=Mins_3.set_index(np.array(indMin_3))


plt.plot(Mins_2,'bx',color='red',label='Mins after the coherence filtration')
plt.plot(Mins_3,'bx',color='black',label='Mins after two filtrations')

S=100
data=D
data_m=data.rolling(S).mean()
data_rms=np.sqrt(data.pow(2).rolling(S).apply(lambda x: np.sqrt(x.mean())))
data_sd=data.rolling(S).std()
#sup=data_m['Potential(V)']+2*data_sd['Potential(V)']
#inf=data_m['Potential(V)']-2*data_sd['Potential(V)']

plt.plot(D['Potential(V)'])
plt.plot(data_m['Potential(V)']+2.5*data_sd['Potential(V)'],color='black',linewidth=0.5)
plt.plot(data_m['Potential(V)']-2.5*data_sd['Potential(V)'],color='black',linewidth=0.5)
plt.fill_between(np.arange(450000,500000),data1_m['Potential(V)']-2.5*data_sd['Potential(V)'] ,data_m['Potential(V)']+2.5*data1_sd['Potential(V)'], color='blue', alpha=0.2)
plt.plot(data_m['Potential(V)'],linestyle='--',color='black',linewidth=0.5)
plt.xlabel('Point number')
plt.ylabel('Potential(V)')
#plt.legend()
plt.title('Filtration based on the Bollinger Bands of the signal')
plt.show()
plt.close()

### Paragaph 6.4

indMin, _=find_peaks(-D['Potential(V)']+max(abs(D['Potential(V)'])),height=0)    #indMin : indices des minimums locaux
Mins=[D['Potential(V)'][e+x1] for e in indMin]
Mins_2=[D['Potential(V)'][e] for e in indMin_2]
Mins_3=[D['Potential(V)'][e] for e in indMin_3]
Mins_4=[D['Potential(V)'][e] for e in indMin_4]

Mins=pd.DataFrame(Mins)
Mins_2=pd.DataFrame(Mins_2)
Mins_3=pd.DataFrame(Mins_3)
Mins_4=pd.DataFrame(Mins_4)

Mins=Mins.set_index(indMin+x1)
Mins_3=Mins_3.set_index(np.array(indMin_3))
Mins_4=Mins_4.set_index(np.array(indMin_4))

#plt.plot(Mins_2,'bx',color='red',label='Mins after the coherence filtration')
plt.plot(Mins_3,'bx',color='red',label='Mins after two filtration')
plt.plot(Mins_4,'bx',color='black',label='Mins after three filtration')

plt.plot(D['Potential(V)'])
plt.grid()
plt.xlabel('Point number')
plt.ylabel('Potential(V)')
plt.legend()
plt.title('Filtration based on the Non-coverage')
plt.show()
plt.close()

### Paragraph 6.5
# MIN

e=453896
data=D

E=0
min_index,l_err,RMSE=opt(e,-1,data)
X=np.arange(0,10+10*min_index,1)
Dat=[data['Potential(V)'][e+i] for i in np.arange(0,10+10*min_index,1)]

with warnings.catch_warnings():

    warnings.simplefilter("error", OptimizeWarning)
    warnings.simplefilter("error", RuntimeError)

    try:
        popt,pcov = scipy.optimize.curve_fit(lambda t,a,b,c: a*np.exp(b*t)+c,  X,  Dat, p0=(-1,-1/100,1),bounds=((-np.inf,-1/20,-np.inf),(0,-1/300,np.inf)),maxfev=800)
        a_er=np.sqrt(np.diag(pcov))[0]
        b_er=np.sqrt(np.diag(pcov))[1]
        c_er=np.sqrt(np.diag(pcov))[2]

    except OptimizeWarning:
        E=1
        print('Maxed out calls.')

    except RuntimeError:
        print("Error - curve_fit failed")
        E=1

    except RuntimeWarning :
        print("Invalid value")
        E=1

    except Warning :
        print('Warning was raised as an exception!')
        E=1

    except FloatingPointError :
        print('Warning was raised as an exception!')
        E=1

plt.plot(X,f(X,popt),label='Exponental fit for probe B',linestyle='--')
plt.plot(Dat,color='blue',label='Potential of Probe B')

data=D2
min_index,l_err,RMSE=opt(e,1,data)
X=np.arange(0,10+10*min_index,1)
Dat=[data['Potential(V)'][e+i] for i in np.arange(0,10+10*min_index,1)]

e=453896
data=D2

E=0
min_index,l_err,RMSE=opt(e,-1,data)
X=np.arange(0,10+10*min_index,1)
Dat=[data['Potential(V)'][e+i] for i in np.arange(0,10+10*min_index,1)]

with warnings.catch_warnings():

    warnings.simplefilter("error", OptimizeWarning)
    warnings.simplefilter("error", RuntimeError)

    try:
        popt,pcov = scipy.optimize.curve_fit(lambda t,a,b,c: a*np.exp(b*t)+c,  X,  Dat, p0=(-1,-1/100,1),bounds=((-np.inf,-1/20,-np.inf),(0,-1/300,np.inf)),maxfev=800)
        a_er=np.sqrt(np.diag(pcov))[0]
        b_er=np.sqrt(np.diag(pcov))[1]
        c_er=np.sqrt(np.diag(pcov))[2]

    except OptimizeWarning:
        E=1
        print('Maxed out calls.')

    except RuntimeError:
        print("Error - curve_fit failed")
        E=1

    except RuntimeWarning :
        print("Invalid value")
        E=1

    except Warning :
        print('Warning was raised as an exception!')
        E=1
    except FloatingPointError :
        print('Warning was raised as an exception!')
        E=1

plt.plot(X,f(X,popt),label='Exponental fit for probe D',linestyle='--')
plt.plot(Dat,color='orange',label='Potential of Probe D')

plt.xlabel('Point Number')
plt.ylabel('Potential(V)')
plt.grid()
plt.title('Exponential fit for the minimums')
plt.legend()
plt.show()

# MAX

e=476545
data=D

E=0
min_index,l_err,RMSE=opt(e,-1,data)
X=np.arange(0,10+10*min_index,1)
Dat=[data['Potential(V)'][e+i] for i in np.arange(0,10+10*min_index,1)]

with warnings.catch_warnings():

    warnings.simplefilter("error", OptimizeWarning)
    warnings.simplefilter("error", RuntimeError)

    try:
        popt,pcov = scipy.optimize.curve_fit(lambda t,a,b,c: a*np.exp(b*t)+c,  X,  Dat, p0=(1,-2/75,1),bounds=((0,-1/20,-np.inf),(np.inf,-1/300,np.inf)),maxfev=800)
        a_er=np.sqrt(np.diag(pcov))[0]
        b_er=np.sqrt(np.diag(pcov))[1]
        c_er=np.sqrt(np.diag(pcov))[2]

    except OptimizeWarning:
        E=1
        print('Maxed out calls.')

    except RuntimeError:
        print("Error - curve_fit failed")
        E=1

    except RuntimeWarning :
        print("Invalid value")
        E=1

    except Warning :
        print('Warning was raised as an exception!')
        E=1

    except FloatingPointError :
        print('Warning was raised as an exception!')
        E=1

plt.plot(X,f(X,popt),label='Exponental fit for probe B',linestyle='--')
plt.plot(Dat,color='blue',label='Potential of Probe B')

data=D2
min_index,l_err,RMSE=opt(e,1,data)
X=np.arange(0,10+10*min_index,1)
Dat=[data['Potential(V)'][e+i] for i in np.arange(0,10+10*min_index,1)]

e=476545
data=D2

E=0
min_index,l_err,RMSE=opt(e,-1,data)
X=np.arange(0,10+10*min_index,1)
Dat=[data['Potential(V)'][e+i] for i in np.arange(0,10+10*min_index,1)]

with warnings.catch_warnings():

    warnings.simplefilter("error", OptimizeWarning)
    warnings.simplefilter("error", RuntimeError)

    try:
        popt,pcov = scipy.optimize.curve_fit(lambda t,a,b,c: a*np.exp(b*t)+c,  X,  Dat, p0=(1,-2/75,1),bounds=((0,-1/20,-np.inf),(np.inf,-1/300,np.inf)),maxfev=800)
        a_er=np.sqrt(np.diag(pcov))[0]
        b_er=np.sqrt(np.diag(pcov))[1]
        c_er=np.sqrt(np.diag(pcov))[2]

    except OptimizeWarning:
        E=1
        print('Maxed out calls.')

    except RuntimeError:
        print("Error - curve_fit failed")
        E=1

    except RuntimeWarning :
        print("Invalid value")
        E=1

    except Warning :
        print('Warning was raised as an exception!')
        E=1

    except FloatingPointError :
        print('Warning was raised as an exception!')
        E=1

plt.plot(X,f(X,popt),label='Exponental fit for probe D',linestyle='--')
plt.plot(Dat,color='orange',label='Potential of Probe D')

plt.xlabel('Point Number')
plt.ylabel('Potential(V)')
plt.grid()
plt.title('Exponential fit for the maximums')
plt.legend()
plt.show()

###Paragraph 8.2

plt.hist(Distances1)
plt.hist(Distances2)
plt.grid()
plt.hist(Distances1,bins=20,color='blue',ec='blue',alpha=0.4,label='Probe B')
plt.hist(Distances2,bins=20,color='darkorange',ec='darkorange',alpha=0.4,label='Probe D')
plt.xlabel('Distance from the MA (in number of sd)')
plt.ylabel('Distance from the MA (in number of sd)')
plt.legend()
plt.title('Histogram of the distance from the MA (in number of sd)')
plt.show()

###Paragraph 10

###Paragraph 11.1

simple=pd.read_csv("/Users/victorletzelter/Desktop/Projet_python/Data_Sim.txt",delimiter='\t')
exec(open("/Users/victorletzelter/Desktop/Projet_python/sampleDreal.txt").read())
plt.plot(simple['S'],label='Simulated data')
plt.plot(Dreal,color='red',label='Real data',alpha=0.8)
#plt.plot(simple['zf'],label='ZF Zones in simulated data',alpha=0.8)
plt.plot()
plt.xlabel('Point number')
plt.ylabel('Potential (V)')
plt.title('Paces of the simulated and the real data')
plt.grid()
plt.legend()
plt.show()

###11.2

mu=0
sigma0=5.866

dat=np.random.normal(mu, sigma0, 50000)
dat=pd.DataFrame(dat,columns=['S'])
dat=dat.set_index(np.arange(450000,500000,1))
datM=dat.rolling(100).mean()
S1=np.std(dat['S'])
S2=np.std(datM['S'])
datM=datM*(S1/S2)

#S_1=savgol_filter(datM['S'],int(len(datM)/10000), 3)
S_1=savgol_filter(datM['S'],7, 2)

dat_s=pd.DataFrame.copy(datM)
x1=450000
x2=500000

for i in range (x1,x2) :
    dat_s['S'][i]=S_1[i-x1]

S_2=savgol_filter(datM['S'],int(len(dat_s)/10)+1, 2)

dat_s2=pd.DataFrame.copy(dat_s)


for i in range (x1,x2) :
    dat_s2['S'][i]=dat_s['S'][i]-S_2[i-x1]

plt.plot(dat_s2,label='Simulated')
plt.plot(D['Potential(V)'],color='red',label='Real',alpha=0.8)
plt.xlabel('Point number')
plt.ylabel('Potential (V)')
plt.title('Paces of the simulated (V2) and the real data ')
plt.grid()
plt.legend()
plt.show()

###11.3
def plot_gamma(U) :
    alpha=np.mean(U)**2/np.var(U) #4.150404428559912
    beta=np.mean(U)/np.var(U) #0.04294735710638756
    print('alpha={}'.format(alpha))
    print('beta={}'.format(beta))
    #X=np.arange(0,200,1)
    min=np.min(U)
    max=np.max(U)
    X=np.linspace(min,max,len(U))
    if X[0]>0 :
        Y=[scipy.stats.gamma.pdf(e,alpha,0,1/beta) for e in X]
    else :
        Y=[scipy.stats.gamma.pdf(-e,alpha,0,1/beta) for e in X]
    return([-e for e in X],Y)
def histA(DP,Numero=1) :
    DP=F_Int_min
    A=[]
    for e in DP :
        if Numero==1 :
            a=score_min(e[0],D)[2][0]
        elif Numero==2 :
            a=score_min(e[0],D2)[2][0]
        A.append(a)
    #plt.hist(A,density=True)
    return(A)
def histB(DP,Numero=1) :
    B=[]
    for e in DP :
        if Numero==1 :
            b=score_min(e[0],D)[2][1]
        elif Numero==2 :
            b=score_min(e[0],D2)[2][1]
        B.append(b)
    #plt.hist(B,density=True)
    return(B)

exec(open('/Users/victorletzelter/Desktop/Projet_python/sampleF_Int_min.txt').read())
H=histH(F_Int_min,'blue')
A=histA(F_Int_min)
B=histB(F_Int_min)

fig, ax = plt.subplots(3, figsize=(3, 1))
plt.xlabel('Values of the parameters')

X,Y=plot_gamma(H)
ax[0].hist(H,density=True)
ax[0].plot(X1,Y1,label='Gamma distribution')
#ax[0].set_xlabel('H values')
ax[0].set_ylabel('h density')
#ax[0].set_title('Fit with a Gamma distribution')
ax[0].legend()

X2,Y2=plot_gamma([-e for e in A])
ax[1].hist(A,density=True)
ax[1].plot(X2,Y2)
#ax[1].set_xlabel('a values')
ax[1].set_ylabel('a density')
#ax[1].set_title('Fit with a Gamma distribution')

#X3,Y3=plot_gamma([-e for e in B])
#ax[2].hist(B,density=True)
#ax[2].plot(X3,Y3)
#ax[2].set_xlabel('Values of the ')
#ax[2].set_ylabel('b density')
#ax[2].set_title('Fit with a Gamma distribution')

#B=[-e for e in B]
X3,Y3=plot_gamma(B)
X3=[-e for e in X3]
ax[2].hist(B,density=True)
ax[2].plot(X3,Y3)
#ax[2].set_xlabel('Values of the ')
ax[2].set_ylabel('b density')
#ax[2].set_title('Fit with a Gamma distribution')

plt.show()

###Paragraph 12.3
import os
import random

def execNr(Nr) :

    h=100
    start=time.time()
    In,Out=to_supervised_with_recoverage(data,h,Nr)
    In2,Out2=to_supervised_with_recoverage(data_test,h,Nr)
    print(time.time()-start)

    activationl=['selu']
    kernell=['lecun_normal']

    Couche1=128
    Couche2=256
    activation1='selu'
    activation2='selu'
    kernel1='lecun_normal'
    kernel2='lecun_normal'
    Decay=0.999
    initial_learning_rate=10**(random.uniform(-5, -2))
    initial_learning_rate=0.0002
    Dropout=0.004
    DropoutName=int(Dropout*100)
    #initial_learning_rate=round(initial_learning_rate,5)

    model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(units=Couche1, activation=activation1,kernel_initializer=kernel1),
    tf.keras.layers.Dropout(Dropout),
    tf.keras.layers.Dense(units=Couche2, activation=activation2,kernel_initializer=kernel2),
    tf.keras.layers.Dropout(Dropout),
    tf.keras.layers.Dense(h,activation='sigmoid')
    ])

    #lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    #    initial_learning_rate=initial_learning_rate,
    #    decay_steps=33046,
    #    decay_rate=Decay,staircase=True) #initial_learning_rate * decay_rate ^ (step / decay_steps)
    #initial_learning_rate=0.001

    earlystopping = tf.keras.callbacks.EarlyStopping(monitor ="val_loss",mode ="min", patience = 10,restore_best_weights = True)

    model.compile(
            optimizer='Adam',
            loss= tf.keras.losses.MeanSquaredError(),
            metrics=[tf.keras.metrics.MeanSquaredError()],
        )

    hist=model.fit(In, Out,batch_size=256,epochs=300,validation_split=0.8,validation_data=(In2,Out2), callbacks =[earlystopping])

    Res=model(In2).numpy()
    os.chdir("/Users/victorletzelter/Desktop/Projet_python")
    #save2(Res,'Res')

    n,p=np.shape(Res)
    print(list(Res[1]))

    ZF=convertV3(Res,h,Nr)
    name='NRmax2_{}'.format(Nr)
    save(ZF,'ZF',name)

execNr(1) #10.238461017608643
execNr(2) #24.466486930847168
execNr(5) #47.196423053741455
execNr(10) #89.77162599563599 q

#X=[1,2,5,10]
#Y=[10.2,24.5,47.19,89.77]
#plt.plot(X,Y)
#plt.show()

###Python 3.9

import os

os.chdir("/Users/victorletzelter/Desktop/Projet_python")
exec(open('Meeting_3_08_functions.py').read())
exec(open('imports_gestion.py').read())
exec(open("/Users/victorletzelter/Desktop/Projet_python/NrMax2_1").read())
ZF_1=pd.DataFrame(ZF)[0][134326:134570]
exec(open("/Users/victorletzelter/Desktop/Projet_python/NrMax2_2").read())
ZF_2=pd.DataFrame(ZF)[0][134326:134570]
exec(open("/Users/victorletzelter/Desktop/Projet_python/NrMax2_5").read())
ZF_3=pd.DataFrame(ZF)[0][134326:134570]
exec(open("/Users/victorletzelter/Desktop/Projet_python/NrMax2_10").read())
ZF_4=pd.DataFrame(ZF)[0][134326:134570]
data_test=pd.read_csv("/Users/victorletzelter/Desktop/Projet_python/Data_WE_test_big",delimiter='\t')

def TZ(ZF) :
    return((ZF>0.9))

ZF_1p=TZ(ZF_1)
ZF_2p=TZ(ZF_2)
ZF_3p=TZ(ZF_3)
ZF_4p=TZ(ZF_4)

plt.plot(data_test['S'][134326:134570],label='Simulated data')
plt.plot(data_test['zf'][134326:134570],label='Presence of ZF (Truth)')
plt.plot(ZF_1,label='Presence of ZF (Deduced by the NN) with n=1')
plt.plot(ZF_2,label='Presence of ZF (Deduced by the NN) with n=2')
plt.plot(ZF_3,label='Presence of ZF (Deduced by the NN) with n=5')
plt.plot(ZF_4,label='Presence of ZF (Deduced by the NN) with n=10')
plt.legend()
plt.title('ZF detection with artificial data')
plt.xlabel('Point Number')
plt.ylabel('Potential(V)')
plt.show()

###
os.chdir("/Users/victorletzelter/Desktop/Projet_python")
exec(open('imports_gestion.py').read())
data_test=pd.read_csv("/Users/victorletzelter/Desktop/Projet_python/Data_WE_test_big",delimiter='\t')
Nr=5
name='NR{}'.format(Nr)
exec(open("/Users/victorletzelter/Desktop/Projet_python/NRmax{}".format(Nr)).read())
ZF=pd.DataFrame(ZF)[0]

plt.plot(data_test['S'],label='Simulated data')
plt.plot(data_test['zf'],label='Presence of ZF (Truth)')
plt.plot(ZF,label='Presence of ZF (Deduced by the NN)')
plt.grid()
plt.legend()
plt.title('ZF detection with artificial data')
plt.xlabel('Point Number')
plt.ylabel('Potential(V)')
plt.show()

###Loss

exec(open("RapportL2").read())
#exec(open("/Users/victorletzelter/Desktop/Projet_python/samplehist.txt").read())
plt.plot(x['loss'],label='train loss')
plt.plot(x['val_loss'],label='test loss')
plt.axvline(x=22,color='red',alpha=0.2)
plt.legend()
plt.xlabel('Epoch number')
plt.ylabel('Loss value')
plt.title('Loss function in train and test set using Adam')
plt.show()

### 12.3 Results on the real data
def convertV4(data,Ys,h=50,Nr=5) :

  n,p=np.shape(Ys)
  Yt=[0]*len(data)
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

###3.8
Nr=5
h=100

execNr(Nr)

exec(open("/Users/victorletzelter/Desktop/Projet_python/sampleDreal.txt").read())

In3=to_supervised_with_recoverageV2(Dreal,h,Nr)

Res=model(In3).numpy()
ZF=convertV4(Dreal,Res,h,Nr)
save(ZF,'ZF','ZFN')

###3.9

import os
os.chdir("/Users/victorletzelter/Desktop/Projet_python")
exec(open('imports_gestion.py').read())
exec(open('/Users/victorletzelter/Desktop/Projet_python/NEW_METHOD_C_OK2.py').read())
exec(open("/Users/victorletzelter/Desktop/Projet_python/ZFN").read())
exec(open("/Users/victorletzelter/Desktop/Projet_python/sampleDreal.txt").read())

ZF=pd.DataFrame(ZF)[0]
ZF=ZF>0.99

plt.plot(Dreal,label='Real data')
plt.plot(ZF,label='Presence of ZF (Deduced by the NN)')
plt.legend()
plt.grid()
plt.title('ZF detection with real data')
plt.xlabel('Point Number')
plt.ylabel('Potential(V)')
plt.show()

###Performance on the real data
###RMSE
import os
os.chdir("/Users/victorletzelter/Desktop/Projet_python")
exec(open('zi1').read())
exec(open('MinRatekeep701').read())
exec(open("/Users/victorletzelter/Desktop/Projet_python/ZFN").read())
ZF=pd.DataFrame(ZF)[0]
GP2_min=[]
DP2_min=[]
GP_min=[]
DP_min=[]
GP2_max=[]
DP2_max=[]
GP_max=[]
DP_max=[]

traitement(zi)

def indicatrice(Int,x1,x2) :

    ind=[0]*(x2-x1)
    ind=pd.DataFrame(ind)
    ind=ind.set_index(np.arange(x1,x2,1))

    for i in range (len(Int)) :

        for l in range (Int[i][0],Int[i][1]) :

            ind[0][l]=1

    return(ind[0])

L1=indicatrice(DP_min,450000,500000)
F1=indicatrice(F_int_min,450000,500000)
L1=pd.DataFrame(L1)
F1=pd.DataFrame(F1)
L1=L1.set_index(ZF.index)
F1=F1.set_index(ZF.index)
L1=L1[0]

ZFF=ZF>0.999

def indicatriceinverse(Ind) :
    L=[]
    i=0
    while i<len(Ind) :
        if Ind[i]==1 :
            min=i
            while Ind[i]==1 :
                i+=1
            max=i
            if max!=min :
                L.append((min,max))
        i+=1
    return(L)

II=indicatriceinverse(ZFF)

def filtrelength(L,seuil) :
    Lprime=[]
    for (a,b) in L :
        if abs(b-a)>seuil :
            Lprime.append((a,b))
    return(Lprime)

IIfiltre=filtrelength(II,40)
ZFFfiltre=indicatrice(IIfiltre,0,len(ZFF))

plt.plot(Dreal,label='Real data')
plt.plot(F1,label='Presence of ZF (real)')
plt.plot(ZFFfiltre,label='Presence of ZF (Deduced by the NN)')
plt.legend()
plt.grid()
plt.title('ZF detection with real data')
plt.xlabel('Point Number')
plt.ylabel('Potential(V)')
plt.show()

a=22200
b=22300
L=b-a
RMSE=np.sqrt(np.sum((ZFFfiltre[a:b]-F1[0][a:b])**2)/L)
print(RMSE)
RMSE=np.sqrt(np.sum((ZFF[a:b]-F1[0][a:b])**2)/L)
print(RMSE)
RMSE=np.sqrt(np.sum((F1[0][a:b])**2)/L)
print(RMSE)


###F_Score
def performance(F_Int_min,F_Int_min2,F_Int_max,F_Int_max2) :

    def indicatrice(Int) :

        ind=[0]*(x2-x1)
        ind=pd.DataFrame(ind)
        ind=ind.set_index(np.arange(x1,x2,1))

        for i in range (len(Int)) :

            for l in range (Int[i][0],Int[i][1]) :

                ind[0][l]=1

        return(ind[0])

    R1=indicatrice(F_Int_min)
    R2=indicatrice(F_Int_max)
    L1=indicatrice(DP_min)
    L2=indicatrice(DP_max)

    produit_min1=R1*L1
    produit_max1=R2*L2

    def filtre2(Int,MAX) : #MAX=1 pour max et MAX=-1 pour min

        F_Int=[]

        for (a,b) in Int :

            keep=0 #We do not keep

            for e in range (a,b) :

                if MAX==-1 :
                    if produit_min1[e]==1 :
                        keep=1

                elif MAX==1 :
                    if produit_max1[e]==1 :
                        keep=1

            if keep==1 :

                F_Int.append((a,b))

        return(F_Int)

    Confusion=filtre2(F_Int_min,-1)

    Precision=len(Confusion)/len(F_Int_min) #Precision : Env 28% (Idem pour le 2 : Vrais Positifs / (Vrais positifs + Faux positifs) "Proportion de vrais positifs parmis ceux sélectionnés"
    Recall=len(Confusion)/len(DP_min) #Recall : Env 15% (Idem pour le 2 : Vrais Positifs / (Vrais positifs + Faux négatifs) "Proportion de vrais positifs parmis ceux qui sont réellement positifs

    F1_score1=s.harmonic_mean([Precision,Recall])

    R1=indicatrice(F_Int_min2)
    R2=indicatrice(F_Int_max2)
    L1=indicatrice(DP2_min)
    L2=indicatrice(DP2_max)

    produit_min1=R1*L1
    produit_max1=R2*L2

    def filtre2(Int,MAX) : #MAX=1 pour max et MAX=-1 pour min

        F_Int=[]

        for (a,b) in Int :

            keep=0 #We do not keep

            for e in range (a,b) :

                if MAX==-1 :
                    if produit_min1[e]==1 :
                        keep=1

                elif MAX==1 :
                    if produit_max1[e]==1 :
                        keep=1

            if keep==1 :

                F_Int.append((a,b))

        return(F_Int)

    Confusion2=filtre2(F_Int_min2,-1)

    Precision2=len(Confusion2)/len(F_Int_min2)
    Recall2=len(Confusion2)/len(DP2_min)

    F1_score2=s.harmonic_mean([Precision2,Recall2])

    F1_s=s.mean([F1_score1,F1_score2])

    return((F1_s,Precision,Recall,Precision2,Recall2))













