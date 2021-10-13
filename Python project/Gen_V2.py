pd.options.mode.chained_assignment = None

dataG=pd.read_csv("/Users/victorletzelter/Desktop/PartieR/Gen2.txt",delimiter=' ')
dataG=dataG.set_index(np.arange(450000,500000,1))
dataG2=dataG.rolling(100).mean()
S1=np.std(dataG['S'])
S2=np.std(dataG2['S'])
I=np.where(dataG['zf']==0)
J=np.where(dataG2['zf']>0)

x1=450000

S_1=savgol_filter(dataG['S'],int(len(dataG)/10)+1, 2)
dataG_s=pd.DataFrame.copy(dataG)

for i in range (x1,x2) :
    dataG_s['S'][i]=S_1[i-x1]

for m in range (len(I[0])) :
    I[0][m]=I[0][m]

for e in I[0] :
    if e+450000<499999 :
        if (dataG2['zf'][450000+e+1]-dataG2['zf'][450000+e])>=0 :
            dataG2['S'][450000+e]=dataG2['S'][450000+e]*(S1/S2)


#for e in J[0] :
#    dataG2['S'][450000+e]=dataG['S'][450000+e]

#dataG[0]=dataG2[0]*(S1/S2)

fig, ax = plt.subplots(2, figsize=(10, 7))

### Version 1

ax[0].plot(dat_s['S'])
#ax[0].plot(dataG_s['zf'])
ax[0].set_xlabel('Point number')
ax[0].set_ylabel('Potential value(V)')
ax[0].set_title('Simulated data')
ax[1].plot(D['Potential(V)'])
ax[1].set_xlabel('Point number')
ax[1].set_ylabel('Potential value(V)')
ax[1].set_title('Real data')
ax[0].get_shared_x_axes().join(ax[0], ax[1])
ax[1].get_shared_y_axes().join(ax[0], ax[1])
plt.show()

plt.close()

### Version 2
#Without the ZF

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
#plt.plot(D['Potential(V)'],label='Real')
plt.grid()
plt.legend()
plt.show()

#After adding the ZF

p=0.0009
h=50
a=-14
b=0.04

np.random.seed(1234)
Rep1={'S':[0]*50000,'zf':[0]*50000}
E=pd.DataFrame(Rep1)
E=E.set_index(np.arange(450000,500000,1))

j=450000 #indice courant

while j<500000 :

    T = min(np.random.geometric(p)+j,500000-h)

    for elt in np.arange(T,T+h,1) :
        E['S'][elt]=a*np.exp(-b*(elt-T))
        E['zf'][elt]=1

    j=T+h

#plt.plot(E['S'])

dat_WE=dat_s2+E

plt.plot(dat_WE['S'])
#plt.plot(D['Potential(V)'])
plt.plot(E['zf'])
plt.show()

os.chdir("/Users/victorletzelter/Desktop/Projet_python")
file = open("Data_Sim.txt", "w")
file.write('{}\t{}\n'.format('S','zf'))

for i in range (452600,500000,1) :
    file.write('{}\t{}\n'.format(round(dat_WE['S'][i],3),E['zf'][i]))

file.close()

exec(open("/Users/victorletzelter/Desktop/Projet_python/sampleRes-2.txt").read())

### Version 3
#Without the ZF
import os
os.chdir("/Users/victorletzelter/Desktop/Projet_python")
exec(open('imports_gestion.py').read())

def gen2(N,rd_seed) :
    mu=0
    sigma0=5.866
    p=0.0009
    h=50
    a=-14
    b=0.04
    x1=0
    x2=N

    dat=np.random.normal(mu, sigma0, N)
    dat=pd.DataFrame(dat,columns=['S'])
    datM=dat.rolling(100).mean()
    S1=np.std(dat['S'])
    S2=np.std(datM['S'])
    datM=datM*(S1/S2)

    #S_1=savgol_filter(datM['S'],int(len(datM)/10000), 3)
    S_1=savgol_filter(datM['S'],7, 2)
    dat_s=pd.DataFrame.copy(datM)

    for i in range (x1,x2) :
        dat_s['S'][i]=S_1[i-x1]

    S_2=savgol_filter(datM['S'],int(len(dat_s)/10)+1, 2)

    dat_s2=pd.DataFrame.copy(dat_s)

    for i in range (x1,x2) :
        dat_s2['S'][i]=dat_s['S'][i]-S_2[i-x1]

    #plt.plot(dat_s2,label='Simulated')
    #plt.plot(D['Potential(V)'],label='Real')

#After adding the ZF

    np.random.seed(rd_seed)
    Rep1={'S':[0]*N,'zf':[0]*N}
    E=pd.DataFrame(Rep1)

    j=0 #indice courant

    while j<N :

        T = min(np.random.geometric(p)+j,N-h)

        for elt in np.arange(T,T+h,1) :
            E['S'][elt]=a*np.exp(-b*(elt-T))
            E['zf'][elt]=1

        j=T+h

    dat_WE=dat_s2+E

    return((dat_WE['S'],E))

(dat_WE,E)=gen2(500000,1234) #800000
(dat_WE2,E2)=gen2(4500000,1222) #200000

#i=0
#while isnan(dat_WE2[i]) :
#    i+=1
#print(i)

def write(dat_WE,E,N,Name) :
    os.chdir("/Users/victorletzelter/Desktop/Projet_python")
    file = open(Name, "w")
    file.write('{}\t{}\n'.format('S','zf'))
    for i in range (int(0.06*N),N,1) :
        file.write('{}\t{}\n'.format(round(dat_WE[i],3),E['zf'][i]))
    file.close()

write(dat_WE,E,500000,'Data_WE_test')
write(dat_WE2,E2,4500000,'Data_WE_train')

### Version 4 same but simpler
#Without the ZF

def gen(N,rd_seed) :
    mu=0
    sigma0=5.866
    p=0.0009
    h=50
    a=-14
    b=0.04

    dat=np.random.normal(mu, sigma0, N)
    dat=pd.DataFrame(dat,columns=['S'])
    #datM=dat.rolling(100).mean()
    #S1=np.std(dat['S'])
    #S2=np.std(datM['S'])

    #After adding the ZF

    np.random.seed(rd_seed)
    Rep1={'S':[0.0]*N,'zf':[0.0]*N}
    E=pd.DataFrame(Rep1)

    j=0 #indice courant

    while j<N :

        T = min(np.random.geometric(p)+j,N-h)

        for elt in np.arange(T,T+h,1) :
            E['S'][elt]=a*np.exp(-b*(elt-T))
            E['zf'][elt]=1

        j=T+h

    #plt.plot(E['S'])

    dat_WE=dat['S']+E['S']
    return(dat_WE,E)

dat_WE,E=gen(80000,1234)
dat_WE2,E2=gen(20000,1222)

#plt.plot(dat_WE)
#plt.plot(D['Potential(V)'])
#plt.plot(E['zf'])
#plt.show()

def write(data_WE,E,N,Name) :
    os.chdir("/Users/victorletzelter/Desktop/Projet_python")
    file = open(Name, "w")
    file.write('{}\t{}\n'.format('S','zf'))
    for i in range (5000,N,1) :
        file.write('{}\t{}\n'.format(round(dat_WE[i],3),E['zf'][i]))
    file.close()

write(dat_WE,E,80000,'Data_WE_train')
write(dat_WE2,E2,20000,'Data_WE_test')

### Version 5 : After adding a distribution for a,b,c and for H
import os
os.chdir("/Users/victorletzelter/Desktop/Projet_python")
exec(open('imports_gestion.py').read())

def gen3(N,rd_seed) :
    mu=0
    sigma0=5.866
    p=0.0009
    h=50
    a=-14
    b=0.04
    x1=0
    x2=N
    alphaH=5.666006367173681
    betaH=0.05205518217191368
    alphaA=6.077479483801698 #Warning : inverse the sign
    betaA=0.4723282091501369 #Warning : inverse the sign
    alphaB=8.340017819817685 #Warning : inverse the sign
    betaB=213.4374537734773 #Warning : inverse the sign
    alphaC=1.337078744676507 #Warning : inverse the sign
    betaC=0.2583374503229954 #Warning : inverse the sign

    dat=np.random.normal(mu, sigma0, N)
    dat=pd.DataFrame(dat,columns=['S'])
    datM=dat.rolling(100).mean()
    S1=np.std(dat['S'])
    S2=np.std(datM['S'])
    datM=datM*(S1/S2)

    #S_1=savgol_filter(datM['S'],int(len(datM)/10000), 3)
    S_1=savgol_filter(datM['S'],7, 2)
    dat_s=pd.DataFrame.copy(datM)

    for i in range (x1,x2) :
        dat_s['S'][i]=S_1[i-x1]

    S_2=savgol_filter(datM['S'],int(len(dat_s)/10)+1, 2)

    dat_s2=pd.DataFrame.copy(dat_s)

    for i in range (x1,x2) :
        dat_s2['S'][i]=dat_s['S'][i]-S_2[i-x1]

    #plt.plot(dat_s2,label='Simulated')
    #plt.plot(D['Potential(V)'],label='Real')

#After adding the ZF

    np.random.seed(rd_seed)
    Rep1={'S':[0]*N,'zf':[0]*N}
    E=pd.DataFrame(Rep1)

    j=0 #indice courant

    while j<N :

        h=int(np.random.gamma(alphaH,1/betaH))
        T = min(np.random.geometric(p)+j,N-h)
        a=-np.random.gamma(alphaA,1/betaA)
        b=np.random.gamma(alphaB,1/betaB)

        for elt in np.arange(T,T+h,1) :
            E['S'][elt]=a*np.exp(-b*(elt-T))
            E['zf'][elt]=1

        j=T+h

    dat_WE=dat_s2['S']+E['S']

    return((dat_WE,E['zf']))

(datgen,E)=gen3(50000,998) #800000
#(datgen2,E2)=gen3(,1222) #200000

exec(open('/Users/victorletzelter/Desktop/Projet_python/sampleDreal.txt').read())

plt.plot(datgen)
plt.plot(Dreal)
plt.plot(E)
plt.grid()
plt.show()

#i=0
#while isnan(dat_WE2[i]) :
#    i+=1
#print(i)

def write(dat_WE,E,N,Name) :
    os.chdir("/Users/victorletzelter/Desktop/Projet_python")
    file = open(Name, "w")
    file.write('{}\t{}\n'.format('S','zf'))
    for i in range (int(0.06*N),N,1) :
        file.write('{}\t{}\n'.format(round(dat_WE[i],3),E['zf'][i]))
    file.close()

write(datgen,E,500000,'Data_WE_test')
write(datgen2,E2,4500000,'Data_WE_train')

### Version 6 : Oscillations
#Without the ZF
import os
os.chdir("/Users/victorletzelter/Desktop/Projet_python")
exec(open('imports_gestion.py').read())

def gen3(N,rd_seed) :
    mu=0
    sigma0=5.866
    p=0.0009
    h=50
    a=-14
    b=0.04
    x1=0
    x2=N
    alphaH=5.666006367173681
    betaH=0.05205518217191368
    alphaA=6.077479483801698 #Warning : inverse the sign
    betaA=0.4723282091501369 #Warning : inverse the sign
    alphaB=8.340017819817685 #Warning : inverse the sign
    betaB=213.4374537734773 #Warning : inverse the sign
    alphaC=1.337078744676507 #Warning : inverse the sign
    betaC=0.2583374503229954 #Warning : inverse the sign

    dat=np.random.normal(mu, sigma0, N)
    dat=pd.DataFrame(dat,columns=['S'])
    datM=dat.rolling(100).mean()
    S1=np.std(dat['S'])
    S2=np.std(datM['S'])
    datM=datM*(S1/S2)

    #S_1=savgol_filter(datM['S'],int(len(datM)/10000), 3)
    S_1=savgol_filter(datM['S'],9, 2)
    dat_s=pd.DataFrame.copy(datM)

    for i in range (x1,x2) :
        dat_s['S'][i]=S_1[i-x1]

    S_2=savgol_filter(datM['S'],int(len(dat_s)/10)+1, 2)

    dat_s2=pd.DataFrame.copy(dat_s)

    for i in range (x1,x2) :
        dat_s2['S'][i]=dat_s['S'][i]-S_2[i-x1]

    #plt.plot(dat_s2,label='Simulated')
    #plt.plot(D['Potential(V)'],label='Real')

#After adding the ZF

    np.random.seed(rd_seed)
    Rep1={'S':[0]*N,'zf':[0]*N}
    E=pd.DataFrame(Rep1)

    j=0 #indice courant

    while j<N :

        h=int(np.random.gamma(alphaH,1/betaH))
        T = min(np.random.geometric(p)+j,N-h)
        a=-np.random.gamma(alphaA,1/betaA)
        b=np.random.gamma(alphaB,1/betaB)
        print('h:{}'.format(h))
        print('a:{}'.format(a))
        print('b:{}'.format(b))
        print('T:{}'.format(T))

        for elt in np.arange(T,T+h,1) :
            E['S'][elt]=a*np.exp(-b*(elt-T))
            E['zf'][elt]=1

        j=T+h

    dat_WE=dat_s2['S']+E['S']

    return((dat_WE,E['zf']))

(datgen,E)=gen3(4500000,998) #800000
(datgen2,E2)=gen3(500000,1222) #200000

write(datgen,E,4500000,'Data_WE_train')
write(datgen2,E2,500000,'Data_WE_test')

exec(open('/Users/victorletzelter/Desktop/Projet_python/sampleDreal.txt').read())

plt.plot(datgen)
plt.plot(Dreal)
plt.plot(E)
plt.grid()
plt.show()

#i=0
#while isnan(dat_WE2[i]) :
#    i+=1
#print(i)

def write(dat_WE,E,N,Name) :
    os.chdir("/Users/victorletzelter/Desktop/Projet_python")
    file = open(Name, "w")
    file.write('{}\t{}\n'.format('S','zf'))
    for i in range (int(0.06*N),N,1) :
        file.write('{}\t{}\n'.format(round(dat_WE[i],3),E[i]))
    file.close()

write(datgen,E,500000,'Data_WE_test')
write(datgen2,E2,4500000,'Data_WE_train')

def test_spectro() :
    #V=int(0.94*50000)
    #datgen=datgen.tail(V)

    Fs=2e6
    F=100
    n=4
    fig, ax = plt.subplots(2, figsize=(8, 7))
    f, t, Sxx = signal.spectrogram(datgen, Fs, window=('gaussian',F))
    f=f[0:n]
    Sxx=Sxx[0:n,:]
    im=ax[0].pcolormesh(t, f, Sxx,shading='auto')
    ax[0].set_ylabel('Frequency [Hz]')
    ax[0].set_xlabel('Time [sec]')
    #fig.colorbar(im, ax=ax[0])

    X=np.arange(0,50000,1)
    X=(X/50000)*t[len(t)-1]

    ax[1].plot(X,datgen)
    ax[1].plot(X,E)
    ax[1].set_xlabel('Time [sec]')
    ax[1].set_ylabel('Potential (V)')
    #fig.colorbar(im2, ax=ax[1])

    ax[0].get_shared_x_axes().join(ax[0], ax[1])














## trash
ax[0].plot(dataG2['S'])
ax[0].plot(dataG2['zf'])
ax[0].set_xlabel('Point number')
ax[0].set_ylabel('Potential value(V)')
ax[0].set_title('Simulated data')
ax[1].plot(D['Potential(V)'])
ax[1].set_xlabel('Point number')
ax[1].set_ylabel('Potential value(V)')
ax[1].set_title('Real data')
ax[0].get_shared_x_axes().join(ax[0], ax[1])
ax[1].get_shared_y_axes().join(ax[0], ax[1])
plt.show()
plt.close()