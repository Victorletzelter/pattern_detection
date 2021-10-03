### Version 5 : After adding a distribution for a,b and for h
import os
os.chdir("/Users/victorletzelter/Desktop/Projet_python")
exec(open('imports_gestion.py').read())

def gen3(N,rd_seed) : #N is the number of points number to be generated
#rd_seed corresponds to the random seed

#This function returns labelled artificial data
    #Generation of data without ZF at first
    mu=0
    sigma0=5.866
    p=0.0009
    a=-14
    b=0.04
    x1=0
    x2=N
    alphaH=5.399
    betaH=0.052
    alphaA=5.111 #Warning : inverse the sign
    betaA=0.361 #Warning : inverse the sign
    alphaB=8.296 #Warning : inverse the sign
    betaB=222.735 #Warning : inverse the sign

    dat=np.random.normal(mu, sigma0, N) #Generation of the normal law
    dat=pd.DataFrame(dat,columns=['S'])
    datM=dat.rolling(100).mean() #Applying moving average
    S1=np.std(dat['S'])
    S2=np.std(datM['S'])
    datM=datM*(S1/S2)

    #Then, the savgol_filter is applied two times

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

    #The ZF are added

    np.random.seed(rd_seed)
    Rep1={'S':[0]*N,'zf':[0]*N}
    E=pd.DataFrame(Rep1)

    j=0

    while j<N :

        T = min(np.random.geometric(p)+j,N-h) #Beginning of the ZF
        h=int(np.random.gamma(alphaH,1/betaH)) #The length of the ZF
        a=-np.random.gamma(alphaA,1/betaA)
        b=np.random.gamma(alphaB,1/betaB)

        for elt in np.arange(T,T+h,1) :
            E['S'][elt]=a*np.exp(-b*(elt-T))
            E['zf'][elt]=1

        j=T+h

    dat_WZ=dat_s2['S']+E['S']

    return((dat_WZ,E['zf']))

def write(dat_WZ,E,N,Name) : #dat_WZ simulated data containing the ZF intervals
    os.chdir("/Users/victorletzelter/Desktop/Projet_python")
    file = open(Name, "w")
    file.write('{}\t{}\n'.format('S','zf'))
    for i in range (int(0.06*N),N,1) : #This lag allows to remove the NaN values
        file.write('{}\t{}\n'.format(round(dat_WZ[i],3),E['zf'][i]))
    file.close()

(datgen,E)=gen3(9500000,998)
(datgen2,E2)=gen3(500000,1222) #200000

write(datgen2,E2,9500000,'Data_WE_train_big')
write(datgen,E,500000,'Data_WE_test_big')