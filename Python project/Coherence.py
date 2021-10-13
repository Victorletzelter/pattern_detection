def Correlation(D,D2,R=50) : #D and D2 and the two data frames
 #R in the value of the rolling average for the time correlation calculus

#This function returns the points numbers which are considered to be highly correlated (in the temporal domain) in 'Potential(V)' component of the two data frames D and D2


    Corr=D['Potential(V)'].rolling(R).corr(D2['Potential(V)'])

    #First filtration
    t=(Corr>0.90).astype(int)  #Idem

    return(t)

def iscorr(e,t,M=100) : #This function returns, for a point e whether it is correlated given the result of the previous function, and a tolerance gap value M
    for i in range (e-M,e+M) :
        if i>=x1 and i<x2 :
            if t[i] :
                return(1) #1 TRUE
    return(0)

def Correlation_s(D,D2,seuil_opt,x1,x2,N=4,F=100) : #seuil_opt stands for the optimal value of the cross spectral product, for a point to be considered as 'high spectral correlation point' : this value was determined by hand labelling of the data

    Fs=2e6
    f, t, Sxx = signal.spectrogram(D['Potential(V)'], Fs, window=('gaussian',F))
    f=f[0:N]
    Sxx=Sxx[0:N,:]

    f2, t2, Sxx2 = signal.spectrogram(D2['Potential(V)'], Fs,window=('gaussian',F))
    f2=f2[0:N]
    Sxx2=Sxx2[0:N,:]

    T=len(t)

    SXX=[Sxx[:,t] for t in range (0,T)]
    SYY=[Sxx2[:,t] for t in range (0,T)]

    Corr=[0]*T

    for j in range (T) :
        Corr[j]=np.dot(SXX[j],SYY[j])

    seuil=np.mean(Corr)

    pas=int((x2-x1)/(T))
    indices=np.arange(x1,x2,pas)
    Corr2=pd.DataFrame([0]*len(D['Potential(V)']),index=D.index)

    for j in range (len(indices)-2) :
        Corr2[0][indices[j]]=Corr[j]*10**(9)

    for k in range (x1,x2) :
        if Corr2[0][k]==0 :
            Corr2[0][k]=np.nan

    Corr2=Corr2.interpolate(method='linear') #The obtain the right number of points, a linear interpolation has been performed

    #First filtration
    #t=(Corr2>seuil*10**(9)).astype(int)  #Idem
    t=(Corr2>seuil_opt).astype(int)  #Idem
    t=t[0]

    return(t)
