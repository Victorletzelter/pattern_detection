#This file allows to perform hand-labeling of the ZF intervals, in order to adjust hyperparameters of the Event_labeling.py file.
#Provided that the variable zi was created or initialized, the user can execute the function learning(), to label by hand the ZF intervals.
#If not, the file exec(open("/Users/victorletzelter/Desktop/Projet_python/samplezi.txt").read()) can be executed first. 
#The process to adopt for hand-labeling is precised in the learning() function. 

#Loading of the data 
data1=pd.read_csv("/Users/victorletzelter/Documents/GitHub/pattern_detection/Python project/Data_files/converted_data1.txt",delimiter=' ')
data2=pd.read_csv("/Users/victorletzelter/Documents/GitHub/pattern_detection/Python project/Data_files/converted_data2.txt",delimiter=' ')

def split(dat,x1,x2) : #This function splits the data frame
    l=len(dat)
    dat1=dat.tail(l-int(x1))
    dat1=dat1.head(int(x2-x1))
    return(dat1)

def onclick(event): #For a point to be considered, the user will have to double-click on it
    if event.dblclick :
            zi.append((event.xdata,event.ydata))

def learning(data1,data2) : #This function allows to label by hand data, by clicking on the temporal time series. The relevant values are then stored in a variable called 'zi'
#When labelling the data, the user has to proceed, if possible, in the order : for the Data1 : select a point just before the peak, at the 
#extremum of the peak, and just after the peak.
#For the Data2 : select proceed in the same way. 
#If a point is selected by mistake, the traitement function will filter it automatically. The corresponding 6-uplet just has to be selected again. 

    Dt=split(data1,X1,X2)
    D2t=split(data2,X1,X2)

    def smooth(D,D2,x1,x2) : #This function allows to smooth the signal using savgol_filter

        S_1=savgol_filter(D['Potential(V)'],int(len(D)/10)+1, 2)
        S_2=savgol_filter(D2['Potential(V)'],int(len(D2)/10)+1, 2)

        D_s=pd.DataFrame.copy(D)
        D2_s=pd.DataFrame.copy(D2)

        for i in range (x1,x2) :
            D_s['Potential(V)'][i]=S_1[i-x1]
            D2_s['Potential(V)'][i]=S_2[i-x1]

        for i in range (x1,x2) :
            D['Potential(V)'][i]=D['Potential(V)'][i]-D_s['Potential(V)'][i]
            D2['Potential(V)'][i]=D2['Potential(V)'][i]-D2_s['Potential(V)'][i]

        return((D,D2))

    Da=smooth(Dt,D2t,X1,X2)
    D=Da[0]
    D2=Da[1]

    t=Correlation_s(D,D2,seuil_opt,X1,X2) #Spectral correlation btw the 2 signals

    def fill2(t,c1='blue') : #This function fills the zones with high correlation, it facilitates the labelling in the right intervals
        i=0
        while i<len(t) :
            if t[x1+i]==1 :
                X=[]
                while t[x1+i]==1 :
                    X.append(x1+i)
                    i+=1
                plt.fill_between(X,-30 ,30, color=c1, alpha=0.1)
            i+=1

    fig, ax = plt.subplots(1,1)

    fill2(t)

    ax.plot(Dt['Potential(V)'])
    ax.plot(D2t['Potential(V)'])

    ax.grid()

    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()

def traitement(zi) : #This function supposes that empty lists called  GP2_min, DP2_min, GP_min, DP_min, GP2_max, DP2_max, GP_max and DP_max were created. 
#It allows to fulfill these lists using the zi variable which was created by hand labelling in the learning() function.
#Otherwise, this functions filters points selected by mistake.

    Dt=split(data1,X1,X2)
    D2t=split(data2,X1,X2)

    i=0

    while i<len(zi)-5 :

        if zi[i+2][0]>zi[i+1][0] and zi[i+1][0]>zi[i][0] and zi[i+4][0]>zi[i+3][0] and zi[i+5][0]>zi[i+4][0] : #We check if the pattern is good

            if abs(zi[i][0]-zi[i+3][0])<100 : #Idem

                a0=[abs(D2t['Potential(V)'][int(zi[i][0])]-zi[i][0]),abs(D2t['Potential(V)'][int(zi[i][0])+1]-zi[i][0])] #Check if permutation
                b0=[abs(Dt['Potential(V)'][int(zi[i][0])]-zi[i][0]),abs(Dt['Potential(V)'][int(zi[i][0])+1]-zi[i][0])]

                if min(b0)<min(a0) :
                    zi[i],zi[i+1],zi[i+2],zi[i+3],zi[i+4],zi[i+5]=zi[i+3],zi[i+4],zi[i+5],zi[i],zi[i+1],zi[i+2]

                else :
                    None

                a0=[abs(D2t['Potential(V)'][int(zi[i][0])]-zi[i][0]),abs(D2t['Potential(V)'][int(zi[i][0])+1]-zi[i][0])]
                b0=[abs(Dt['Potential(V)'][int(zi[i][0])]-zi[i][0]),abs(Dt['Potential(V)'][int(zi[i][0])+1]-zi[i][0])]

                a1=[abs(D2t['Potential(V)'][int(zi[i+1][0])]-zi[i+1][0]),abs(D2t['Potential(V)'][int(zi[i+1][0])+1]-zi[i+1][0])]
                b1=[abs(Dt['Potential(V)'][int(zi[i+1][0])]-zi[i+1][0]),abs(Dt['Potential(V)'][int(zi[i+1][0])+1]-zi[i+1][0])]

                a2=[abs(D2t['Potential(V)'][int(zi[i+2][0])]-zi[i+2][0]),abs(D2t['Potential(V)'][int(zi[i+2][0])+1]-zi[i+2][0])]
                b2=[abs(Dt['Potential(V)'][int(zi[i+2][0])]-zi[i+2][0]),abs(Dt['Potential(V)'][int(zi[i+2][0])+1]-zi[i+2][0])]

                a3=[abs(D2t['Potential(V)'][int(zi[i+3][0])]-zi[i+3][0]),abs(D2t['Potential(V)'][int(zi[i+3][0])+1]-zi[i+3][0])]
                b3=[abs(Dt['Potential(V)'][int(zi[i+3][0])]-zi[i+3][0]),abs(Dt['Potential(V)'][int(zi[i+3][0])+1]-zi[i+3][0])]

                a4=[abs(D2t['Potential(V)'][int(zi[i+4][0])]-zi[i+4][0]),abs(D2t['Potential(V)'][int(zi[i+4][0])+1]-zi[i+4][0])]
                b4=[abs(Dt['Potential(V)'][int(zi[i+4][0])]-zi[i+4][0]),abs(Dt['Potential(V)'][int(zi[i+4][0])+1]-zi[i+4][0])]

                a5=[abs(D2t['Potential(V)'][int(zi[i+5][0])]-zi[i+5][0]),abs(D2t['Potential(V)'][int(zi[i+5][0])+1]-zi[i+5][0])]
                b5=[abs(Dt['Potential(V)'][int(zi[i+5][0])]-zi[i+5][0]),abs(Dt['Potential(V)'][int(zi[i+5][0])+1]-zi[i+5][0])]

                n0=int(zi[i][0])+a0.index(min(a0))
                n1=int(zi[i+1][0])+a1.index(min(a1))
                n2=int(zi[i+2][0])+a2.index(min(a2))

                n3=int(zi[i+3][0])+a3.index(min(a3))
                n4=int(zi[i+4][0])+a4.index(min(a4))
                n5=int(zi[i+5][0])+a5.index(min(a5))

                if zi[i+1][1]<zi[i][1] :

                    GP2_min.append((n0,n1))
                    DP2_min.append((n1,n2))
                    GP_min.append((n3,n4))
                    DP_min.append((n4,n5))

                elif zi[i+1][1]>zi[i][1] :

                    GP2_max.append((n0,n1))
                    DP2_max.append((n1,n2))
                    GP_max.append((n3,n4))
                    DP_max.append((n4,n5))

                print(i)
        i+=1

def fill(Int_min,Int_max,c1='blue',c2='red') : #This function enables to fulfill the alleged ZF intervals with colors

    for i in range (0,len(Int_max)) :
        (a,b)=Int_max[i][0],Int_max[i][1]
        X=np.arange(a,b,1)
        ax.fill_between(X,-30 ,30, color=c1, alpha=0.1)

    for i in range (0,len(Int_min)) :
        (a,b)=Int_min[i][0],Int_min[i][1]
        X=np.arange(a,b,1)
        ax.fill_between(X,-30 ,30, color=c2, alpha=0.1)

def plot_mouse() : #This function allows to add new points on zi that are not yet chosen

    fig, ax = plt.subplots()
    fill(DP_min,DP_max)
    fill(DP2_min,DP2_max,'green','orange')
    fill(GP_min,DP_max)
    fill(GP2_min,GP2_max,'green','orange')
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    ax.grid()

    ax.plot(D['Potential(V)'])
    ax.plot(D2['Potential(V)'])
    plt.title('Labelled ZF')
    plt.show()

def Coh_opt(DP,DP2,Rate_keep=95,N=4,F=100,X1=450000,X2=500000) : #DP and DP2 are lists of couples, that correspond to ZF intervals for both probes.
#Rate_keep is a parameter that adjusts the iota coefficient in accordance with the quality of the ZF chosen for the detection
#N adjust the maximum frequency to be considered
#F is the size of the gaussian window to be considered
#This function returns the optimal value of the coherence : the iota coefficient.

    Dt=split(data1,X1,X2)
    D2t=split(data2,X1,X2)

    Fs=2e6
    f, t, Sxx = signal.spectrogram(Dt['Potential(V)'], Fs, window=('gaussian',F))
    f=f[0:N]
    Sxx=Sxx[0:N,:]

    f2, t2, Sxx2 = signal.spectrogram(D2t['Potential(V)'], Fs,window=('gaussian',F))
    f2=f2[0:3]
    Sxx2=Sxx2[0:N,:]

    T=len(t)

    SXX=[Sxx[:,t] for t in range (0,T)]
    SYY=[Sxx2[:,t] for t in range (0,T)]

    Corr=[0]*T

    for j in range (T) :

        Corr[j]=np.dot(SXX[j],SYY[j])

    pas=int((X2-X1)/(T))
    indices=np.arange(X1,X2,pas)
    Corr2=pd.DataFrame([0]*len(Dt['Potential(V)']),index=Dt.index)

    for j in range (len(indices)-1) :
        Corr2[0][indices[j]]=Corr[j]*10**(9)

    for l in range (X1,X2) :
        if Corr2[0][l]==0 :
            Corr2[0][l]=np.nan

    Corr2=Corr2.interpolate(method='linear')
    Corr2=Corr2[0]

    Liste_coh=[]
    Liste_coh2=[]

    for e in DP :
        e=e[0]
        Liste_coh.append(Corr2[e])

    for e in DP2 :
        e=e[0]
        Liste_coh2.append(Corr2[e])

    Seuil1=np.percentile(Liste_coh,100-Rate_keep)
    Seuil2=np.percentile(Liste_coh2,100-Rate_keep)

    Seuil_opt=min(Seuil1,Seuil2)

    return(Seuil_opt)

### a,b and c and T values

def valeur_opt(DP,DP2,MAX,Rate_keep=80) : #MAX=1 if DP and DP2 contains damping intervals that begin with a maximum value, and -1 else
#Rate_keep : adjusting parameter that refers to the rate of 'real ZF' to be kept

#This function provides the optimal values for the a, b and c coefficient for the ZF detection algorithm

    Dt=split(data1,X1,X2)
    D2t=split(data2,X1,X2)

    C=[]
    A_e=[]
    B_e=[]
    C_e=[]
    T=[]

    for e in DP :

        if MAX==1 :
            res=score_max(e[0],Dt)
        else :
            res=score_min(e[0],Dt)

        score=res[0]
        s=res[1]
        T.append(10+10*s)
        C.append(score)

    for e in DP2 :

        if MAX==1 :
            res=score_max(e[0],D2t)
        else :
            res=score_min(e[0],D2t)

        score=res[0]

        T.append(10+10*s)
        C.append(score)

    for elt in C :
        A_e.append(elt[0])
        B_e.append(elt[1])
        C_e.append(elt[2])

    am=np.median(A_e) #1.59 #1.51
    bm=np.median(B_e) #0.0100
    cm=np.median(C_e) #1.51 #1.45

    a_eropt=np.percentile(A_e, Rate_keep)
    b_eropt=np.percentile(B_e, Rate_keep)
    c_eropt=np.percentile(C_e, Rate_keep)

    plt.hist(T)
    tm=np.median(T) #100

    return((a_eropt,b_eropt,c_eropt))

#a_eropt_min,b_eropt_min,c_eropt_min=valeur_opt(DP_min,DP2_min,-1)
#a_eropt_max,b_eropt_max,c_eropt_max=valeur_opt(DP_max,DP2_max,1)

### Distance with the moving average ?

###For the mins
def Distances_opt(DP,DP2,Rate_keep=90,MAX=-1,S=100) : #This function provides the optimal distance with the average when a ZF appears

    S=100
    MAX=-1
    Rate_keep=90
    DP=DP_min
    DP2=DP2_min

    Dt=split(data1,X1,X2)
    D2t=split(data2,X1,X2)

    Distances1=[]
    Distances2=[]
    D_m=Dt.rolling(S).mean()['Potential(V)']
    D2_m=D2t.rolling(S).mean()['Potential(V)']
    D_sd=Dt.rolling(S).std()['Potential(V)']
    D2_sd=D2t.rolling(S).std()['Potential(V)']
    D_rms=np.sqrt(Dt.pow(2).rolling(S).apply(lambda x: np.sqrt(x.mean())))['Potential(V)']
    D2_rms=np.sqrt(D2t.pow(2).rolling(S).apply(lambda x: np.sqrt(x.mean())))['Potential(V)']

    for elt in DP :
        Distances1.append((Dt['Potential(V)'][elt[0]]-D_m[elt[0]])/D_sd[elt[0]])

    for elt in DP2 :
        Distances2.append((D2t['Potential(V)'][elt[0]]-D2_m[elt[0]])/D2_sd[elt[0]])

    if MAX==-1 :

        Distance_opt=max(abs(np.percentile(Distances1, Rate_keep)),abs(np.percentile(Distances2, Rate_keep)))

    elif MAX==1 :

        Distance_opt=max(abs(np.percentile(Distances1, 100-Rate_keep)),abs(np.percentile(Distances2, 100-Rate_keep)))

    return(Distance_opt)

def plotidist() : #This function plots the distogram of distances with the average (in number of sd)

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

###For the max
def plot_dist() :
    Distances1=[]
    Distances2=[]
    S=100

    for elt in DP_max :
        Distances1.append((D['Potential(V)'][elt[0]]-D_m[elt[0]])/D_sd[elt[0]])

    for elt in DP2_max :
        Distances2.append((D2['Potential(V)'][elt[0]]-D2_m[elt[0]])/D2_sd[elt[0]])

    plt.grid()
    plt.hist(Distances1,20,color='blue',alpha=0.4)
    plt.hist(Distances2,20,color='orange',alpha=0.4)
    plt.title('Distance from the MA (in number of sd)')
    plt.show()

#np.mean(Distances1)
#-2.1596255864703964

#np.mean(Distances2)
#-2.1167086057968336

###Filter on the length

def indicatrice(Int) :

    ind=[0]*(X2-X1)
    ind=pd.DataFrame(ind)
    ind=ind.set_index(np.arange(X1,X2,1))

    for i in range (len(Int)) :

        for l in range (Int[i][0],Int[i][1]) :

            ind[0][l]=1

    return(ind[0])

def liste_quotient(Int,Int2) :

    Int=DP_min
    Int2=DP2_min

    ind_=indicatrice(Int)
    ind2_=indicatrice(Int2)

    produit=ind_*ind2_

    Lp=[]

    for (a,b) in Int :
        for (c,d) in Int2 :
            if (a<c and b>c) or (a<c and d<b) or (c<a and d>a) or (c<a and b<d) : #The two intervals recover

                liste=[produit[e] for e in range (min(a,c),max(b,d))]

                n=len(liste)
                liste=np.asarray(liste)
                p=len(np.where(liste==1)[0])/n

                Lp.append(p)

    return(Lp)

def read(file) :
    with open(file) as f:
        lines = f.readlines()

def save(var,name) :
    namefile="sample{}.txt".format(name)
    file = open(namefile, "w")
    file.write("%s = %s\n" %(name, var))
    file.close()


















