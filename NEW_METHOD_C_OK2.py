def split(dat,x1,x2) :
    l=len(dat)
    dat1=dat.tail(l-int(x1))
    dat1=dat1.head(int(x2-x1))
    return(dat1)

def learning() :

    Dt=split(data1,X1,X2)
    D2t=split(data2,X1,X2)

    Da=smooth(Dt,D2t,X1,X2) #in RMS_gestion
    D=Da[0]
    D2=Da[1]

    t=Correlation_s(D,D2,seuil_opt,X1,X2) #Spectral correlation btw the 2 signals

    def fill2(t,c1='blue') :
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

    def onclick(event):
        if event.dblclick :
                zi.append((event.xdata,event.ydata))

        #print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
        #      ('double' if event.dblclick else 'single', event.button,
        #      event.x, event.y, event.xdata, event.ydata))

    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()

def traitement(zi) :

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

def fill(Int_min,Int_max,c1='blue',c2='red') :

    for i in range (0,len(Int_max)) :
        (a,b)=Int_max[i][0],Int_max[i][1]
        X=np.arange(a,b,1)
        ax.fill_between(X,-30 ,30, color=c1, alpha=0.1)

    for i in range (0,len(Int_min)) :
        (a,b)=Int_min[i][0],Int_min[i][1]
        X=np.arange(a,b,1)
        ax.fill_between(X,-30 ,30, color=c2, alpha=0.1)

def plot_mouse() :

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

def Coh_opt(DP,DP2,Rate_keep=95,N=4,F=100,X1=450000,X2=500000) :

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

def valeur_opt(DP,DP2,MAX,Rate_keep=80) : #MAX=1 if max

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

### Distance with the average ?

###For the mins
def Distances_opt(DP,DP2,Rate_keep=90,MAX=-1,S=100) :

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


def plotidist() :

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

#np.percentile(Distances1, 90)

#plt.grid()
#plt.hist(Distances1,20,color='blue',alpha=0.4)
#plt.hist(Distances2,20,color='orange',alpha=0.4)
#plt.title('Distance from the MA (in number of sd)')
#plt.show()

#np.mean(Distances1)
#-2.1596255864703964

#np.mean(Distances2)
#-2.1167086057968336


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

### We check if times are ok

def times_ok() :
    DT=[0]*(len(DP_min)+len(DP2_min))

    for i in range (len(DP_min)) :
        DT[i]=T[i]-(DP_min[i][1]-DP_min[i][0])

    for i in range (len(DP_min),len(DP2_min)+len(DP_min)) :
        DT[i]=T[i]-(DP2_min[i-len(DP_min)][1]-DP2_min[i-len(DP_min)][0])

###

def pourcentage_corr(N,F) :

    t=Correlation_s(D,D2,N,F) #Spectral correlation btw the 2 signals

    Cr=[]

    for e in DP_min :
        Cr.append(iscorr(e[0],t))

    for e in DP2_min :
        Cr.append(iscorr(e[0],t))

    Cr=np.asarray(Cr)

    p=len(np.where(Cr==1)[0])/len(Cr) #57% when N=3

    return(p)

#Xn=np.arange(10,100,5)
#Yn=[]
#for e in Xn :
#    Yn.append(pourcentage_corr(3,e))

#plt.plot(Xn,Yn)
#plt.show()

#Pour N=3, MAX pour F=30 et env 0.696

#Max d'une fonction à 2 var ?

#XN=np.arange(1,10,1)
#XF=np.arange(10,100,10)
#Y=np.zeros((len(XN),len(XF)))
#for i in range (len(XN)) :
#    for j in range(len(XF)) :
#        Y[i,j]=pourcentage_corr(XN[i],XF[j])

#print(Y[np.where(Y==np.max(Y))]) #0.826086
#Nopt=XN[np.where(Y==np.max(Y))[0][0]] #Nopt=1
#Fopt=XF[np.where(Y==np.max(Y))[1][0]] #Fopt=90

###

def times_rms(data,S) :
    data_m=data.rolling(S).mean()
    data_rms=np.sqrt(data.pow(2).rolling(S).apply(lambda x: np.sqrt(x.mean())))
    data_sd=data.rolling(S).std()
    sup=data_m['Potential(V)']+data_sd['Potential(V)']
    inf=data_m['Potential(V)']-data_sd['Potential(V)']
    t=(data['Potential(V)']>=sup).astype(int)
    t_=(data['Potential(V)']<=inf).astype(int)
    return(t,t_)

def pourcentage(S) :

    t,t_=times_rms(D,S)
    t2,t2_=times_rms(D2,S)

    ISOK=[]

    for e in DP_min :
        ISOK.append(isok_1(e[0],-1,t,t_))

    for e in DP2_min :
        ISOK.append(isok_1(e[0],-1,t2,t2_))

    ISOK=np.asarray(ISOK)

    p2=len(np.where(ISOK==1)[0])/len(ISOK) #88% when S=100

    return(p2)

#X=np.arange(10,200,10)
#X=np.arange(40,60,2)

#Y=[]
#for e in X :
#    Y.append(pourcentage(e))

#S=X[Y.index(max(Y))] #S=50 / p2=96.4%

###Optimal parameters

def opt_para() :
    XN=np.arange(3,7,1)
    XF=np.arange(50,110,10)
    XS=np.arange(50,140,10)
    M=np.zeros((len(XN),len(XF),len(XS)))

    Tot=len(XN)*len(XF)*len(XS)

    start_time = time.time()

    M[0,0,0]=pourcentage_corr(XN[0],XF[0])*pourcentage(XS[0])

    print("--- %s seconds ---" % (time.time() - start_time))

    N=0
    start_time = time.time()

    for i in range (len(XN)) :
        for j in range (len(XF)) :
            for k in range (len(XS)) :
                M[i,j,k]=pourcentage_corr(XN[i],XF[j])*pourcentage(XS[k])
                N+=1
                print(N/Tot)

    print("--- %s seconds ---" % (time.time() - start_time))

    liste=[]
    for i in range (len(XN)) :
        for j in range (len(XF)) :
            for k in range (len(XS)) :
                if M[i,j,k]==np.max(M) :
                    liste.append((i,j,k))

    #--- 5216.969802618027 seconds ---

def tests_para() :
    ### Test1
    W=np.where(M==np.max(M)) #Max : 58%
    #W=(array([0, 0, 0]), array([4, 4, 4]), array([3, 4, 5]))
    Nopt2=XN[4] #=5
    Fopt2=XF[4] #=90
    Sopt2=XS[4] #=90

    #OR
    Nopt2=XN[3] #=4
    Fopt2=XF[4] #=90
    Sopt2=XS[5] #=110

    ###Test 2

    #np.max(M)
    #Max value 0.6228373702422145
    #Index (array([0, 1]), array([0, 0]), array([0, 0]))
    #XNopt=XN[
    #XF=np.arange(50,110,10)
    #XS=np.arange(50,140,10)

    #liste
    #[(0, 0, 0), (1, 0, 0)]

    #NoptV2=XN[0] 3 or 4
    #FoptV2=XF[0] 50
    #SoptV2=XS[0] 50

###Filter on the length

def indicatrice(Int) :

    ind=[0]*(X2-X1)
    ind=pd.DataFrame(ind)
    ind=ind.set_index(np.arange(X1,X2,1))

    for i in range (len(Int)) :

        for l in range (Int[i][0],Int[i][1]) :

            ind[0][l]=1

    return(ind[0])

def liste_quotient(Int,Int2) : #This Function provides the list of length quotients between the two intervals that recover

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

def brouillon() :
    Lp=liste_quotient(DP_min,DP2_min)
    Lp2=liste_quotient(DP_max,DP2_max)

    plt.hist(Lp)
    plt.hist(Lp2)
    p_opt=min(np.percentile(Lp, 10),np.percentile(Lp2, 10))

    plt.show()








def random_grid() :

    ### Test with a random grid
    ### DO NOT LAUNCH

    import random as rd

    XN=np.arange(2,6,1)
    XF=np.arange(50,120,10)
    XS=np.arange(50,120,10)

    Shp=len(XN)*len(XF)*len(XS)

    np.random.seed(seed=12345)

    XN=np.random.randint(2,6,(1,Shp))
    XF=np.random.randint(50,120,(1,Shp))
    XS=np.random.randint(50,120,(1,Shp))

    XN=XN[0].tolist()
    XF=XF[0].tolist()
    XS=XS[0].tolist()

    #from mpl_toolkits import mplot3d
    #fig = plt.figure()
    #ax = plt.axes(projection='3d')
    #ax.set_xlabel('XN')
    #ax.set_ylabel('XF')
    #ax.set_zlabel('XS');
    #ax.scatter3D(XN, XF, XS, cmap='Greens');
    #plt.show()

    M2=np.zeros((len(XN),len(XF),len(XS)))
    N=0

    for i in range (len(XN)) :
        for j in range (len(XF)) :
            for k in range (len(XS)) :
                M2[i,j,k]=pourcentage_corr(XN[i],XF[j])*pourcentage(XS[k])
                N+=1
            print(N/Shp)

def read(file) :
    with open(file) as f:
        lines = f.readlines()

def save(var,name) :
    namefile="sample{}.txt".format(name)
    file = open(namefile, "w")
    file.write("%s = %s\n" %(name, var))
    file.close()

def apprentissage_essais() :

    def onpick(event):
        if event.artist != line:
            return
        n = len(event.ind)
        if not n:
            return
        fig, axs = plt.subplots(n, squeeze=False)
        for dataind, ax in zip(event.ind, axs.flat):
            ax.plot(X[dataind])
            ax.text(0.05, 0.9,
                    f"$\\mu$={xs[dataind]:1.3f}\n$\\sigma$={ys[dataind]:1.3f}",
                    transform=ax.transAxes, verticalalignment='top')
            ax.set_ylim(-0.5, 1.5)
        fig.show()
        return True


    fig.canvas.mpl_connect('pick_event', onpick)
    plt.show()


    fig, ax = plt.subplots()
    ax.set_title('click on points')

    line, = ax.plot(np.random.rand(100), 'o',
                    picker=True, pickradius=5)  # 5 points tolerance

    def onpick(event):
        thisline = event.artist
        xdata = thisline.get_xdata()
        ydata = thisline.get_ydata()
        ind = event.ind
        points = tuple(zip(xdata[ind], ydata[ind]))
        print('onpick points:', points)

    fig.canvas.mpl_connect('pick_event', onpick)

    plt.show()


















