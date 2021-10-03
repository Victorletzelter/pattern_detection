import os
import time

start_time = time.time()

print("Current Working Directory " , os.getcwd())
os.chdir("/Users/victorletzelter/Desktop/Projet_python")

exec(open('imports_gestion.py').read())
exec(open('Coherence_gestion.py').read())
exec(open('LRC_gestion.py').read())
exec(open('Non_coverage.py').read())
exec(open('Exp_gestion_3.py').read())

#exec(open('Synthese.py').read())

os.chdir('/Users/victorletzelter/Desktop/Dat')

data1=pd.read_csv("/Users/victorletzelter/Desktop/Dat/converted_data1.txt",delimiter=' ')
data2=pd.read_csv("/Users/victorletzelter/Desktop/Dat/converted_data2.txt",delimiter=' ')

#plt.plot(data1['Potential(V)'])
#plt.plot(data2['Potential(V)'])
#plt.show()

def split(dat,x1,x2) :
    l=len(dat)
    dat1=dat.tail(l-int(x1))
    dat1=dat1.head(int(x2-x1))
    return(dat1)

x1=450000
x2=500000

X1=450000
X2=500000 #bounds used for the training

D=split(data1,x1,x2)
D2=split(data2,x1,x2)

D=round(D,2) #To improve calculation time
D2=round(D2,2)

plt.plot(data1['Potential(V)'])
plt.plot(data2['Potential(V)'])
plt.show()
#plt.close()

os.chdir("/Users/victorletzelter/Desktop/Projet_python")
exec(open('RMS_gestion_2.py').read())
exec(open('/Users/victorletzelter/Desktop/Projet_python/NEW_METHOD_C_OK2.py').read())


### Smoothing the Signal, then Removing the offset by Savitzky-Golay method

Da=smooth(D,D2,x1,x2) #in RMS_gestion
D=Da[0]
D2=Da[1]

### Loading of the training part

exec(open("/Users/victorletzelter/Desktop/Projet_python/sampleziNEW4.txt").read()) #Getting the zi variable

###Optimal values for the next step :

GP2_min=[]
DP2_min=[]
GP_min=[]
DP_min=[]

GP2_max=[]
DP2_max=[]
GP_max=[]
DP_max=[]
Rate_keep=80

traitement(zi) #Intervals in the variables GP2_min,DP2_min,GP_min,DP_min,GP2_max,DP2_max,GP_max,DP_max

###Optimal values for N, F and S

###GENERAL CODE !!!!(Values chosen for the moment N=4, F=100 and S=100)

###Value of the thresold factor for the coherence

seuil_min=Coh_opt(DP_min,DP2_min,Rate_keep,N=4,F=100,X1=450000,X2=500000)
seuil_max=Coh_opt(DP_max,DP2_max,Rate_keep,N=4,F=100,X1=450000,X2=500000)
seuil_opt=min(seuil_min,seuil_max)

###Optimal values for the distance of the peak with the MA (in number of moving SD)

NSD_min=Distances_opt(DP_min,DP2_min,Rate_keep=Rate_keep,MAX=-1,S=100)
NSD_max=Distances_opt(DP_max,DP2_max,Rate_keep=Rate_keep,MAX=1,S=100) #Not enough data for the maxs
NSD_opt=min(NSD_min,NSD_max)

###Optimal values of the errors on the exponential fitting

a_eropt_min,b_eropt_min,c_eropt_min=valeur_opt(DP_min,DP2_min,MAX=-1,Rate_keep=Rate_keep)
a_eropt_max,b_eropt_max,c_eropt_max=valeur_opt(DP_max,DP2_max,MAX=1,Rate_keep=Rate_keep)

###Optimal values of the coverage of the invertals thresold rate

Lp=liste_quotient(DP_min,DP2_min)
Lp2=liste_quotient(DP_max,DP2_max)
p_opt=min(np.percentile(Lp, 100-Rate_keep),np.percentile(Lp2, 100-Rate_keep))

### Détection of the intervals of Zonal Flows

def Detect(data,c1='blue',c2='red',NSD=NSD_opt,N=4,F=100,S=100) :

    #data=D
    #N=4
    #F=100
    #S=100
    #NSD=NSD_opt

    #Detection des minimums locaux
    indMin, _=find_peaks(-data['Potential(V)']+max(abs(data['Potential(V)'])),height=0)    #indMin : indices des minimums locaux

    #Détection des maximums locaux

    indMax, _=find_peaks(data['Potential(V)'],height=0)    #indMin : indices des minimums locaux

    #We keep only the peaks where the Corr is >0.8 / we use a confidence interval

    indMin=indMin+x1
    indMax=indMax+x1

    ###First filtration based on LRC :

    indMin_2=[]
    indMax_2=[]

    t=Correlation_s(D,D2,seuil_opt,x1,x2,N,F) #Spectral correlation btw the 2 signals

    for e in indMax :
        if iscorr(e,t,100) :
            indMax_2.append(e)

    for e in indMin :
        if iscorr(e,t,100) :
            indMin_2.append(e)

    ###Second filtration based on the RMS / STD :

    indMin_3=[]
    indMax_3=[]

    def times_rms(data,S) :
        data_m=data.rolling(S).mean()
        data_rms=np.sqrt(data.pow(2).rolling(S).apply(lambda x: np.sqrt(x.mean())))
        data_sd=data.rolling(S).std()
        sup=data_m['Potential(V)']+NSD*data_sd['Potential(V)']
        inf=data_m['Potential(V)']-NSD*data_sd['Potential(V)']
        t=(data['Potential(V)']>=sup).astype(int)
        t_=(data['Potential(V)']<=inf).astype(int)
        return(t,t_)

    t,t_=times_rms(data,S)

    for e in indMax_2 :
        if isok_1(e,1,t,t_) :
            indMax_3.append(e)

    for e in indMin_2 :
        if isok_1(e,-1,t,t_) :
            indMin_3.append(e)

    ### Third filtration based on the non-coverage

    indMin_4=copy.deepcopy(indMin_3)
    indMin_4=update(indMin_3,-1,data)
    indMax_4=update(indMax_3,1,data)

    ###Fourth filtration based on the exponential decay

    ### Exp with the Data 1
    filtre_min=filtre_exp_min(indMin_4,data)
    indMin_5=filtre_min[0]
    Sorties_min=filtre_min[1]
    filtre_max=filtre_exp_max(indMax_4,data)
    indMax_5=filtre_max[0]
    Sorties_max=filtre_max[1]

    indMin_6=indMin_5
    indMax_6=indMax_5

    ###Deduction of the ZF intervals :
    Int_max=[]

    for i in range (0,len(filtre_max[0])) :
        x=indMax_6[i]
        y=Sorties_max[i]
        Int_max.append((x,x+10*(y+1)))
        X=np.arange(x,x+10*(y+1),1)
        plt.fill_between(X,-30 ,30, color=c1, alpha=0.1)

    Int_min=[]

    for i in range (0,len(filtre_min[0])) :
        x=indMin_6[i]
        y=Sorties_min[i]
        Int_min.append((x,x+10*(y+1)))
        X=np.arange(x,x+10*(y+1),1)
        plt.fill_between(X,-30,30, color=c2, alpha=0.1)

    return((Int_min,Int_max))

I=Detect(D,'blue','red')
Int_min=I[0]
Int_max=I[1]
I2=Detect(D2,'green','orange')
Int_min2=I2[0]
Int_max2=I2[1]

###Next step : We keep only the times where the two intervals recover (red/orange or blue/green)

def indicatrice(Int) :

    ind=[0]*(x2-x1)
    ind=pd.DataFrame(ind)
    ind=ind.set_index(np.arange(x1,x2,1))

    for i in range (len(Int)) :

        for l in range (Int[i][0],Int[i][1]) :

            ind[0][l]=1

    return(ind[0])

ind_min=indicatrice(Int_min)
ind2_min=indicatrice(Int_min2)

ind_max=indicatrice(Int_max)
ind2_max=indicatrice(Int_max2)

produit_min=ind_min*ind2_min
produit_max=ind_max*ind2_max

def filtre(Int,MAX) : #MAX=1 pour max et MAX=-1 pour min

    F_Int=[]

    for (a,b) in Int :

        keep=0 #We do not keep

        for e in range (a,b) :

            if MAX==-1 :
                if produit_min[e]==1 :
                    keep=1

            elif MAX==1 :
                if produit_max[e]==1 :
                    keep=1

        if keep==1 :

            F_Int.append((a,b))

    return(F_Int)

F_Int_min=filtre(Int_min,-1)
F_Int_max=filtre(Int_max,1)
F_Int_min2=filtre(Int_min2,-1)
F_Int_max2=filtre(Int_max2,1)

###Next step again : We keep only, among the intervals that recover, those for which the quotient of the lengths remains raisonable

def filtre_length(Int,Int2,MAX) :

    F_Int=[]
    F_Int2=[]

    for (a,b) in Int :
        for (c,d) in Int2 :

            if (a<c and b>c) or (a<c and d<b) or (c<a and d>a) or (c<a and b<d) : #The two intervals recover
                if MAX==1 :
                    liste=[produit_max[e] for e in range (min(a,c),max(b,d))]
                else :
                    liste=[produit_min[e] for e in range (min(a,c),max(b,d))]

                n=len(liste)
                liste=np.asarray(liste)
                p=len(np.where(liste==1)[0])/n

                if p>=p_opt :
                    if not((a,b) in F_Int) :
                        F_Int.append((a,b))
                    if not((c,d) in F_Int2) :
                        F_Int2.append((c,d))

    return(F_Int,F_Int2)

F_Int_min,F_Int_min2=filtre_length(F_Int_min,F_Int_min2,-1)
F_Int_max,F_Int_max2=filtre_length(F_Int_max,F_Int_max2,1)

def fill(I_min,I_max,c1='blue',c2='red') :

    for i in range (0,len(I_max)) :
        (a,b)=I_max[i][0],I_max[i][1]
        X=np.arange(a,b,1)
        plt.fill_between(X,-30 ,30, color=c1, alpha=0.1)

    for i in range (0,len(I_min)) :
        (a,b)=I_min[i][0],I_min[i][1]
        X=np.arange(a,b,1)
        plt.fill_between(X,-30 ,30, color=c2, alpha=0.1)

plt.close()

fill(F_Int_min,F_Int_max,'blue','red')
fill(F_Int_min2,F_Int_max2,'green','orange')

#fill(DP_min,DP_max,'grey','grey')
#fill(DP2_min,DP2_max,'olive','olive')

Vide=[]

#fill(Confusion,[],'blue','red')
#fill(Confusion2,[],'green','orange')

plt.grid()
plt.xlabel('Point Number')
plt.ylabel('Potential(V)')
plt.plot(D['Potential(V)'])
plt.plot(D2['Potential(V)'])

plt.show()

### Histograms and statistics :

### h value : length of the interval

def histH(DP,c='blue') :

    H=[]

    for e in DP :
        H.append(e[1]-e[0])

    plt.hist(H,color=c,ec='blue',alpha=0.4,density=True)

    return(H)

H=histH(F_Int_min,'blue')
#K=histH(DP_min,'red')
plt.show()

#Gamma distribution ?
def plot_gamma(H) :

    alpha=np.mean(H)**2/np.var(H) #4.150404428559912
    beta=np.mean(H)/np.var(H) #0.04294735710638756
    X=np.arange(0,200,1)
    Y=[scipy.stats.gamma.pdf(e,alpha,0,1/beta) for e in X]
    plt.plot(X,Y)

    plt.show()

    return(alpha,beta)

alphaH,betaH=plot_gamma(H)
#400000-525000,86 : alpha : 5.666006367173681, beta : 0.05205518217191368

#First simple estimiation
h=np.mean(H) #50.425
#108.84615384615384

### Distance from the average value :

def histD(DP,DP2,Numero=1,Rate_keep=90,MAX=-1,S=100) :

    if Numero==1 :

        Dt=split(data1,X1,X2)

        Distances1=[]
        D_m=Dt.rolling(S).mean()['Potential(V)']
        D_sd=Dt.rolling(S).std()['Potential(V)']
        D_rms=np.sqrt(Dt.pow(2).rolling(S).apply(lambda x: np.sqrt(x.mean())))['Potential(V)']
        for elt in DP :
            Distances1.append((Dt['Potential(V)'][elt[0]]-D_m[elt[0]])/D_sd[elt[0]])

        plt.hist(Distances1,bins=20,color='blue',ec='blue',alpha=0.4)

    elif Numero==2 :

        D2t=split(data2,X1,X2)
        Distances2=[]
        D2_m=D2t.rolling(S).mean()['Potential(V)']
        D2_sd=D2t.rolling(S).std()['Potential(V)']
        D2_rms=np.sqrt(D2t.pow(2).rolling(S).apply(lambda x: np.sqrt(x.mean())))['Potential(V)']

        for elt in DP2 :
            Distances2.append((D2t['Potential(V)'][elt[0]]-D2_m[elt[0]])/D2_sd[elt[0]])

        plt.hist(Distances2,bins=20,color='red',ec='red',alpha=0.4)


histD(DP_min,DP2_min,1)
histD(DP_min,DP2_min,2)
plt.show()
histD(F_Int_min,F_Int_min2,1)
histD(F_Int_min,F_Int_min2,2)
plt.show()

### Sigma_0 value

def histS(D) :
    hist, bin_edges=np.histogram(D['Potential(V)'],bins=50)
    plt.plot(hist)

histS(D)
histS(D2)
plt.show()

mu=np.mean(D['Potential(V)'])
sigma0=np.std(D['Potential(V)']) #5.866104253900185
#sigma0 = 6.009092136913768

plt.show()

### Value of p :
#Sort DP_min :

def sort(DP) :
    DP_ini=np.asarray([e[0] for e in DP])
    Indexes=list(np.argsort(DP_ini,axis=0))
    Indexes=np.asarray(Indexes)
    DP_sorted=[DP[k] for k in Indexes]
    return(DP_sorted)

def histP(DP) :
    DP_sorted=sort(DP)
    WT=[]

    for i in range (len(DP_sorted)) :
        if i>0 :
            WT.append(DP_sorted[i][0]-DP_sorted[i-1][1]+1)

    plt.hist(WT,bins=50,color='red',ec='red',alpha=0.4)
    return(WT)

WT=histP(DP_min)
plt.show()

p=1/(np.mean(WT)) #0.0008902686785216975

### Values of a and b :

def histA(DP,Numero=1) :

    A=[]

    for e in DP :

        if Numero==1 :
            a=score_min(e[0],D)[2][0]

        elif Numero==2 :
            a=score_min(e[0],D2)[2][0]

        A.append(a)

    plt.hist(A,density=True)

    return(A)

def histB(DP,Numero=1) :

    B=[]

    for e in DP :
        if Numero==1 :
            b=score_min(e[0],D)[2][1]
        elif Numero==2 :
            b=score_min(e[0],D2)[2][1]

        B.append(b)

    plt.hist(B,density=True)

    return(B)

def histC(DP,Numero=1) :

    C=[]

    for e in DP :
        if Numero==1 :
            c=score_min(e[0],D)[2][2]
        elif Numero==2 :
            c=score_min(e[0],D2)[2][2]

        C.append(c)

    plt.hist(C,density=True)

    return(C)

A=histA(DP_min,1) #-22.231832043504525
A2=histA(F_Int_min)
alphaA,betaA=plot_gamma([-e for e in A2]) #alphaA=6.077479483801698
#betaA=0.4723282091501369
#X=np.arange(-50,0,1)
#Y=[scipy.stats.gamma.pdf(-e,alphaA,0,1/betaA) for e in X]
#plt.plot(X,Y)
a=np.median(A) #pour A : -14.790542450583118 pour A2 : -13.570775134998874
plt.show()
B=histB(DP_min)
B2=histB(F_Int_min)
alphaB,betaB=plot_gamma([-e for e in B2]) #alphaB=8.340017819817685
#betaB=213.4374537734773
X=np.arange(-0.1,0.01,0.001)
Y=[scipy.stats.gamma.pdf(-e,alphaB,0,1/betaB) for e in X]
plt.plot(X,Y)
plt.show()
b=np.median(B) #pour B : -0.046185968401266285 pour B2 :  -0.04122607345704972
plt.show()
C=histC(DP_min)
C2=histC(F_Int_min)
alphaC,betaC=plot_gamma(C2) #alphaC=1.337078744676507, betaC=0.2583374503229954
c=np.median(C2) #pour C : 4.219917756500447 pour C2 : 4.33025818557546
plt.show()


def plot_gamma(H) :

    alpha=np.mean(H)**2/np.var(H) #4.150404428559912
    beta=1/(np.var(H)/np.mean(H)) #0.04294735710638756
    X=np.arange(0,200,1)
    Y=[scipy.stats.gamma.pdf(-e,alpha,0,1/beta) for e in X]
    plt.plot(X,Y)

    plt.show()

    return(alpha,beta)



### Performance of the detection of the same interval :

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

Precision2=len(Confusion2)/len(F_Int_min2) #Precision : Env 27.3% (Idem pour le 2 : Vrais Positifs / (Vrais positifs + Faux positifs) "Proportion de vrais positifs parmis ceux sélectionnés"
Recall2=len(Confusion2)/len(DP2_min) #Recall : Env 15% (Idem pour le 2 : Vrais Positifs / (Vrais positifs + Faux négatifs) "Proportion de vrais positifs parmis ceux qui sont réellement positifs

F1_score2=s.harmonic_mean([Precision2,Recall2])

F1_s=s.mean([F1_score1,F1_score2])

print("--- %s seconds ---" % (time.time() - start_time))









#Confusion matrix ?

l=[]

for e in F_Int_min :









plt.show()

print("--- %s seconds ---" % (time.time() - start_time))

b=[]

for e in DP_min :
    b.append(score_min(e[0],D)[2][1])

plt.hist(b)
plt.show()



















#fill(DP_min,DP_max,c1='blue',c2='red')
#fill(DP2_min,DP2_max,c1='green',c2='orange')

plt.show()

def indicatrice(Int,x1=X1,x2=X2) :

    ind=[0]*(x2-x1)
    ind=pd.DataFrame(ind)
    ind=ind.set_index(np.arange(x1,x2,1))

    for i in range (len(Int)) :

        for l in range (Int[i][0],Int[i][1]) :

            ind[0][l]=1

    return(ind[0])

def bay() :
    I1_min=indicatrice(F_Int_min,x1,x2)
    I2_min=indicatrice(F_Int_min2,x1,x2)

    I1_max=indicatrice(F_Int_max,x1,x2)
    I2_max=indicatrice(F_Int_max2,x1,x2)

    #stock=x2
    x2=530000

    fichier=open("labelled_dat_min_V1.txt", "w")
    fichier.write('{}\t{}\t{}\t{}\t{}\n'.format('Point number','Data1','ZF_min_data1','Data2','ZF_min_data2'))

    for i in range (x1,x2) :
            fichier.write('{}\t{}\t{}\t{}\t{}\n'.format(i,round(D['Potential(V)'][i],2),I1_min[i],round(D2['Potential(V)'][i],3),I2_min[i]))

    fichier=open("labelled_dat_max_V1.txt", "w")
    fichier.write('{}\t{}\t{}\t{}\t{}\n'.format('Point number','Data1','ZF_min_data1','Data2','ZF_min_data2'))

    for i in range (x1,x2) :
            fichier.write('{}\t{}\t{}\t{}\t{}\n'.format(i,round(D['Potential(V)'][i],2),I1_max[i],round(D2['Potential(V)'][i],3),I2_max[i]))




























##

def bb()
    S=100
    data=D2
    data_m=data.rolling(S).mean()
    data_rms=np.sqrt(data.pow(2).rolling(S).apply(lambda x: np.sqrt(x.mean())))
    data_sd=data.rolling(S).std()
    sup=data_m['Potential(V)']+data_sd['Potential(V)']
    inf=data_m['Potential(V)']-data_sd['Potential(V)']

    plt.plot(D['Potential(V)'])
    plt.plot(D2['Potential(V)'])
    plt.plot(data_m['Potential(V)']+data_sd['Potential(V)'],color='black',linewidth=0.5)
    plt.plot(data_m['Potential(V)']-data_sd['Potential(V)'],color='black',linewidth=0.5)
    plt.fill_between(np.arange(450000,500000),data_m['Potential(V)']-data_sd['Potential(V)'] ,data_m['Potential(V)']+data_sd['Potential(V)'], color='blue', alpha=0.2)
    plt.plot(data_m['Potential(V)'],linestyle='--',color='black',linewidth=0.5)
    plt.xlabel('Point number')
    plt.ylabel('Potential(V)')
    #plt.legend()
    plt.title('Filtration based on the Bollinger Bands of the signal')
































###OTHER FUNCTIONS

### Aim : enlarge to get the complete ZF intervals

Entree_correcte_min=[0]*len(indMin_6)
Entree_correcte_max=[0]*len(indMax_6)

def Entree_min(indMin_6) :
    for i in range (0,len(indMin_6)) :

        e=indMin_6[i]

        if t[e]==1 :

            while t[e]==1 :
                e=e-1

            Entree_correcte_min[i]=e

        else :
            Entree_correcte_min[i]=e

    return(Entree_correcte_min)

def Entree_max(indMax_6) :
    for i in range (0,len(indMax_6)) :

        e=indMax_6[i]

        if t[e]==1 :

            while t[e]==1 :
                e=e-1

            Entree_correcte_max[i]=e

        else :
            Entree_correcte_max[i]=e

    return(Entree_correcte_max)

Entree_correcte_min=Entree_min(indMin_6)
Entree_correcte_max=Entree_max(indMax_6)

Int_max=[]

for i in range (0,len(indMax_6)) :
    x=indMax_6[i]
    x_min=Entree_correcte_max[i]
    y=min(Sorties_max[i],Sorties_max_2[i])
    Int_max.append((x,x+10*(y+1)))
    X=np.arange(x_min,x+10*(y+1),1)
    plt.fill_between(X,-30 ,30, color='blue', alpha=0.1)

Int_min=[]

for i in range (0,len(indMax_6)) :
    x=indMin_6[i]
    x_min=Entree_correcte_min[i]
    y=min(Sorties_min[i],Sorties_min_2[i])
    Int_min.append((x_min,x+10*(y+1)))
    X=np.arange(x_min,x+10*(y+1),1)
    plt.fill_between(X,-30,30, color='red', alpha=0.1)


plt.grid()
plt.title('Damping part of a ZF')
plt.xlabel('Point Number')
plt.ylabel('Potential(V)')
plt.plot(D['Potential(V)'])
plt.plot(D2['Potential(V)'])

plt.show()

print("--- %s seconds ---" % (time.time() - start_time))

plt.close()


def eltc(l1,l2) :
    N=0
    for e in l1 :
        if e in l2 :
            N+=1
    return(N)

###Normal law?

#plt.close()
#hist, bin_edges=np.histogram(D['Potential(V)'],bins=50)
#plt.plot(hist)
#plt.show()

#statsmodels.graphics.gofplots.qqplot(D['Potential(V)'])

#plt.show()





