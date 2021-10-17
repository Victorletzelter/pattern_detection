#Code written by Victor LETZELTER
#Feel free to contact be for any requests : letzelter.victor@hotmail.fr

#This file corresponds to the algorithm for the events detection given the times series of the two probes.
#The results of this algorithm will, then, feed a neural network for the detection of the "Zonal Flows" events related to the data
#of a single probe.

#The detail of each step of the algorithm is provided in the Report "Ciemat__internship-VF.pdf". 

import os #Execution for the python scripts related to the algorithm.
os.chdir("/Users/victorletzelter/Documents/GitHub/pattern_detection/Python_project") #This line may have to be adapted

exec(open('Imports.py').read())
exec(open('Coherence.py').read())
exec(open('Non_coverage.py').read())
exec(open('Exp.py').read())

data1=pd.read_csv("/Users/victorletzelter/Documents/GitHub/pattern_detection/Python project/Data_files/converted_data1.txt",delimiter=' ')
data2=pd.read_csv("/Users/victorletzelter/Documents/GitHub/pattern_detection/Python project/Data_files/converted_data2.txt",delimiter=' ')

def split(dat,x1,x2) : #This function allows to split the data and to work with a specific part of it : the indexes between x1 and x2.
    l=len(dat)
    dat1=dat.tail(l-int(x1))
    dat1=dat1.head(int(x2-x1))
    return(dat1)

x1=450000
x2=500000 #bounds of indexes of the data on which the algorithm is applied.

X1=450000
X2=500000 #bounds used for the "hand-labelling part" whose usefulness is to adjust hyperparameters of the detection algorithm  
#(See the Hand_labeling.py file)

D=split(data1,x1,x2)
D2=split(data2,x1,x2)

D=round(D,2) #To improve calculation time
D2=round(D2,2)

#plt.plot(data1['Potential(V)'])
#plt.plot(data2['Potential(V)'])
#plt.show()
#plt.close()

exec(open('Hand_labeling.py').read())
exec(open('Stats.py').read())

### Smoothing the Signal, then Removing the offset by Savitzky-Golay method

Da=smooth(D,D2,x1,x2) #in RMS_gestion
D=Da[0]
D2=Da[1]

### Loading of the training part

exec(open("/Users/victorletzelter/Documents/GitHub/pattern_detection/Python_project/Data_files/samplezi.txt").read()) #Getting the zi variable

###Optimal values for the next step :

#DP and GP refer to "Dumping part" and "Growing part" respectively
#Min and max refer to "Dumping part" and "Growing part" respectively
#These lists will contain the indexes (point number) related to the beginning and the end of the pattern detected given the data of two probes. 

GP2_min=[]
DP2_min=[]
GP_min=[]
DP_min=[]

GP2_max=[]
DP2_max=[]
GP_max=[]
DP_max=[]

Rate_keep=80 #This variable provides quantifies the quality of the ZF intervals that are desired. It gives the proportion of the training
#examples which are desired to be detected on the same interval (between X1 and X2). With high values of the Rate_keep, but the precision
#of the algorithm will dicrease. On the contrary, low values of the Rate_keep will provide more selective but more precise results.

traitement(zi) #Intervals in the variables GP2_min,DP2_min,GP_min,DP_min,GP2_max,DP2_max,GP_max,DP_max

###Value of the thresold factor for the Coherence 
#The values N and F are hyperparameters which were adjusted using the Hand_lebeling.py file. Their significance can be found on the .pdf 
#report of the project (#N adjusts the maximum frequency to be considered , F is the size of the gaussian window for the STFT).

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

### Détection the Zonal Flows intervals 

###The values chosen : N=4, F=100 and S=100  are parameters of the event detection function which were adjusted (See the file R_Opitmisation.py)

def Detect(data,c1='blue',c2='red',NSD=NSD_opt,N=4,F=100,S=100) :

    #Detection of local mimimums
    indMin, _=find_peaks(-data['Potential(V)']+max(abs(data['Potential(V)'])),height=0)    #indMin : indices of the local minimums

    #Détection of local maximums

    indMax, _=find_peaks(data['Potential(V)'],height=0)    #indMin : indices of the local maximums

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

    ###Third filtration based on the non-coverage

    indMin_4=copy.deepcopy(indMin_3)
    indMin_4=update(indMin_3,-1,data)
    indMax_4=update(indMax_3,1,data)

    ###Fourth filtration based on the exponential decay

    ###Exp with the Data 1
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

#Execution of the function
 
I=Detect(D,'blue','red')
Int_min=I[0]
Int_max=I[1]
I2=Detect(D2,'green','orange')
Int_min2=I2[0]
Int_max2=I2[1]

###Next step : We keep only the times where the two intervals, deduced by D and D2, recover. This filter permits to improve the quality
#of the detection. 

def indicator(Int) : #This function gives 1 when events occur, and 0 in other cases

    ind=[0]*(x2-x1)
    ind=pd.DataFrame(ind)
    ind=ind.set_index(np.arange(x1,x2,1))

    for i in range (len(Int)) :

        for l in range (Int[i][0],Int[i][1]) :

            ind[0][l]=1

    return(ind[0])

ind_min=indicator(Int_min)
ind2_min=indicator(Int_min2)

ind_max=indicator(Int_max)
ind2_max=indicator(Int_max2)

produit_min=ind_min*ind2_min
produit_max=ind_max*ind2_max

def filtre(Int,MAX) : #MAX=1 for max et MAX=-1 for min

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

#F in a prefix for "Filtered" ; the following lists are the filtered intervals 

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

def fill(I_min,I_max,c1='blue',c2='red') : #The fill function fills the intervals zones

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

plt.grid()
plt.xlabel('Point Number')
plt.ylabel('Potential(V)')
plt.plot(D['Potential(V)'])
plt.plot(D2['Potential(V)'])

plt.show()

### Histograms and statistics :
#This part is related to the estimations of the parameters of the probabilitistic model which was designed to generate artificial data
#whose structure is as close as possible to the real data. The histograms of the values of the parameters in the real data, and the
#fit with the distribution probabilities deduced by the classical methods (MLE, MM) can be plot executing the following functions. 
#(See the pdf report and the Data_generator.py file)

### h value : length of the interval

def histH(DP,c='blue') :

    H=[]

    for e in DP :
        H.append(e[1]-e[0])

    plt.hist(H,color=c,ec='blue',alpha=0.4,density=True)

    return(H)

H=histH(F_Int_min,'blue') #Histogram of lengths of the intervals detected by the previous function
#K=histH(DP_min,'red') #Histogram of lengths of the labelled intervals 
plt.show()

#The histogram has the shape of a gamma distribution
def plot_gamma(H) :

    alpha=np.mean(H)**2/np.var(H) 
    beta=np.mean(H)/np.var(H) 
    X=np.arange(0,200,1)
    Y=[scipy.stats.gamma.pdf(e,alpha,0,1/beta) for e in X]
    plt.plot(X,Y)

    plt.show()

    return(alpha,beta)

alphaH,betaH=plot_gamma(H) #Fit of the histograms with a gamma distribution. 

#First simple estimiation
h=np.mean(H) 

### Histograms of the distance with the average value

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
histD(F_Int_min,F_Int_min2,1)
histD(F_Int_min,F_Int_min2,2)
plt.show() #Histogram of distances from the average value (for the detected, and the labelled intervals)

### Sigma_0 value

def histS(D) :
    hist, bin_edges=np.histogram(D['Potential(V)'],bins=50)
    plt.plot(hist)

histS(D)
histS(D2)
plt.show()

mu=np.mean(D['Potential(V)'])
sigma0=np.std(D['Potential(V)']) 
plt.show()

### Value of p (parameter of the geometric law for events appearance)
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

p=1/(np.mean(WT)) #Value deduced of the parameter p of the geometric law. 

### Values of a and b : the parameters of the exponential fit t->a*exp(b*(t-0))+c at each event

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

def Distributions_plots() :

    A=histA(DP_min,1) 
    A2=histA(F_Int_min)
    alphaA,betaA=plot_gamma([-e for e in A2]) 
    a=np.median(A) 
    B=histB(DP_min)
    B2=histB(F_Int_min)
    alphaB,betaB=plot_gamma([-e for e in B2])
    X=np.arange(-0.1,0.01,0.001)
    Y=[scipy.stats.gamma.pdf(-e,alphaB,0,1/betaB) for e in X]
    #plt.plot(X,Y)
    #plt.show()
    b=np.median(B)
    C=histC(DP_min)
    C2=histC(F_Int_min)
    alphaC,betaC=plot_gamma(C2) 
    c=np.median(C2)

def plot_gamma(H) : #Fit of the histogram of H value with a gammma distribution

    alpha=np.mean(H)**2/np.var(H) #4.150404428559912
    beta=1/(np.var(H)/np.mean(H)) #0.04294735710638756
    X=np.arange(0,200,1)
    Y=[scipy.stats.gamma.pdf(-e,alpha,0,1/beta) for e in X]
    plt.plot(X,Y)

    plt.show()

    return(alpha,beta)

### Performance evaluation : the next lines precise a way to evaluate the F_score of the automatic labelling algorithm : the F_Score. 
#For that purpose, the performance on the algorithm for Event labeling has feed executed on the same window as the Hand labeling algorithm. 

### Performance of the detection of the same interval :
def indicator(Int,x1,x2) :

        ind=[0]*(x2-x1)
        ind=pd.DataFrame(ind)
        ind=ind.set_index(np.arange(x1,x2,1))

        for i in range (len(Int)) :

            for l in range (Int[i][0],Int[i][1]) :

                ind[0][l]=1

        return(ind[0])

def filtre2(Int,produit_min,produit_max,MAX) : #MAX=1 pour max et MAX=-1 pour min

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

def F_score(DP_min,DP2_min,DP_max,DP2_max,F_Int_min,F_Int_min2,F_Int_max,F_Int_max2,x1,x2) :

    R1=indicator(F_Int_min,x1,x2)
    R2=indicator(F_Int_max,x1,x2)
    L1=indicator(DP_min,x1,x2)
    L2=indicator(DP_max,x1,x2)

    produit_min1=R1*L1
    produit_max1=R2*L2

    Confusion=filtre2(F_Int_min,produit_min1,produit_max1,-1) 

    Precision=len(Confusion)/len(F_Int_min) #Precision : ~28% 
    Recall=len(Confusion)/len(DP_min) #Recall : ~15% 

    F1_score1=s.harmonic_mean([Precision,Recall]) #F_score given the results in the first data

    R1=indicator(F_Int_min2)
    R2=indicator(F_Int_max2)
    L1=indicator(DP2_min)
    L2=indicator(DP2_max)

    produit_min=R1*L1
    produit_max=R2*L2

    Confusion2=filtre2(F_Int_min2,produit_min2,produit_max2,-1)

    Precision2=len(Confusion2)/len(F_Int_min2) #Precision : ~27.3% 
    Recall2=len(Confusion2)/len(DP2_min) #Recall : ~15% 

    F1_score2=s.harmonic_mean([Precision2,Recall2])

    F1_s=s.mean([F1_score1,F1_score2])

    return(F1_s)





