#This file contains the functions related statistics with regards to the data. These functions are used in particular in the 
#'Event_labeling.py' file. 

def smooth(D,D2,x1,x2) :

    S_1=savgol_filter(D['Potential(V)'],window_length=int((X2-X1)/10)+1,polyorder=2)
    S_2=savgol_filter(D2['Potential(V)'],window_length=int((X2-X1)/10)+1,polyorder=2)

    D_s=pd.DataFrame.copy(D)
    D2_s=pd.DataFrame.copy(D2)

    for i in range (x1,x2) :
        D_s['Potential(V)'][i]=S_1[i-x1]
        D2_s['Potential(V)'][i]=S_2[i-x1]

    for i in range (x1,x2) :
        D['Potential(V)'][i]=D['Potential(V)'][i]-D_s['Potential(V)'][i]
        D2['Potential(V)'][i]=D2['Potential(V)'][i]-D2_s['Potential(V)'][i]

    return((D,D2))

def smooth2(D,D2,x1,x2) :

    S_1=savgol_filter(D['Potential(V)'],int(len(D)/10)+1, 2)
    S_2=savgol_filter(D2['Potential(V)'],int(len(D2)/10)+1, 2)

    D_s=pd.DataFrame.copy(D)
    D2_s=pd.DataFrame.copy(D2)

    for i in range (x1,x2) :
        D_s['Potential(V)'][i]=S_1[i-x1]
        D2_s['Potential(V)'][i]=S_2[i-x1]

    return((D_s,D2_s))

def stats(D,D2) :

    global data1_m
    global data1_rms
    global data1_sd

    global data2_m
    global data2_rms
    global data2_sd

    data1_m=D.rolling(100).mean()
    data1_rms=np.sqrt(D.pow(2).rolling(100).apply(lambda x: np.sqrt(x.mean())))
    data1_sd=D.rolling(100).std()

    data2_m=D2.rolling(100).mean()
    data2_rms=np.sqrt(D2.pow(2).rolling(100).apply(lambda x: np.sqrt(x.mean())))
    data2_sd=D2.rolling(100).std()

#smooth(D,D2,x1,x2)
#stats(D,D2)

def plot_rms() :

    plt.plot(D['Potential(V)'])
    plt.plot(D2['Potential(V)'])
    plt.plot(data1_m['Potential(V)']+data1_sd['Potential(V)'],color='black')
    plt.plot(data1_m['Potential(V)']-data1_sd['Potential(V)'],color='black')
    plt.plot(data1_m['Potential(V)'])
    plt.show()

def isok_1(e,MAX,t,t_) : #MAX=1 if MAX, -1 if MIN
    if MAX==1 :
        if t[e] :
            return(1)
    else :
        if t_[e] :
            return(1)
    return(0)