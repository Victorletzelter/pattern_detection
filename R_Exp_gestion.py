from sklearn.metrics import mean_squared_error
from math import *
import warnings
from scipy.optimize import OptimizeWarning

def func(x,popt) :
#x in a point number
#popt is a list containing the three values of the coefficients

#This function returns the value of the exponential
    a=popt[0]
    b=popt[1]
    c=popt[2]
    return(a*np.exp(b*x)+c)

# Get the optimal size of the interval :
def opt(e,MAX,data) :
#e in the point number of the extremum
#MAX : 1 if the extremum is a maximum and -1 if it is a minimum
#data is the source of data

#This function returns the optimal size of the interval for exponential fitting

    Er=0 #While Er=0, there was no warnings
    a_e=[100]*20
    b_e=[100]*20
    c_e=[100]*20
    l_err=[100]*20
    RMSE=[100]*20
    j=2

    while j<20 and e+10+10*j<len(data['Potential(V)'])+data.index[0] :
        E=0
        X=np.arange(0,10+10*j,1)
        Dat=[data['Potential(V)'][e+i] for i in np.arange(0,10+10*j,1)]

        with warnings.catch_warnings():
            warnings.simplefilter("error", OptimizeWarning)
            warnings.simplefilter("error", RuntimeWarning)
            warnings.simplefilter("error", RuntimeError)

            try:
                if MAX==-1 :
                    popt,pcov = scipy.optimize.curve_fit(lambda t,a,b,c: a*np.exp(b*t)+c,  X,  Dat, p0=(-1,-1/100,1),bounds=((-np.inf,-1/20,-np.inf),(0,-1/300,np.inf)),maxfev=800)

                elif MAX==1 :
                    popt,pcov = scipy.optimize.curve_fit(lambda t,a,b,c: a*np.exp(b*t)+c,  X,  Dat, p0=(1,-2/75,1),bounds=((0,-1/20,-np.inf),(np.inf,-1/300,np.inf)),maxfev=800)

                a_er=np.sqrt(np.diag(pcov))[0]/len(X)
                b_er=np.sqrt(np.diag(pcov))[1]/len(X)
                c_er=np.sqrt(np.diag(pcov))[2]/len(X)

            except OptimizeWarning:
                Er=1
                print('Maxed out calls.')

            except RuntimeError:
                print("Error - curve_fit failed")
                Er=1

            except RuntimeWarning :
                print("Invalid value")
                Er=1

            except Warning :
                print('Warning was raised as an exception!')
                Er=1

            except FloatingPointError :
                print('Warning was raised as an exception!')
                Er=1

        if Er==0 :
            a_e[j]=a_er
            b_e[j]=b_er
            c_e[j]=c_er
            l_err[j]=(a_er+100*b_er+c_er)/102
            RMSE[j]=sqrt(mean_squared_error(Dat, func(X,popt)))

        else :
            a_e[j]=100
            b_e[j]=100
            c_e[j]=100
            l_err[j]=100
            RMSE[j]=100

        j+=1

    min_value_2 = min(RMSE)
    min_index_2 = RMSE.index(min_value_2)

    min_value = min(l_err)
    min_index = l_err.index(min_value)

    if np.max(l_err)!=np.min(l_err) and np.max(RMSE)!=np.min(RMSE) :

        mix=0.999*(l_err-np.min(l_err))/(np.max(l_err)-np.min(l_err))+0.001*(RMSE-np.min(RMSE))/(np.max(RMSE)-np.min(RMSE))

    else :

        mix=[100]*15

    min_value_3=np.min(mix)
    min_index_3 = np.where(mix==min_value_3)[0][0]

    return(min_index_3,mix,RMSE)

def filtre_exp_max(indMax_4,data) :

    indMax_5=[]
    Sorties_max=[]

    for e in indMax_4 :

        res=score_max(e,data)
        score=res[0]
        s=res[1]

        #if score[0]<1.5 and score[1]<0.02 and score[1]>0 and score[2]<1.5  :
        #if score[0]<a_eropt_max and score[1]<b_eropt_max and score[1]>0 and score[2]<c_eropt_max and (abs(coeff[1]+0.05))>epsi and (abs(coeff[1]+1/300)>epsi) :
        if score[0]<a_eropt_max and score[1]<b_eropt_max and score[1]>0 and score[2]<c_eropt_max :
            indMax_5.append(e)
            Sorties_max.append(s)

    return(indMax_5,Sorties_max)

#Principe :

def filtre_exp_min(indMin_4,data) :

    indMin_5=[]
    Sorties_min=[]

    for e in indMin_4 :

        res=score_min(e,data)
        s=res[1]
        score=res[0]
        #if score[0]<2.5 and score[1]<0.015 and score[1]>0 and score[2]<2.5 and s>2 :

        #if score[0]<a_eropt_min and score[1]<b_eropt_min and score[1]>0 and score[2]<c_eropt_min and (coeff[1]+0.05)>epsi and (coeff[1]+1/300>epsi) :
        if score[0]<a_eropt_min and score[1]<b_eropt_min and score[1]>0 and score[2]<c_eropt_min :
            indMin_5.append(e)
            Sorties_min.append(s)

    return(indMin_5,Sorties_min)

def score_min(e,data) : #This function provides the error associated with the exponential fit for the point e, when it is a minimum
    E=0 #While E=0, there was no warnings
    min_index,l_err,RMSE=opt(e,-1,data)
    X=np.arange(0,10+10*min_index,1)

    Dat=[data['Potential(V)'][e+i] for i in np.arange(0,min(10+10*min_index,data.index[len(data)-1]+1-e),1)]

    if len(X)==len(Dat) :

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

    else :
        E=1

    if E==0 :
        return((a_er,b_er,c_er),min_index,popt)

    else :
        return((100,100,100),min_index,popt)

def score_max(e,data) :

    E=0
    min_index,l_err,RMSE=opt(e,1,data)
    X=np.arange(0,10+10*min_index,1)
    Dat=[data['Potential(V)'][e+i] for i in np.arange(0,min(10+10*min_index,data.index[len(data)-1]+1-e),1)]

    if len(X)==len(Dat) :

        with warnings.catch_warnings():

            warnings.simplefilter("error", OptimizeWarning)
            warnings.simplefilter("error", RuntimeError)

            try:
                popt,pcov = scipy.optimize.curve_fit(lambda t,a,b,c: a*np.exp(b*t)+c,  X,  Dat, p0=(1,-2/75,1),bounds=((0,-1/20,-np.inf),(np.inf,-1/300,np.inf)),maxfev=800)
                #popt,pcov = scipy.optimize.curve_fit(lambda t,a,b,c: a*np.exp(b*t)+c,  X,  Dat, p0=(-1,1,1),maxfev=800)
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

    else :
        E=1

    if E==0 :
        RMSE=sqrt(mean_squared_error(Dat, func(X,popt)))
        return((a_er,b_er,c_er),min_index,popt)

    else :
        return((100,100,100),min_index,popt)

def test() :

    #RMSE=sqrt(mean_squared_error(Dat, func(X,popt)))
    #print(0.9*(a_er+100*b_er+c_er)/102+0.1*RMSE)
    plt.plot(X,func(X,popt))
    plt.plot(Dat)
    plt.show()
    plt.close()

