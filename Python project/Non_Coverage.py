def diff(l) : #This function provides the list of the differences between the values of the list l
    L=[]
    for i in range (1,len(l)) :
        L.append(l[i]-l[i-1])
    return(L)

def count(Diff,S=60) : #This function counts the number of values of Diff whose value is lower than S, a threshold value
    N=0
    for e in Diff :
        if e<=S :
            N+=1
    return(N)

def update(i_3,MAX,data,S=60) : #i_3" is a list of indexes
#if MAX=1 if MAX, -1 if MIN
#S is the thresold value from the precedent function

#This function filters the list of indexes i_3, to prevent the coverage of ZF ; among the group of indexes which are too close to each other, only the best are kept (the minimums or maximums, depending on the times of points specified by the variable MAX)
    i_4=copy.deepcopy(i_3)
    Diff=diff(i_4)
    To_del=[]

    while count(Diff,S)>0 :

        To_del=[]

        Diff=diff(i_4)
        for i in range (len(Diff)) :

            if Diff[i]<=S :

                Liste=[data['Potential(V)'][i_4[i+1]],data['Potential(V)'][i_4[i]]]

                if MAX==-1 :

                    k=Liste.index(max(Liste))
                    To_del.append(i_4[i+(1-k)])

                elif MAX==1 :

                    k=Liste.index(min(Liste))
                    To_del.append(i_4[i+(1-k)])

        for p in To_del :
            if p in i_4 :
                i_4.remove(p)

        Diff=diff(i_4)

    return(i_4)
