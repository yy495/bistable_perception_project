'''this script contains the code to generate histogram vectors from state of nodes'''
import numpy as np

def domi_period_count(state_list_input,dim):
    cross_list = []
    #element in cross list: (index,cross_type)
    #cross type = 1: upper bound going up
    #cross type = 2: lower bound going down
    for i in range(1,len(state_list_input)):
        if state_list_input[i-1]>=0.2*dim**2 and state_list_input[i]<0.2*dim**2:
            cross_list.append((i,2))
        elif state_list_input[i-1]<=0.8*dim**2 and state_list_input[i]>0.8*dim**2:
            cross_list.append((i,1))
    switch_list = [cross_list[0]]
    cross = cross_list[0][1]
    for i in range(1,len(cross_list)):
        if cross_list[i][1] != cross:
            cross = cross_list[i][1]
            switch_list.append(cross_list[i])
    domi_list =[]
    for i in range(1,len(switch_list)):
        domi_list.append(switch_list[i][0]-switch_list[i-1][0])
    return domi_list

def generate_hist(s,p,no_samples,no_bins,cut_domi,dim=5,a=0,b=1):
    #no_bins: number of bins in histogram
    #s: simulated array containing state of node
    state_no_array = []
    limit=(a+b)/2

    #state of node to state of the whole RMF (no. node > 0)
    for i in range(p):
        state_no_array_i = []
        for j in range(no_samples):
            state_no_array_i.append(len(np.where(s[j,i,:,:]>limit)[0]))
        state_no_array.append(state_no_array_i)
    state_no_array = np.array(state_no_array)

    #state of node to dominance duration
    domi_list = []
    for state_list_i in state_no_array:
        domi_list = domi_list + domi_period_count(state_list_i,dim)
    domi_list = np.array(domi_list)

    #note: according to previous experience, it might be necessary to cut the tail of distribution
    domi_list = np.array(domi_list)
    domi_list = np.sort(domi_list)[:-cut_domi-1]

    return np.histogram(domi_list,no_bins)
