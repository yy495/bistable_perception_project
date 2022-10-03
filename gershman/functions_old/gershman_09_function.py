import numpy as np
import numpy.random as random

'''functions for Gershman 2009 result recreation'''
#Gershman 2009 implementation:
#z: inferred samples, defined as a 4*4 array
#z_star: proposed sample, defined as a 4*4 array
def likelihood(z,a = 1,b = -1,sigma = 0.3):
    #bi-modal, mixture of Gaussian function
    #2 peaks at -1 and 1
    #return: a scalar, p(x|z) as a functino of z
    z_vec = z.reshape(-1)
    likelihood_ele = np.exp(-0.5/sigma**2*(z+a)**2)+np.exp(-0.5/sigma**2*(z+b)**2)
    return np.prod(likelihood_ele)

def MRF_prior(z,lamda = 0.1,mu_array = np.array([0])):
    #return the Gaussian MRF of the perception
    #'reflection on boundary first
    dim = z.shape[0]
    z_extended_h = np.hstack((z[:,0].reshape(-1,1),z,z[:,-1].reshape(-1,1)))
    z_extended = np.vstack((z_extended_h[0,:],z_extended_h,z_extended_h[-1,:]))
    sum_dis = 0
    for i in range(1,dim):
        for j in range(1,dim):
            sum_dis = sum_dis + (z_extended[i][j]-z_extended[i+1][j])**2 + (z_extended[i][j]-z_extended[i-1][j])**2
            sum_dis = sum_dis + (z_extended[i][j]-z_extended[i][j+1])**2 + (z_extended[i][j]-z_extended[i][j-1])**2
    return np.exp(-lamda*sum_dis)*np.prod(np.exp(mu_array))

def proposal_function(z, sigma = 1.5,dim=4):
    #generate a new proposal from the current sample
    #change only one element at a time!
    dim = z.shape[0]
    z_star = z.copy()
    i = random.randint(0,dim-1)
    j = random.randint(0,dim-1)
    z_star[i,j] = z_star[i,j] + sigma*np.random.normal()
    return z_star

def sample_rejection(z,z_star):
    #input: z is the current sample
    #z_star is the proposed sample
    #if sample accepted: return True
    #if rejected: return False
    p_z = likelihood(z)*MRF_prior(z)
    p_z_star = likelihood(z_star)*MRF_prior(z_star)
    p_acc = min(1,p_z_star/p_z)
    if random.random()<p_acc:
        return True
    else:
        return False
    
def state_count(z):
    #cound the number of element in z that is larger than 0
    return len(np.where(z>0)[0])

def domi_period_count(state_list_input,dim = 4):
    cross_list = []
    #element in cross list: (index,cross_type)
    #cross type = 1: upper bound going up
    #cross type = 2: lower bound going down
    for i in range(1,len(state_list_input)):
        if state_list_input[i-1]>dim**2/3 and state_list_input[i]<dim**2/3:
            cross_list.append((i,2))
        elif state_list_input[i-1]<2*dim**2/3 and state_list_input[i]>2*dim**2/3:
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
    return np.array(domi_list)





