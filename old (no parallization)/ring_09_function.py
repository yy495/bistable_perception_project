'''ring topology for testing travelling waves from Gershman 2009 model'''
#default constant: a = 1 b = -1, sigma = 0.3 (width of mode), lamda = 0.8, sigma1 = 1.5 (proposal function), mu_array = np.array([0])

import numpy as np
import numpy.random as random
import scipy.signal as sig
def likelihood_ring(z,a,b,sigma):
    #bi-modal, mixture of Gaussian function
    #2 peaks at -1 and 1
    #return: a scalar, p(x|z) as a functino of z
    z_vec = z.reshape(-1)
    likelihood_ele = np.exp(-0.5/sigma**2*(z+a)**2)+np.exp(-0.5/sigma**2*(z+b)**2)
    return np.prod(likelihood_ele)

def MRF_prior_ring(z,lamda,mu_array):
    #return the Gaussian MRF of the perception
    #'reflection on boundary first
    dim = len(z)
    z_extended = np.hstack((z[-1],z,z[0]))
    sum_dis = 0
    for i in range(1,dim+1):
        sum_dis = sum_dis + (z_extended[i]-z_extended[i+1])**2 + (z_extended[i]-z_extended[i-1])**2
    return np.exp(-lamda*sum_dis)*np.prod(np.exp(mu_array))

def proposal_function_ring(z,sigma1 = 1.5):
    #generate a new proposal from the current sample
    #change only one element at a time!
    dim = z.shape[0]
    z_star = z.copy()
    i = random.randint(0,dim)
    z_star[i] = z_star[i] + sigma1*np.random.normal()
    return z_star

def sample_rejection_ring(z,z_star,a = 1, b = -1, sigma = 0.3, lamda = 0.80, sigma1 = 1.5 ,mu_array=np.array([0])):
    #input: z is the current sample
    #z_star is the proposed sample
    #if sample accepted: return True
    #if rejected: return False
    p_z = likelihood_ring(z,a,b,sigma)*MRF_prior_ring(z,lamda,mu_array)
    p_z_star = likelihood_ring(z_star,a,b,sigma)*MRF_prior_ring(z_star,lamda,mu_array)
    p_acc = min(1,p_z_star/p_z)
    if random.random()<p_acc:
        return True
    else:
        return False
    
def state_count(z):
    #cound the number of element in z that is larger than 0
    return len(np.where(z>0)[0])

def corr(x,y,n):
    #x, y: arrays to calculate correlation
    #n: cover correlation rxy(0) to rxy(n-1)
    conv_xy=sig.convolve(np.flip(x),y)
    rxy=np.zeros(int(n))
    for i in range(n):
        rxyi=conv_xy[len(x)+i-1]
        #for j in range(min(len(y)-n,len(x))):
            #rxyi=rxyi+x[j]*y[j+i]
        rxyi=rxyi/min(len(y)-n,len(x))
        rxy[i]=rxyi
    return rxy
