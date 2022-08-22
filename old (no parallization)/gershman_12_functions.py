'''functions to repeat Gershman 2012 result'''
import numpy as np
import numpy.random as random

def p_w(index,w,x_l,x_r,pi_l,pi_r,beta,var,b,tau,gamma,alpha):
    #index: the index of the weight to be sampled, tuple (i,j)
    i,j = index[0],index[1]
    s_n = w[i][j]*x_l[i][j]+(1-w[i][j])*x_r[i][j]
    E_s = (b[i][j]-s_n)**2
    w_extended_h = np.hstack((w[:,0].reshape(-1,1),w,w[:,-1].reshape(-1,1)))
    w_extended = np.vstack((w_extended_h[0,:],w_extended_h,w_extended_h[-1,:]))
    E_s = E_s + beta*((w_extended[i+1][j+1]-w_extended[i][j+1])**2+(w_extended[i+1][j+1]-w_extended[i+1][j])**2)
    E_s = E_s + beta*((w_extended[i+1][j+1]-w_extended[i+2][j+1])**2+(w_extended[i+1][j+1]-w_extended[i+1][j+2])**2)
    E_x = 1/(2*var)*(pi_l[i][j]*(x_l[i][j]-s_n)**2+pi_r[i][j]*(x_r[i][j]-s_n)**2)
    return np.exp(-tau*(E_x+E_s))

def p_pi_l(index,w,x_l,x_r,pi_l,pi_r,beta,var,b,tau,gamma,alpha):
    i,j = index[0],index[1]
    s_n = w[i][j]*x_l[i][j]+(1-w[i][j])*x_r[i][j]
    E_xl = 1/(2*var)*(pi_l[i][j]*(x_l[i][j]-s_n)**2)
    pi_extended_h = np.hstack((pi_l[:,0].reshape(-1,1),pi_l,pi_l[:,-1].reshape(-1,1)))
    pi_extended = np.vstack((pi_extended_h[0,:],pi_extended_h,pi_extended_h[-1,:]))
    E_pi_l = alpha*(pi_extended[i+2,j+1]+pi_extended[i+1,j+2]+pi_extended[i,j+1]+pi_extended[i+1,j])
    E_pi_l = E_pi_l + gamma*((pi_extended[i+1][j+1]-pi_extended[i][j+1])**2+(pi_extended[i+1][j+1]-pi_extended[i+1][j])**2)
    E_pi_l = E_pi_l + gamma*((pi_extended[i+1][j+1]-pi_extended[i+2][j+1])**2+(pi_extended[i+1][j+1]-pi_extended[i+1][j+2])**2)
    return np.exp(-tau*(E_xl+E_pi_l))

def p_pi_r(index,w,x_l,x_r,pi_l,pi_r,beta,var,b,tau,gamma,alpha):
    i,j = index[0],index[1]
    s_n = w[i][j]*x_l[i][j]+(1-w[i][j])*x_r[i][j]
    E_xr = 1/(2*var)*(pi_r[i][j]*(x_r[i][j]-s_n)**2)
    pi_extended_h = np.hstack((pi_r[:,0].reshape(-1,1),pi_r,pi_r[:,-1].reshape(-1,1)))
    pi_extended = np.vstack((pi_extended_h[0,:],pi_extended_h,pi_extended_h[-1,:]))
    E_pi_r = alpha*(pi_extended[i+2,j+1]+pi_extended[i+1,j+2]+pi_extended[i,j+1]+pi_extended[i+1,j])
    E_pi_r = E_pi_r + gamma*((pi_extended[i+1][j+1]-pi_extended[i][j+1])**2+(pi_extended[i+1][j+1]-pi_extended[i+1][j])**2)
    E_pi_r = E_pi_r + gamma*((pi_extended[i+1][j+1]-pi_extended[i+2][j+1])**2+(pi_extended[i+1][j+1]-pi_extended[i+1][j+2])**2)
    return np.exp(-tau*(E_xr+E_pi_r))

#perform MH algorithm on a single variable conditioned on all other variable
#special note: for pi, the sample could only be 0 or 1!
def proposal_function(z, index, sigma = 1):
    #generate a new proposal from the current sample
    #change only one element at a time!
    z_star = z.copy()
    i,j = index[0],index[1]
    z_star[i,j] = z_star[i,j] + sigma*random.normal()
    return z_star

def binary_proposal(z,index,p):
    #p: the probability of switch, to stablize the system
    z_star = z.copy()
    if random.rand()<p:
        z_star[index[0],index[1]] = 1-z[index[0],index[1]]
    return z_star

def sample_rejection_w(index,w,w_star,x_l,x_r,pi_l,pi_r,beta,var,b,tau,gamma,alpha):
    p_w_cur = p_w(index,w,x_l,x_r,pi_l,pi_r,beta,var,b,tau,gamma,alpha)
    p_w_star = p_w(index,w_star,x_l,x_r,pi_l,pi_r,beta,var,b,tau,gamma,alpha)
    p_acc = min(1,p_w_star/p_w_cur)
    if random.random()<p_acc:
        return True
    else:
        return False
    
def sample_rejection_pi_l(index,w,x_l,x_r,pi_l,pi_l_star,pi_r,beta,var,b,tau,gamma,alpha):
    p_pi_l_cur = p_pi_l(index,w,x_l,x_r,pi_l,pi_r,beta,var,b,tau,gamma,alpha)
    p_pi_l_star = p_pi_l(index,w,x_l,x_r,pi_l_star,pi_r,beta,var,b,tau,gamma,alpha)
    p_acc = min(1,p_pi_l_star/p_pi_l_cur)
    if random.random()<p_acc:
        return True
    else:
        return False
    
def sample_rejection_pi_r(index,w,x_l,x_r,pi_l,pi_r,pi_r_star,beta,var,b,tau,gamma,alpha):
    p_pi_r_cur = p_pi_r(index,w,x_l,x_r,pi_l,pi_r,beta,var,b,tau,gamma,alpha)
    p_pi_r_star = p_pi_r(index,w,x_l,x_r,pi_l,pi_r_star,beta,var,b,tau,gamma,alpha)
    p_acc = min(1,p_pi_r_star/p_pi_r_cur)
    if random.random()<p_acc:
        return True
    else:
        return False
