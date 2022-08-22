'''this script contains functions on HM sampling and main simulation loop'''
import numpy as np
import torch
from prob_gershman_09 import likelihood_new, MRF_prior_new

def proposal_function_new(z, p, sigma1 = 1.5):
    #generate a new proposal from the current sample
    #change only one element at a time!
    #sigma1: std of step size in proposal
    dim = z.shape[1]
    z_star = z.clone()
    index = torch.randint(0,dim,(p,2))
    for i in range(p):
        z_star[i,index[i,0],index[i,1]] = z_star[i,index[i,0],index[i,1]] + sigma1*torch.randn(1)
    return z_star

def sample_rejection_new(z,z_star,p,a = 1, b = -1, sigma = 0.3, lamda = 0.05):
    #input: z is the current sample
    #z_star is the proposed sample
    #if sample accepted: return True
    #if rejected: return False
    p_acc = likelihood_new(z_star,a,b,sigma,p)*MRF_prior_new(z_star,lamda,p)/(likelihood_new(z,a,b,sigma,p)*MRF_prior_new(z,lamda,p))
    p_ran = torch.rand(p)
    acc = torch.sign(p_acc-p_ran)
    return acc

def simulation(p,no_samples,device,a = 1, b = -1, sigma = 0.3, lamda = 0.05,dim = 4):
    state_list_par = torch.zeros(no_samples,p,dim,dim)
    state_list_par.to(device)
    #define initial z
    z = (torch.rand(p,dim,dim)-0.5)*2
    z.to(device)
    for i in range(int(no_samples)):
        z_star = proposal_function_new(z,p)
        z_star.to(device)
        acc = sample_rejection_new(z,z_star,p,a = 1, b = -1, sigma = 0.3, lamda = 0.05)
        for j in range(p):
            if acc[j] == 1:
                z[j,:,:] = z_star[j,:,:]
        z_a = z.clone()
        state_list_par[i] = z_a
    return state_list_par