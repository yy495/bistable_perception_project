'''this script contains functions to calculate various probability'''
import numpy as np
import torch

def likelihood_new(z,a,b,sigma,p):
    #z: percept
    #bi-modal, mixture of Gaussian function
    #2 peaks at a and b
    #sigma: equal variance of both modes
    #return: a vectos, p(x|z_i) as a functino of z_i for every sequences running in parallel
    likelihood_ele = torch.exp(-0.5/sigma**2*(torch.reshape(z,(p,-1))-a)**2) + torch.exp(-0.5/sigma**2*(torch.reshape(z,(p,-1))-b)**2)
    return torch.prod(likelihood_ele,axis=1)

def MRF_prior_new(z,lamda,p):
    #return the Gaussian MRF of the perception for each parallel sequence
    #reflection on boundary first
    #lamda: smoothness penalty term
    
    dim = z.shape[1]
    z_extended = torch.cat((torch.reshape(z[:,:,0],(p,-1,1)),z,torch.reshape(z[:,:,-1],(p,-1,1))),axis=2)
    z_extended = torch.cat((torch.reshape(z_extended[:,0,:],(p,1,-1)),z_extended,torch.reshape(z_extended[:,-1,:],(p,1,-1))),axis=1)
    sum_dis = torch.sum(torch.sum((z-z_extended[:,:dim,1:dim+1])**2,axis=1),axis=1)+torch.sum(torch.sum((z-z_extended[:,2:,1:dim+1])**2,axis=1),axis=1)
    sum_dis = sum_dis+torch.sum(torch.sum((z-z_extended[:,1:dim+1,:dim])**2,axis=1),axis=1)+torch.sum(torch.sum((z-z_extended[:,1:dim+1:,2:])**2,axis=1),axis=1)
    return torch.exp(-lamda*sum_dis)