#functions for differetialble histogram
import numpy as np
import torch

def diff_hist_p1(s,N,p,R,hist_min,hist_max):
    #R: number of bins
    #delta: bin width
    #s: sample set (p runs, N samples each run)
    #p: parallel runs
    #hist_min, hist_max: 
    f = torch.nn.ReLU()
    delta = (hist_max - hist_min)/(R-1)
    s = torch.reshape(s,(p,N,1))
    r = torch.linspace(1,R-2,R-2).repeat(p,N,1)
    T_r = hist_min + r*delta
    T_rm1 = hist_min + r*delta - delta
    T_rp1 = hist_min + r*delta + delta
    return f(torch.sum(f(s-T_rm1) - 2.0*f(s-T_r) + f(s - T_rp1), axis = 1)/(N*delta))

def kl_loss(s, target_hist):
    #s: samples from simulation
    #target_hist: target histogram, assume with same no. bins, bin no., and range of histogram
    sim_hist = diff_hist_p1(s,N1,p,R,hist_min,hist_max)

    #hist_cost: KL divergence of desired and simulated histogram
    
    hist_cost = torch.sum(torch.sum(target_hist*torch.log((target_hist + 1e-20)/(sim_hist + 1e-20)),axis = 1))
    hist_cost_2 = torch.sum(torch.sum(sim_hist*torch.log((sim_hist + 1e-20)/(target_hist + 1e-20)),axis = 1))

    #out_range_cost: penalize samples that exceed the range of the histogram binning
    out_range_cost = torch.mean(f(hist_min-s)**2 + f(s-hist_max)**2)
    return hist_cost + hist_cost_2 + 10*out_range_cost



