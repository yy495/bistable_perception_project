import torch
from probability_12 import p_w_new, p_pi_l_new, p_pi_r_new

def proposal_function_new(z, index, p, sigma = 1):
    #generate a new proposal from the current sample
    #change only one element at a time!
    z_star = z.clone()
    i,j = index[0],index[1]
    z_star[:,i,j] = z_star[:,i,j] + sigma*torch.randn(p)
    return z_star

def binary_proposal_new(z,index):
    #p: the probability of switch, to stablize the system
    z_star = z.clone()
    z_star[:,index[0],index[1]] = 1-z[:,index[0],index[1]]
    return z_star

def sample_rejection_w_new(index,w,w_star,x_l,x_r,pi_l,pi_r,beta,var,b,tau,p):
    p_acc = p_w_new(index,w_star,x_l,x_r,pi_l,pi_r,beta,var,b,tau,p)/p_w_new(index,w,x_l,x_r,pi_l,pi_r,beta,var,b,tau,p)
    p_ran = torch.rand(p)
    acc = torch.sign(p_acc-p_ran)
    return acc

def sample_rejection_pi_l_new(index,w,x_l,x_r,pi_l,pi_l_star,var,b,tau,gamma,p):
    p_acc = p_pi_l_new(index,w,x_l,x_r,pi_l_star,var,b,tau,gamma,p)/p_pi_l_new(index,w,x_l,x_r,pi_l,var,b,tau,gamma,p)
    p_ran = torch.rand(p)
    acc = torch.sign(p_acc-p_ran)
    return acc
    
def sample_rejection_pi_r_new(index,w,x_l,x_r,pi_r,pi_r_star,var,b,tau,gamma,p):
    p_acc = p_pi_r_new(index,w,x_l,x_r,pi_r_star,var,b,tau,gamma,p)/p_pi_r_new(index,w,x_l,x_r,pi_r,var,b,tau,gamma,p)
    p_ran = torch.rand(p)
    acc = torch.sign(p_acc-p_ran)
    return acc

def simulation_new(var,beta,gamma,alpha,tau,b,dim,p,no_sample,device):
    #p: no. parallel sampling sequences
    #no. sample: no. samples in each sequence
    #dim: dimension of each MRF

    #set initial condition for w, pi_r, pi_l
    #set visual input x_r, x_l as conflicting input
    x_l = torch.ones((p,dim,dim))
    x_l.to(device)
    x_r = torch.zeros((p,dim,dim))
    x_r.to(device)
    w = torch.rand(p,dim,dim)
    w.to(device)
    pi_l = torch.randint(0,2,(p,dim,dim))
    pi_l.to(device)
    pi_r = torch.randint(0,2,(p,dim,dim))
    pi_r.to(device)

    s = torch.zeros((no_sample,p,dim,dim))
    s.to(device)
    for m in range(int(no_sample)):
        #order of sweep is randomnized
        _,sampling_order_i = torch.sort(torch.rand(dim))
        _,sampling_order_j = torch.sort(torch.rand(dim))
        #the following for loop update on pi_l
        for i in sampling_order_i:
            for j in sampling_order_j:
                index = (i,j)
                pi_r_star = binary_proposal_new(pi_r, index)
                acc_1 = sample_rejection_pi_r_new(index,w,x_l,x_r,pi_r,pi_r_star,var,b,tau,gamma,p)
                for j in range(p):
                    if acc_1[j] == 1:
                        pi_r[j,:,:] = pi_r_star[j,:,:]

        #the following for loop update on pi_r
        for i in sampling_order_i:
            for j in sampling_order_j:
                index = (i,j)
                pi_l_star = binary_proposal_new(pi_l, index)
                acc_2 = sample_rejection_pi_l_new(index,w,x_l,x_r,pi_l,pi_l_star,var,b,tau,gamma,p)
                for j in range(p):
                    if acc_2[j] == 1:
                        pi_l[j,:,:] = pi_l_star[j,:,:]
    
            #the following for loop update on w
            for i in sampling_order_i:
                for j in sampling_order_j:
                    index = (i,j)
                    w_star = proposal_function_new(w, index, p)
                    acc_3 = sample_rejection_w_new(index,w,w_star,x_l,x_r,pi_l,pi_r,beta,var,b,tau,p)
                    for j in range(p):
                        if acc_3[j] == 1:
                            w[j,:,:] = w_star[j,:,:]
        
        s_m = w*x_l+(1-w)*x_r
        s_m_a = s_m.clone()
        s[m] = s_m_a
    return s