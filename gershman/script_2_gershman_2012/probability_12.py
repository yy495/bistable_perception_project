import torch

def p_w_new(index,w,x_l,x_r,pi_l,pi_r,beta,var,b,tau,p):
    #index: the index of the weight to be sampled, tuple (i,j)
    i,j = index[0],index[1]
    s_n = w[:,i,j]*x_l[:,i,j]+(1-w[:,i,j])*x_r[:,i,j]
    E_s = (b[:,i,j]-s_n)**2
    w_extended = torch.cat((torch.reshape(w[:,:,0],(p,-1,1)),w,torch.reshape(w[:,:,-1],(p,-1,1))),axis=2)
    w_extended = torch.cat((torch.reshape(w_extended[:,0,:],(p,1,-1)),w_extended,torch.reshape(w_extended[:,-1,:],(p,1,-1))),axis=1)
    E_s = E_s + beta*((w_extended[:,i+1,j+1]-w_extended[:,i,j+1])**2+(w_extended[:,i+1,j+1]-w_extended[:,i+1,j])**2)
    E_s = E_s + beta*((w_extended[:,i+1,j+1]-w_extended[:,i+2,j+1])**2+(w_extended[:,i+1,j+1]-w_extended[:,i+1,j+2])**2)
    E_x = 1/(2*var)*(pi_l[:,i,j]*(x_l[:,i,j]-s_n)**2+pi_r[:,i,j]*(x_r[:,i,j]-s_n)**2)
    return torch.exp(-tau*(E_x+E_s))

def p_pi_l_new(index,w,x_l,x_r,pi_l,var,b,tau,gamma,p):
    #sparse coefficient alpha TBC
    i,j = index[0],index[1]
    s_n = w[:,i,j]*x_l[:,i,j]+(1-w[:,i,j])*x_r[:,i,j]
    E_xl = 1/(2*var)*pi_l[:,i,j]*(x_l[:,i,j]-s_n)**2
    pi_extended = torch.cat((torch.reshape(pi_l[:,:,0],(p,-1,1)),pi_l,torch.reshape(pi_l[:,:,-1],(p,-1,1))),axis=2)
    pi_extended = torch.cat((torch.reshape(pi_extended[:,0,:],(p,1,-1)),pi_extended,torch.reshape(pi_extended[:,-1,:],(p,1,-1))),axis=1)
    #E_pi_l = alpha*(pi_extended[i+2,j+1]+pi_extended[i+1,j+2]+pi_extended[i,j+1]+pi_extended[i+1,j])
    '''E_pi_l = E_pi_l + gamma*((pi_extended[:,i+1,j+1]-pi_extended[:,i,j+1])**2+(pi_extended[:i+1,j+1]-pi_extended[:,i+1,j])**2)'''
    E_pi_l = gamma*((pi_extended[:,i+1,j+1]-pi_extended[:,i,j+1])**2+(pi_extended[:,i+1,j+1]-pi_extended[:,i+1,j])**2)
    E_pi_l = E_pi_l + gamma*((pi_extended[:,i+1,j+1]-pi_extended[:,i+2,j+1])**2+(pi_extended[:,i+1,j+1]-pi_extended[:,i+1,j+2])**2)
    return torch.exp(-tau*(E_xl+E_pi_l))

def p_pi_r_new(index,w,x_l,x_r,pi_r,var,b,tau,gamma,p):
    #sparse coefficient alpha TBC
    i,j = index[0],index[1]
    s_n = w[:,i,j]*x_l[:,i,j]+(1-w[:,i,j])*x_r[:,i,j]
    E_xr = 1/(2*var)*pi_r[:,i,j]*(x_r[:,i,j]-s_n)**2
    pi_extended = torch.cat((torch.reshape(pi_r[:,:,0],(p,-1,1)),pi_r,torch.reshape(pi_r[:,:,-1],(p,-1,1))),axis=2)
    pi_extended = torch.cat((torch.reshape(pi_extended[:,0,:],(p,1,-1)),pi_extended,torch.reshape(pi_extended[:,-1,:],(p,1,-1))),axis=1)
    #E_pi_l = alpha*(pi_extended[i+2,j+1]+pi_extended[i+1,j+2]+pi_extended[i,j+1]+pi_extended[i+1,j])
    '''E_pi_l = E_pi_l + gamma*((pi_extended[:,i+1,j+1]-pi_extended[:,i,j+1])**2+(pi_extended[:i+1,j+1]-pi_extended[:,i+1,j])**2)'''
    E_pi_r = gamma*((pi_extended[:,i+1,j+1]-pi_extended[:,i,j+1])**2+(pi_extended[:,i+1,j+1]-pi_extended[:,i+1,j])**2)
    E_pi_r = E_pi_r + gamma*((pi_extended[:,i+1,j+1]-pi_extended[:,i+2,j+1])**2+(pi_extended[:,i+1,j+1]-pi_extended[:,i+1,j+2])**2)
    return torch.exp(-tau*(E_xr+E_pi_r))