def p_w_ring(index,w,x_l,x_r,pi_l,pi_r,beta,var,b,tau,gamma,alpha):
    #index: the index of the weight to be sampled, int
    i = int(index)
    s_n = w[i]*x_l[i]+(1-w[i])*x_r[i]
    E_s = (b[i]-s_n)**2
    w_extended = np.hstack((w[-1],w,w[0])) #ring topology
    E_s = E_s + beta*((w_extended[i+1]-w_extended[i])**2+(w_extended[i+1]-w_extended[i+2])**2)
    E_x = 1/(2*var)*(pi_l[i]*(x_l[i]-s_n)**2+pi_r[i]*(x_r[i]-s_n)**2)
    return np.exp(-tau*(E_x+E_s))

def p_pi_l_ring(index,w,x_l,x_r,pi_l,pi_r,beta,var,b,tau,gamma,alpha):
    i = int(index)
    s_n = w[i]*x_l[i]+(1-w[i])*x_r[i]
    E_xl = 1/(2*var)*(pi_l[i]*(x_l[i]-s_n)**2)
    pi_extended = np.hstack((pi_l[-1],pi_l,pi_l[0]))
    E_pi_l = alpha*(pi_extended[i+2]+pi_extended[i])
    E_pi_l = E_pi_l + gamma*((pi_extended[i+2]-pi_extended[i+1])**2+(pi_extended[i+1]-pi_extended[i])**2)
    return np.exp(-tau*(E_xl+E_pi_l))

def p_pi_r_ring(index,w,x_l,x_r,pi_l,pi_r,beta,var,b,tau,gamma,alpha):
    i = int(index)
    s_n = w[i]*x_l[i]+(1-w[i])*x_r[i]
    E_xr = 1/(2*var)*(pi_r[i]*(x_r[i]-s_n)**2)
    pi_extended = np.hstack((pi_r[-1],pi_r,pi_r[0]))
    E_pi_r = alpha*(pi_extended[i+2]+pi_extended[i])
    E_pi_r = E_pi_r + gamma*((pi_extended[i+2]-pi_extended[i+1])**2+(pi_extended[i+1]-pi_extended[i])**2)
    return np.exp(-tau*(E_xr+E_pi_r))

#perform MH algorithm on a single variable conditioned on all other variable
#special note: for pi, the sample could only be 0 or 1!
def proposal_function_ring(z, index, sigma = 1):
    #generate a new proposal from the current sample
    #change only one element at a time!
    z_star = z.copy()
    i = int(index)
    z_star[i] = z_star[i] + sigma*random.normal()
    return z_star

def binary_proposal_ring(z,index,p):
    #p: the probability of switch, to stablize the system
    z_star = z.copy()
    if random.rand()<p:
        z_star[int(index)] = 1-z[int(index)]
    return z_star

def sample_rejection_w_ring(index,w,w_star,x_l,x_r,pi_l,pi_r,beta,var,b,tau,gamma,alpha):
    p_w_cur = p_w_ring(index,w,x_l,x_r,pi_l,pi_r,beta,var,b,tau,gamma,alpha)
    p_w_star = p_w_ring(index,w_star,x_l,x_r,pi_l,pi_r,beta,var,b,tau,gamma,alpha)
    p_acc = min(1,p_w_star/p_w_cur)
    if random.random()<p_acc:
        return True
    else:
        return False
    
def sample_rejection_pi_l_ring(index,w,x_l,x_r,pi_l,pi_l_star,pi_r,beta,var,b,tau,gamma,alpha):
    p_pi_l_cur = p_pi_l_ring(index,w,x_l,x_r,pi_l,pi_r,beta,var,b,tau,gamma,alpha)
    p_pi_l_star = p_pi_l_ring(index,w,x_l,x_r,pi_l_star,pi_r,beta,var,b,tau,gamma,alpha)
    p_acc = min(1,p_pi_l_star/p_pi_l_cur)
    if random.random()<p_acc:
        return True
    else:
        return False
    
def sample_rejection_pi_r_ring(index,w,x_l,x_r,pi_l,pi_r,pi_r_star,beta,var,b,tau,gamma,alpha):
    p_pi_r_cur = p_pi_r_ring(index,w,x_l,x_r,pi_l,pi_r,beta,var,b,tau,gamma,alpha)
    p_pi_r_star = p_pi_r_ring(index,w,x_l,x_r,pi_l,pi_r_star,beta,var,b,tau,gamma,alpha)
    p_acc = min(1,p_pi_r_star/p_pi_r_cur)
    if random.random()<p_acc:
        return True
    else:
        return False

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
