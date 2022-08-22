from sampling_12 import simulation_new
from post_proccess import generate_hist
import torch
import numpy as np

'''var: noise variance for both eyes' evidence, equal'''
'''beta and gamma: coefficient on smoothness penalty of pi_l, pi_r and w'''
'''tau: a global coefficient on all penalty term'''
var,beta,gamma,alpha,tau,dim = 0.0001,10,10,0,1/100,5

'''change no parallel sequence and no. samples in each sequence here'''
p,no_sample = 400,22000
b = 0.5*torch.ones((p,dim,dim))

'''set device variable here'''
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


s = simulation_new(var,beta,gamma,alpha,tau,b,dim,p,no_sample,device)

no_bins = 1000
cut_domi = 0

s.to('cpu')
s.numpy()
hist,bins = generate_hist(s,p,no_sample,no_bins,cut_domi)

'''output hist and bins variable to a csv file'''
import numpy.random as random
import numpy as np
rand_ind = random.randint(0,10000)
file_name_hist = 'hist'+str(rand_ind)
file_name_bins = 'bins'+str(rand_ind)
np.savetxt(file_name_hist, hist, delimiter=",")
np.savetxt(file_name_bins, bins, delimiter=",")