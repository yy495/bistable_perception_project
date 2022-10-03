'''this is the main script to run simulation on Gershman 09's model'''

'''import functions'''
import torch
from sampling_gershman_09 import simulation
from post_proccess import generate_hist

'''the followings are parameters for simulation'''

'''p is the number of parallel sequences, no_samples are the number of samples in each sequence'''
p = 200
no_samples=22000

'''other parameters'''
'''no_bins: number of bins in histogram'''
'''cut_domi: the number of (longest) dominance period to be cutted'''
'''dim: dimension of simulation, set to 4'''
'''definition of a,b,sigma,lamda is in prob_gershman_09'''

no_bins = 1000
cut_domi = 20

'''set device variable here'''
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

s = simulation(p,no_samples,device,a = 1, b = -1, sigma = 0.3, lamda = 0.05,dim = 4)
s.to('cpu')
s.numpy()
hist,bins = generate_hist(s,p,no_samples,no_bins,cut_domi,dim=4,a=1,b=-1)

'''output hist and bins variable to a csv file'''
import numpy.random as random
import numpy as np
rand_ind = random.randint(0,10000)
file_name_hist = 'hist'+str(rand_ind)
file_name_bins = 'bins'+str(rand_ind)
np.savetxt(file_name_hist, hist, delimiter=",")
np.savetxt(file_name_bins, bins, delimiter=",")