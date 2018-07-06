"""
This script tests the impact of additive gaussian noise on the estimation
of multifractal properties.

Precisely, let X_t be a multifractal random process; and let N_t be a 
white gaussian noise. We analyze here the signal Y_t = X_t + sigma*N_t
aiming to see for which values of sigma the log-cumulants of Y_t are 
sufficiently close to the log-cumulants of X_t.
"""

import mfanalysis as mf
import os
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

#-------------------------------------------------------------------------------
# Function to load data
#-------------------------------------------------------------------------------
def get_data_from_mat_file(filename):
    contents = loadmat(filename)
    return contents['data'][0]


#-------------------------------------------------------------------------------
# Load and normalize data, generate noise
#-------------------------------------------------------------------------------
# multifractal random walk (c_1=0.75, c_2=-0.05, N=32768)
data_file = 'example_data/mrw07005n32768.mat'
current_dir = os.path.dirname(os.path.abspath(__file__))
data_file = os.path.join(current_dir, data_file)
data = get_data_from_mat_file(data_file)

# normalize signal
data = data / data.std()

# generate noise
noise =np.random.normal(loc = 0, scale = 1.0, size=len(data)) #  np.sin(np.arange(len(data))) 

# vector of sigma^2 (variances)
sigma2 = np.linspace(0., 0.2, 1000) # np.array([0., 0.000001]) 

#-------------------------------------------------------------------------------
# MFA parameters
#-------------------------------------------------------------------------------
# Multifractal analysis object
mfa = mf.MFA()
mfa.wt_name = 'db3'
mfa.p = np.inf
mfa.j1 = 8
mfa.j2 = 12
mfa.n_cumul = 3
mfa.gamint = 0.0  # !!!!!!!!!!!!!!!!!!!!!!!!
mfa.verbose = 1
mfa.wtype = 0

# # get cumulants
# mfa.analyze(data)
# cp  = mfa.cumulants.log_cumulants
# print("c1 = ", cp[0])
# print("c2 = ", cp[1])
# print("c3 = ", cp[2])
#-------------------------------------------------------------------------------
# Analyze data
#-------------------------------------------------------------------------------

c1_list = []
c2_list = []

for s in sigma2:
	signal = data + np.sqrt(s)*noise	
	mfa.analyze(signal)
	cp  = mfa.cumulants.log_cumulants
	c1_list.append(cp[0])
	c2_list.append(cp[1])

# Plot c1
plt.figure()
plt.plot(sigma2, c1_list, 'bo')
plt.grid()

# Plot c2
plt.figure()
plt.plot(sigma2, c2_list, 'ro')
plt.grid()


# Plot data and noisy data
plt.figure()
plt.plot(data, 'b')
plt.plot(data + np.sqrt(s)*noise, 'r')
plt.grid()

plt.show()

