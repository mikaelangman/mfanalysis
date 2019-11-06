"""
This script shows how to use the MFA class to compute the structure functions,
cumulants and log-cumulants.

NOTE:
    This example uses the same 1d data as the code demo_pleaders_basic.m written
    by Roberto Leonarduzzi for the PLBMF Matlab toolbox, found at:
    https://www.irit.fr/~Herwig.Wendt/software.html
"""

#-------------------------------------------------------------------------------
# Import mfanalysis package
#-------------------------------------------------------------------------------
import mfanalysis as mf

#-------------------------------------------------------------------------------
# Other imports
#-------------------------------------------------------------------------------
import os
import numpy as np
import scipy.ndimage
from scipy.io import loadmat
from PIL import Image
#-------------------------------------------------------------------------------
# Function to load data
#-------------------------------------------------------------------------------
def get_data_from_mat_file(filename):
    contents = loadmat(filename)
    return contents['data']

def get_data_from_image_file(filename):
    return np.array(Image.open(filename).convert("L"))

#-------------------------------------------------------------------------------
# Select example data
#-------------------------------------------------------------------------------
mf_process = 1; # 1 or 2

if mf_process == 1:
    # fractional Brownian motion (H=0.8, N=4096)
    data_file = 'examples\example_data\Lesion_Selection_7.png'
    #data_file = "example_data/cmcLN2d_00125_0025_n1024.jpg"
    #data_file = "examples/example_data/Normal_Selection_23.png"
    #data_file = "examples\example_data\cmcLN2d_00125_0025_n1024.mat"
    #data_file = "examples/example_data/1D1F3679-00A9-42E8-BAD3-DBDAE3CC1991.jpg"
elif mf_process == 2:
    # multifractal random walk (c_1=0.75, c_2=-0.05, N=32768)
    data_file = 'example_data/mrw07005n32768.mat'

# Complete path to file
current_dir = os.getcwd()
data_file = os.path.join(current_dir, data_file)

#-------------------------------------------------------------------------------
# Load data
#-------------------------------------------------------------------------------
#data = get_data_from_mat_file(data_file)
data = get_data_from_image_file(data_file)

data = data[0:256, 0:256]

#-------------------------------------------------------------------------------
# Setup analysis
#-------------------------------------------------------------------------------

# Multifractal analysis object
mfa = mf.MFA()

### Set parameters. They can also be set in the constructor of mf.MFA()

# wavelet to be used (see PyWavelets documentation for more options)
mfa.wt_name = 'db3'

# value of p for p-leaders, can be numpy.inf
# NOTE: instead of defining the value of p, we can set the variable mfa.formalism,
#       e.g., mfa.formalism = 'wlmf' (corresponding to p = np.inf) or
#             mfa.formalism = 'wcmf' (which uses wavelet coefficients only, not leaders)
mfa.p = 2


# scaling range
mfa.j1 = 2
mfa.j2 = 4

# range of orders q
mfa.q = np.arange(-4, 4)

# number of cumulants to be computed
mfa.n_cumul = 3

# fractional integration order
mfa.gamint = 0

# verbosity level (0: nothing,
#                  1: only text,
#                  2: plots)
mfa.verbose = 2

# regression type (0: ordinary least squares,
#                  1: weighted least squares, weights = nj)
mfa.wtype = 0

#-------------------------------------------------------------------------------
# Analyze data and get results
#-------------------------------------------------------------------------------

import time
import multiprocessing

#n = 10000

#start = time.time()

#for i in range(n):
#    mfa.analyze(data)


#end = time.time() - start
#print("TID:", end/n*1000)
# get cumulants
# See mfanalysis/cumulants.py for more attributes/methods

#mfa.analyze(data)
mfa._set_and_verify_parameters()

# Clear previously computed data
mfa.wavelet_coeffs = None
mfa.wavelet_leaders = None
mfa.structure = None
mfa.cumulants = None

# Compute wavelet coefficients and wavelet leaders

mfa.analyze(data)



cp  = mfa.cumulants.log_cumulants
#print("c1 = ", cp[0])
#print("c2 = ", cp[1])
#print("c3 = ", cp[2])


# wavelet coefficients and wavelet leaders
# - wt_coeffs[j] = wavelet coefficients at scale j
# - leaders[j]   = wavelet (p-) leaders at scale j
# See mfanalysis/multiresquantity.py for more attributes/methods
wt_coeffs = mfa.wavelet_coeffs.values
leaders   = mfa.wavelet_leaders.values

# structure function
# - structure_vals[ind_q, ind_j] = values of S(j, q),
#     with q = structure.q[ind_q] and j = structure.j[ind_j]
# See mfanalysis/structurefunction.py for more attributes/methods
structure      = mfa.structure
structure_vals = mfa.structure.values

spectrum_Dq = mfa.spectrum.Dq
spectrum_hq = mfa.spectrum.hq


print(spectrum_Dq)
print(spectrum_hq)
print(structure.zeta)
print(cp)
print(mfa.j2_eff)

# plot cumulants
#mfa.plot_cumulants(show = False)

# plot structure function and scaling function
#mfa.plot_structure(show = False)

# plot multifractal spectrum
#mfa.plot_spectrum(show = False)

# show plots 
#mfa.plt.show()