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

root_pth = "C:/Users/mikae/Desktop/helsa"

pth_dict = {}

for label_folder in os.listdir(root_pth):

    pth_dict[label_folder] = []

    for img_pth in os.listdir(os.path.join(root_pth, label_folder)):
        pth_dict[label_folder].append(os.path.join(root_pth, label_folder, img_pth))


mfa = mf.MFA()
mfa.wt_name = 'db3'
mfa.p = 2
mfa.j1 = 1
mfa.j2 = 12
mfa.q = np.arange(-4, 4)
mfa.n_cumul = 3
mfa.wtype = 0
mfa.verbose = 0
mfa._set_and_verify_parameters()

avg_feature = {}


def construct_feature_matrix(m):

    return np.concatenate([m.cumulants.log_cumulants, m.spectrum.Dq, m.spectrum.hq, m.structure.zeta])
    #return np.concatenate([mfa.spectrum.Dq, mfa.spectrum.hq])

for label in pth_dict:

    n_neg_hmin = 0
    
    n_imgs = len(pth_dict[label])
    avg_feature[label] = {}

    for img_pth in pth_dict[label]:

        mfa.wavelet_coeffs = None
        mfa.wavelet_leaders = None
        mfa.structure = None
        mfa.cumulants = None
        #ITU-R 601-2 luma transform
        img = np.array(Image.open(img_pth).convert("L"))
        img = img[0:128, 0:128]
        #mfa._wavelet_analysis(img)
        #mfa._estimate_hmin()

        mfa.j1 = 1
        mfa.j2 = 12

        mfa.analyze_no_estimation(img)

        oj1 = mfa.j1
        oj2eff = mfa.j2_eff

        for jj1 in range(oj1, oj2eff):
            avg_feature[label][jj1] = {}

            for jj2 in range(jj1+1, oj2eff+1):
                mfa.j1 = jj1
                mfa.j2_eff = jj2
                mfa.perform_estimations()
                
                if not jj2 in avg_feature[label][jj1]:
                     avg_feature[label][jj1][jj2] = 0

                avg_feature[label][jj1][jj2] += construct_feature_matrix(mfa)/n_imgs

dist = {}

labels = [key for key in avg_feature]

#Calculate pairwise distance between all labels, all scale ranges
for j1 in avg_feature[labels[0]]:
    dist[j1] = {}
    for j2 in avg_feature[labels[0]][j1]:
        for i in range(len(labels)-1):
            for j in range(i+1,len(labels)):
                print(j1, j2, labels[i], labels[j])
                if not j2 in dist[j1]:
                    dist[j1][j2] = 0
                #Sum of distance of all dimensions
                dist[j1][j2] += np.sum(np.abs(avg_feature[labels[i]][j1][j2] - avg_feature[labels[j]][j1][j2]))

print(dist)



       # mfa.j1 = 1
       # mfa.j2_eff = 3
       # mfa.perform_estimations()
#
       # cp  = mfa.cumulants.log_cumulants
       # spectrum_Dq = mfa.spectrum.Dq
       # spectrum_hq = mfa.spectrum.hq
       # structure      = mfa.structure
       # structure_func = structure.zeta
       # hmin = mfa.hmin
#
       # mfa.j1 = 2
       # mfa.j2_eff = 5
       # mfa.perform_estimations()
#
       # cp  = mfa.cumulants.log_cumulants
       # spectrum_Dq = mfa.spectrum.Dq
       # spectrum_hq = mfa.spectrum.hq
       # structure      = mfa.structure
       # structure_func = structure.zeta
       # hmin = mfa.hmin
#
       # if hmin < 0:
       #     n_neg_hmin+=1

    #print("# neg hmin:", n_neg_hmin, "/",n_imgs)
