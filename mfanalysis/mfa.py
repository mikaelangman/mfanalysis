from __future__ import print_function
from __future__ import unicode_literals

"""
The code in this package is based on the Wavelet p-Leader and Bootstrap based MultiFractal analysis (PLBMF) Matlab toolbox
written by Herwig Wendt (https://www.irit.fr/~Herwig.Wendt/software.html) and on the documents provided in his
website (his PhD thesis in particular, which can be found at https://www.irit.fr/~Herwig.Wendt/data/ThesisWendt.pdf).
"""

"""
# NOTES:

* Computation of wavelet coefficients:
    - I had to multiply the PyWavelet high-pass filter by -1, that is -1*np.array(wt.dec_hi), in order to obtain the same
    wavelet coefficients as the matlab toolbox. Why? At the end, this should not change the result, since we take the absolute
    values of the coefficients.
    - Why is the index of the first good value equal to filter_len-2 instead of filter_len-1? If filter_len == 2, there is
    still one corrupted value at the beginning.

* Possible bug reports for MatLab toolbox:
    - when nwt is changed, we need to do a clear all, otherwise the filter "h" may not have the correct size/values
    - signal of high-pass filter
    - number of wavelet leaders at j=1 is not correct the in mf_obj.mrq.leader.nj
"""

import warnings
import pywt
import numpy as np
import matplotlib
matplotlib.use("WXAgg")
import matplotlib.pyplot as plt

from scipy.signal import convolve
from .cumulants import *
from .structurefunction import *
from .multiresquantity import *
from .mfspectrum import *
from .utils import conv2d

from .bivariate_structurefunction import *
from .bivariate_cumulants import *

class MFA:
    """
    Class used for multifractal analysis.

    Args:
        formalism(str)            : 'wcmf'      - wavelet coefficient multifractal formalism
                                    'wlmf'      - wavelet leader multifractal formalism
                                    'p-leader'  - p-leader multifractal formalism

        p (float)                 :  p-leaders exponent, should be a positive value or numpy.inf
                                     Important: if p is not None, the parameter 'formalism' is ignored
                                     and the formalism is set according to the value of p:
                                     - for p > 0,      formalism = 'p-leader'
                                     - for p = np.inf, formalism = 'wlmf'

        wt_name                   : wavelet name (according to the ones available in
                                    pywavelets, see https://pywavelets.readthedocs.io/en/latest/regression/wavelet.html)
                                    For instance:
                                    'db2' - Daubechies with 2 vanishing moments

        j1 (int)                  : smallest scale analysis

        j2 (int)                  : largest scale analysis

        j2_eff(int)               : "effective" value of j2.
                                    it is the minimum between j2 and the maximum
                                    possible scale for wavelet decomposition, which
                                    depends on the length of the data.

        max_level(int)            : maximum scale of wavelet decomposition

        wtype (int)               : 0 for ordinary regression, 1 for weighted regression

        gamint(float)             : fractional integration parameter


        q  (numpy.array)          : numpy array containing the exponents for which
                                    to compute the structure functions

        n_cumul(int)              : number of cumulants to compute

        normalization(int)        : type of wavelet coefficient normalization  (1 for L1, 2 for L2 etc)

        weighted (bool)           : True for weighted linear regression, False for ordinary
                                    linear regression

        verbose(int)              : verbosity level

        hi (numpy.array)          : wavelet high-pass filter

        lo (numpy.array)          : wavelet low-pass filter

        filter_len(int)           : length of the filters

        plt                       : pointer to matplotlib.pyplot for convenience

        wavelet_coeffs (MultiResolutionQuantity)  : stores wavelet coefficients

        wavelet_leaders (MultiResolutionQuantity) : stores wavelet leaders

        structure(StructureFunction)              : structure function (depends on the chosen formalism)

        cumulants(Cumulants)                      : cumulants C_p(j)

        spectrum (MultifractalSpectrum)           : multifractal spectrum object

        hmin                       : uniform regularity exponent

        eta_p (float)              : value of the wavelet scaling function at exponent p,
                                      used to correct the values of the cumulants when p-leaders are used

        hurst_structure (numpy.array): structure function log2(S(j, 2)) computed in function compute_hurst()
    """
    def __init__(self,
                 formalism = 'wlmf',
                 p = None,
                 wt_name = 'db3',
                 j1 = 1,
                 j2 = 10,
                 gamint = 0.0,
                 q = [2],
                 n_cumul = 3,
                 verbose = 1,
                 normalization = 1,
                 wtype   = 1
                 ):
        self.formalism  = formalism
        self.wt_name    = wt_name
        self.j1        = int(j1)
        self.j2        = int(j2)
        self.wtype     = wtype
        self.gamint    = gamint
        self.verbose   = verbose
        self.q  = q
        self.normalization = normalization
        self.n_cumul   = n_cumul
        self.max_level = 0
        self.wavelet_coeffs = None
        self.wavelet_leaders = None
        self.structure = None
        self.cumulants = None
        self.spectrum  = None
        self.hmin      = None
        self.eta_p     = None
        self.p = p

        # Set parameters (verify formalism, initialize wavelet filters)
        # and verify if everything is ok
        self._set_and_verify_parameters()

        # Keep pointer to matplotlib.pyplot for convenience
        self.plt   = plt

        # Constants
        self.STRUCTURE_FIG_LABEL  = 'Structure Functions'
        self.SCALING_FIG_LABEL    = 'Scaling Function'
        self.CUMUL_FIG_LABEL      = 'Cumulants'
        self.HMIN_FIG_LABEL       = 'hmin'

        # Other
        self.utils = Utils() # used for linear regression


    def _set_and_verify_parameters(self):
        """
        Called at __init__() and before each analysis, to set the parameters or
        adapt changed parameters.
        """

        # Check formalism
        self._check_formalism()

        # Initialize wavelet filters
        self._initialize_wavelet_filters()

        # Verify j1
        assert self.j1 >= 1, "j1 should be equal or greater than 1."


    def _initialize_wavelet_filters(self):
        """
        Initialize filters and their parameters
        """
        wt =  pywt.Wavelet(self.wt_name)
        self.wt = wt
        self.hi =  -1*np.array(wt.dec_hi)
        self.lo =  np.array(wt.dec_lo)
        self.filter_len = len(self.hi)

    def _check_formalism(self):
        """
        Check formalism according to the value of p
        """
        p = self.p
        if p is not None:
            assert p > 0
            if np.isinf(p):
                self.formalism = 'wlmf'
            else:
                self.formalism = 'p-leader'

    

    def _wavelet_analysis(self, signal):
        """
        Compute all the wavelet coefficients from scales 1 to self.j2

        Wavelet coefficients are usually L^1 normalized, see page 5 of the document:
            http://perso-math.univ-mlv.fr/users/jaffard.stephane/pdf/Mandelbrot.pdf

        Explanation:
            When computing the wavelet coefficients, the values corrupted
            by border effects are set to infinity (np.inf).

            This makes it easier to compute the wavelet leaders, since corrupted values
            will also be infinite and can be removed.
        """

        # Initialize structures
        self.wavelet_coeffs = MultiResolutionQuantity(self.formalism)
        self.wavelet_coeffs_xpos = MultiResolutionQuantity("coef x")
        self.wavelet_coeffs_ypos = MultiResolutionQuantity("coef y")

        self.wavelet_leaders = MultiResolutionQuantity(self.formalism)
        self.wavelet_leaders_xpos = MultiResolutionQuantity("leader x")
        self.wavelet_leaders_ypos = MultiResolutionQuantity("leader y")


        # Check maximum decomposition level
        data_len = np.min(signal.shape)
    
        max_level = int(np.floor( np.log2( data_len / (self.filter_len+1) ) ))
        self.max_level = min( int(np.floor(np.log2(data_len))), max_level)
        #self.max_level = min(self.max_level, self.j2)


        # Warning if j2 is greater than max_level
        if self.j2 > self.max_level and self.verbose > 0:
            print(" ")
            print("(!) Warning: Value of j2 is higher than the maximum allowed level. Max level and j2 set to ", self.max_level)
            print(" ")
            

        # Check whether to compute wavelet leaders
        if self.formalism == 'wlmf' or self.formalism == 'p-leader':
            compute_leaders = True
        else:
            compute_leaders = False


        # Wavelet decomposition
        approx = signal
        sidata = signal.shape

        for j in range(1, self.max_level+1):

            #2D signal
            if len(approx.shape) == 2:                
               
                nj_temp = np.array(approx.shape)

                #border effect
                fp = self.filter_len - 1     # index of first good value
                lp = nj_temp - 1 # index of last good value

                # offsets
                x0 = 2
                x0Appro = self.filter_len #2*self.nb_vanishing_moments
                # apply filters
                # note: 'direct' method MUST be used, since there are elements
                # that are np.inf inside approx

               # %-- OH convolution and subsampling
               # OH = conv2(LL, gg1); OH(isnan (OH)) = Inf;
               # OH(:, 1         : fp - 1) = Inf;
               # OH(:, lp(2) + 1 : end   ) = Inf;
               # OH = OH(:, (1 : 2 : njtemp(2)) + x0 - 1);

                OH = conv2d(approx, self.hi, mode = 'full', method='direct')
                OH[:, :fp] = np.inf
                OH[:, lp[1]+1:] = np.inf
                OH_idx = np.arange(0, nj_temp[1], 2) + x0 - 1
                OH = OH[:, OH_idx]
                OH[np.isnan(OH)] = np.inf

               # %-- HH convolution and subsampling
               # HH = conv2(OH, gg1'); HH(isnan (HH)) = Inf;
               #
               # HH(1         : fp - 1 , :) = Inf;
               # HH(lp(1) + 1 : end    , :) = Inf;
               # HH = HH((1 : 2 : njtemp(1)) + x0 - 1, :);

                HH = conv2d(OH, self.hi, mode = 'full', method='direct', axis=0)
                HH[:fp, :] = np.inf
                HH[lp[0]+1:, :] = np.inf
                HH_idx = np.arange(0, nj_temp[0], 2) + x0 - 1
                HH = HH[HH_idx, :] 
                HH[np.isnan(HH)] = np.inf

               # %-- LH convolution and subsampling
               # LH = conv2 (OH, hh1'); LH(isnan (LH)) = Inf;
               # LH(1         : fp - 1, :) = Inf;
               # LH(lp(1) + 1 : end   , :) = Inf;
               # LH = LH((1 : 2 : njtemp(1)) + x0Appro - 1, :);

                LH = conv2d(OH, self.lo, mode = 'full', method='direct', axis=0)
                LH[:fp, :] = np.inf
                LH[lp[0]+1:, :] = np.inf
                LH_idx = np.arange(0, nj_temp[0], 2) + x0Appro - 1
                LH = LH[LH_idx, :] 
                LH[np.isnan(LH)] = np.inf

                #%-- OL convolution and subsampling
            #   OL = conv2 (LL, hh1); OL(isnan (OL)) = Inf;
            #   OL(:, 1         : fp - 1) = Inf;
            #   OL(:, lp(2) + 1 : end   ) = Inf;
            #   OL = OL(:, (1 : 2 : njtemp(2)) + x0Appro - 1);

                OL = conv2d(approx, self.lo, mode = 'full', method='direct')
                OL[:, 0:fp] = np.inf
                OL[:, lp[1]+1:] = np.inf
                OL_idx = np.arange(0, nj_temp[1], 2) + x0Appro - 1
                OL = OL[:, OL_idx]
                OL[np.isnan(OL)] = np.inf
                #%-- HL convolution and subsampling
                #HL = conv2 (OL, gg1'); HL(isnan (HL)) = Inf;
                #HL(1         : fp - 1, :) = Inf;
                #HL(lp(1) + 1 : end   , :) = Inf;
                #HL=HL((1 : 2 : njtemp(1)) + x0 - 1, :);

                HL = conv2d(OL, self.hi, mode = 'full', method='direct',  axis=0)
                HL[:fp, :] = np.inf
                HL[lp[0]+1:, :] = np.inf
                HL_idx = np.arange(0, nj_temp[0], 2) + x0 - 1
                HL = HL[HL_idx, :] 
                HL[np.isnan(HL)] = np.inf
                #%-- LL convolution and subsampling
                #LL = conv2 (OL, hh1'); LL(isnan (LL)) = Inf;
                #LL(1         : fp - 1 , :) = Inf;
                #LL(lp(1) + 1 : end    , :) = Inf;
                #LL = LL((1 : 2 : njtemp(1)) + x0Appro - 1, :);

                LL = conv2d(OL, self.lo, mode = 'full', method='direct', axis=0)
                LL[0:fp, :] = np.inf
                LL[lp[0]+1:, :] = np.inf
                LL_idx = np.arange(0, nj_temp[0], 2) + x0Appro - 1
                LL = LL[LL_idx, :]
                LL[np.isnan(LL)] = np.inf

                approx = LL
                details = [LH, HL, HH]

                # normalization 
                #details = [detail*2**(j*(0.5-1/self.normalization)) for detail in details]
                details = [detail/(2**(j/self.normalization)) for detail in details]

                # fractional integration
                details = [detail*2.0**(self.gamint*j) for detail in details]

                #-- get position of leader
                lesx = np.arange(0, sidata[0] + 1, 2**j)
                lesy = np.arange(0, sidata[1] + 1, 2**j)

                # remove infinite values and store wavelet coefficients

                startsx = []
                startsy = []
                endsx = []
                endsy = []


                for detail in details:
                    finite_idx_coef = np.where(np.abs(detail) != np.inf)
                    if finite_idx_coef[0].size == 0:
                        self.max_level = j-1
                        self.j2_eff = min(self.max_level, self.j2) # "effective" j2, used in linear regression
                        return

                    startsx.append(min(finite_idx_coef[0]))
                    startsy.append(min(finite_idx_coef[1]))
                    endsx.append(max(finite_idx_coef[0]))
                    endsy.append(max(finite_idx_coef[1]))

                startx = max(startsx)
                starty = max(startsy)
                endx = min(endsx) + 1
                endy = min(endsy) + 1

                finite_coefs = [detail[startx:endx, starty:endy] for detail in details]                

                self.wavelet_coeffs.add_values(finite_coefs, j)
                self.wavelet_coeffs_xpos.add_values(lesx[startx:endx], j)
                self.wavelet_coeffs_ypos.add_values(lesy[starty:endy], j)

                # wavelet leaders
                if compute_leaders:

                    if j == 1:

                        sans_voisin = [np.abs(detail) for detail in details]

                        if self.formalism == 'p-leader':
                        #    # detail_abs = (2.0**j)*(detail_abs**self.p)
                            sans_voisin = [2**(2*j)*detail_abs**self.p for detail_abs in sans_voisin]

                    else:
                        max_index = np.floor( np.array(sans_voisin[0].shape)/2 ).astype(int) 
                        
                        for idx in range(len(sans_voisin)):

                            sans_voisin[idx] = self._compute_leader_sans_voisin(details[idx], sans_voisin[idx], max_index, j)
                    leaders = []
                    
                    for idx in range(len(finite_coefs)):
                        leaders.append(self._compute_leader_from_neigbourhood(sans_voisin[idx]))

                    startsx = []
                    startsy = []
                    endsx = []
                    endsy = []

                    for leader in leaders:
                        finite_idx_wl = np.where(np.abs(leader) != np.inf)

                        startsx.append(min(finite_idx_wl[0]))
                        startsy.append(min(finite_idx_wl[1]))
                        endsx.append(max(finite_idx_wl[0]))
                        endsy.append(max(finite_idx_wl[1]))

                    startx = max(startsx)
                    starty = max(startsy)
                    endx = min(endsx) +1
                    endy = min(endsy) +1
                    
                    finite_wls = [leader[startx:endx, starty:endy] for leader in leaders]
                    
                    if self.formalism == "p-leader":
                        finite_wl = np.sum(finite_wls, axis=0)

                    else:
                        finite_wl = np.max(finite_wls, axis=0)

                    if finite_wl.size == 0:
                        self.max_level = j-1
                        self.j2_eff = min(self.max_level, self.j2) # "effective" j2, used in linear regression
                        break

                    if self.formalism == "p-leader":
                        finite_wl = np.power(  np.power(2., -2*j)*finite_wl, 1./self.p )

                    self.wavelet_leaders.add_values(finite_wl, j)
                    self.wavelet_leaders_xpos.add_values(lesx[startx:endx], j) 
                    self.wavelet_leaders_ypos.add_values(lesy[starty:endy], j)
                    print(j, lesx[startx:endx])
                    
                    #print(finite_wl.shape)
                    #print(finite_wl)


            #1D signal
            else:

                # apply filters
                # note: 'direct' method MUST be used, since there are elements
                # that are np.inf inside approx
                high    = convolve(approx, self.hi, mode = 'full', method='direct')
                low     = convolve(approx, self.lo, mode = 'full', method='direct')

                high[np.isnan(high)] = np.inf
                low[np.isnan(low)] = np.inf

                # index of first good value
                fp = self.filter_len - 2
                # index of last good value
                lp = nj_temp - 1

                # replace border with Inf
                high[0:fp]  = np.inf
                low[0:fp]   = np.inf
                high[lp+1:] = np.inf
                low[lp+1:]  = np.inf


                # offsets
                x0 = 2
                x0Appro = self.filter_len #2*self.nb_vanishing_moments

                # centering and subsampling
                detail_idx = np.arange(0, nj_temp, 2) + x0 - 1
                approx_idx = np.arange(0, nj_temp, 2) + x0Appro - 1
                detail = high[detail_idx]
                approx = low[approx_idx]

                # normalization
                detail = detail*2**(j*(0.5-1/self.normalization))

                # fractional integration
                detail = detail*2.0**(self.gamint*j)

                # remove infinite values and store wavelet coefficients
                finite_idx_coef  = np.logical_not( np.isinf( np.abs(detail) )  )

                if np.sum(finite_idx_coef) == 0:
                    self.max_level = j-1
                    break
                self.wavelet_coeffs.add_values(detail[finite_idx_coef], j)


                # wavelet leaders
                if compute_leaders:
                    detail_abs = np.abs(detail)
                    if j == 1:
                        if self.formalism == 'p-leader':
                            # detail_abs = (2.0**j)*(detail_abs**self.p)
                            detail_abs = np.power(2., j)*np.power(detail_abs,self.p)
                            leaders = np.vstack((
                                        detail_abs[0:len(detail_abs)-2],
                                        detail_abs[1:len(detail_abs)-1],
                                        detail_abs[2:len(detail_abs)]
                                        )).sum(axis = 0)
                            #leaders = (2.0**(-j)*leaders)**(1.0/self.p)
                            leaders = np.power(  np.power(2., -j)*leaders,
                                                1./self.p )

                        else:
                            leaders = np.vstack((
                                        detail_abs[0:len(detail_abs)-2],
                                        detail_abs[1:len(detail_abs)-1],
                                        detail_abs[2:len(detail_abs)]
                                        )).max(axis=0)

                        sans_voisin = detail_abs

                    else:
                        max_index    = int(np.floor( len(sans_voisin)/2 ))
                        detail_abs   = detail_abs[:max_index]

                        if self.formalism == 'p-leader':
                            #detail_abs = (2**j)*(detail_abs**self.p)
                            detail_abs = np.power(2., j)*np.power(detail_abs,self.p)
                            sans_voisin = np.vstack((
                                        detail_abs,
                                        sans_voisin[0:2*max_index:2],
                                        sans_voisin[1:2*max_index:2]
                                        )).sum(axis=0)

                            leaders = np.vstack((
                                        sans_voisin[0:len(sans_voisin)-2],
                                        sans_voisin[1:len(sans_voisin)-1],
                                        sans_voisin[2:len(sans_voisin)]
                                        )).sum(axis=0)

                            #leaders = (2.0**(-j)*leaders)**(1/self.p)
                            leaders = np.power(  np.power(2., -j)*leaders,
                                                1./self.p )
                        else:
                            sans_voisin = np.vstack((
                                        detail_abs,
                                        sans_voisin[0:2*max_index:2],
                                        sans_voisin[1:2*max_index:2]
                                        )).max(axis=0)
                            leaders = np.vstack((
                                        sans_voisin[0:len(sans_voisin)-2],
                                        sans_voisin[1:len(sans_voisin)-1],
                                        sans_voisin[2:len(sans_voisin)]
                                        )).max(axis=0)

                    # remove infinite values and store wavelet leaders
                    finite_idx_wl  = np.logical_not( np.isinf( np.abs(leaders) )  )
                    if np.sum(finite_idx_wl) == 0:
                        self.max_level = j-1
                        break
                    self.wavelet_leaders.add_values(leaders[finite_idx_wl], j)


        self.j2_eff = min(self.max_level, self.j2) # "effective" j2, used in linear regression
        
    def _estimate_hmin(self):
        """
        Estimate the value of the uniform regularity exponent hmin using
        wavelet coefficients.
        """
        sup_coeffs = np.zeros( self.j2_eff - self.j1 + 1)

        for j in range(self.j1, self.j2_eff+1):
            c_j = np.abs(self.wavelet_coeffs.values[j])
            sup_c_j = c_j.max()
            sup_coeffs[j-self.j1] = sup_c_j

        log_sup_coeffs = np.log2(sup_coeffs)

        # x, y and weights for linear regression
        x  = np.arange(self.j1, self.j2_eff+1)
        y  = log_sup_coeffs
        if self.wtype == 1:
            nj = self.wavelet_coeffs.get_nj_interv(self.j1, self.j2_eff)
        else:
            nj = np.ones(len(x))

        # linear regression
        slope, intercept = self.utils.linear_regression(x, y, nj)
        self.hmin        = slope


        # warning
        if (self.hmin < 0) and self.verbose > 0 and self.formalism != "p-leader":
            print("(!) hmin < 0. The value of gamint should be increased.")

        # plot
        plt = self.plt
        if self.verbose >= 2:
            # plot log_sup_coeffs
            plt.figure(self.HMIN_FIG_LABEL)
            plt.plot(x, y, 'r--.')
            plt.xlabel('j')
            plt.ylabel('$\log_2(\sup_k |d(j,k)|)$')
            plt.suptitle('$h_\mathrm{min}$')
            plt.draw()
            plt.grid()

            # plot regression line
            x0 = self.j1
            x1 = self.j2_eff 
            y0 = slope*x0 + intercept
            y1 = slope*x1 + intercept
            legend = '$h_\mathrm{min}$ = '+'%.5f' % (self.hmin)

            plt.plot([x0, x1], [y0, y1], color='k',
                    linestyle='-', linewidth=2, label = legend)
            plt.legend()
            plt.draw()

    def _estimate_eta_p(self):
        """
        Estimate the value of eta_p
        """
        assert self.formalism == 'p-leader'
        wavelet_structure = StructureFunction(self.wavelet_coeffs,
                                              np.array([self.p]),
                                              self.j1, self.j2_eff, self.wtype)
        self.eta_p = wavelet_structure.zeta[0]


    def _correct_leaders(self):
        """
        Correct p-leaders for nonlinearity (according to the Matlab toolbox)
        """
        if self.eta_p > 0:
            JJ = np.arange(1, self.max_level+1)
            J1LF = 1
            JJ0 = JJ - J1LF + 1
            zqhqcorr = np.log2(  (1 - np.power(2., -JJ0*self.eta_p)) / ( 1 - np.power(2., -self.eta_p))   )
            ZPJCorr  = np.power(2, (-1.0/self.p)*zqhqcorr)

            for ind_j, j in enumerate(JJ):
                self.wavelet_leaders.values[j] = \
                    self.wavelet_leaders.values[j]*ZPJCorr[ind_j]
        else:
            if self.verbose > 0:
                print("(!) Warning: eta(p) <= 0, p-Leaders correction was not applied. A smaller value of p (or larger value of gamint) should be selected.")


    def plot_cumulants(self, show = False):
        """
        Plot cumulants
        """
        assert self.cumulants is not None, u"mfa.analyze() should be called before plotting"
        self.cumulants.plot(self.CUMUL_FIG_LABEL)

        if show:
            self.plt.show()

    def plot_structure(self, show = False):
        """
        Plot structure function
        """
        assert self.structure is not None, u"mfa.analyze() should be called before plotting"
        self.structure.plot(self.STRUCTURE_FIG_LABEL, self.SCALING_FIG_LABEL)

        if show:
            self.plt.show()


    def plot_spectrum(self, show = False):
        """
        Plot multifractal spectrum
        """
        assert self.spectrum is not None, u"mfa.analyze() should be called before plotting"
        self.spectrum.plot()

        if show:
            self.plt.show()

    

    def analyze_no_estimation(self,signal):
        self._set_and_verify_parameters()

        # Clear previously computed data
        self.wavelet_coeffs = None
        self.wavelet_leaders = None
        self.structure = None
        self.cumulants = None

        # Compute wavelet coefficients and wavelet leaders
        self._wavelet_analysis(signal)

        # p-leader correction
        if self.formalism == 'p-leader':
            self._estimate_eta_p()
            self._correct_leaders()
        else:
            self.eta_p = np.inf


    def perform_estimations_subset(self, subset, j1, j2):
        old_j1 = self.j1
        old_j2 = self.j2_eff
        old_coef = self.wavelet_coeffs
        old_leaders = self.wavelet_leaders

        self.wavelet_coeffs = MultiResolutionQuantity(self.formalism)
        self.wavelet_leaders = MultiResolutionQuantity(self.formalism)

        for j in range(j1, j2+1):

            c2c = old_coef.values[j][1] - old_coef.values[j][0]

           # sub_coeff = old_coef.values[j][]

            #sub_leaders = 
            wavelet_coeffs.add_values(sub_coeff, j)
            wavelet_leaders.add_values(sub_leaders, j)

        self.j1 = j1
        self.j2_eff = j2
  

        self._estimate_hmin()





        self.j1 = old_j1
        self.j2_eff = old_j2
        self.wavelet_coeffs = old_coef
        self.wavelet_leaders = old_leaders

    def perform_estimations(self):
        # Compute hmin
        #self._estimate_hmin()

        # Compute structure functions and cumulants
        if self.formalism == 'wcmf':
            self.structure = StructureFunction(self.wavelet_coeffs,
                                self.q,
                                self.j1,
                                self.j2_eff,
                                self.wtype)
            self.cumulants = Cumulants(self.wavelet_coeffs,
                                self.n_cumul,
                                self.j1,
                                self.j2_eff,
                                self.wtype)
            self.spectrum = MultifractalSpectrum(self.wavelet_coeffs,
                                     self.q,
                                     self.j1,
                                     self.j2_eff,
                                     self.wtype)


        elif self.formalism == 'wlmf' or self.formalism == 'p-leader':
            self.structure = StructureFunction(self.wavelet_leaders,
                                self.q,
                                self.j1,
                                self.j2_eff,
                                self.wtype)
            self.cumulants = Cumulants(self.wavelet_leaders,
                                self.n_cumul,
                                self.j1,
                                self.j2_eff,
                                self.wtype)

            self.spectrum = MultifractalSpectrum(self.wavelet_leaders,
                                     self.q,
                                     self.j1,
                                     self.j2_eff,
                                     self.wtype)

    def analyze(self, signal):
        # Verify parameters
        self._set_and_verify_parameters()

        # Clear previously computed data
        self.wavelet_coeffs = None
        self.wavelet_leaders = None
        self.structure = None
        self.cumulants = None

        # Compute wavelet coefficients and wavelet leaders
        self._wavelet_analysis(signal)

        # Compute hmin
        self._estimate_hmin()

        # p-leader correction
        if self.formalism == 'p-leader':
            self._estimate_eta_p()
            self._correct_leaders()
        else:
            self.eta_p = np.inf

        # Compute structure functions and cumulants
        if self.formalism == 'wcmf':
            self.structure = StructureFunction(self.wavelet_coeffs,
                                self.q,
                                self.j1,
                                self.j2_eff,
                                self.wtype)
            self.cumulants = Cumulants(self.wavelet_coeffs,
                                self.n_cumul,
                                self.j1,
                                self.j2_eff,
                                self.wtype)
            self.spectrum = MultifractalSpectrum(self.wavelet_coeffs,
                                     self.q,
                                     self.j1,
                                     self.j2_eff,
                                     self.wtype)


        elif self.formalism == 'wlmf' or self.formalism == 'p-leader':
            self.structure = StructureFunction(self.wavelet_leaders,
                                self.q,
                                self.j1,
                                self.j2_eff,
                                self.wtype)
            self.cumulants = Cumulants(self.wavelet_leaders,
                                self.n_cumul,
                                self.j1,
                                self.j2_eff,
                                self.wtype)

            self.spectrum = MultifractalSpectrum(self.wavelet_leaders,
                                     self.q,
                                     self.j1,
                                     self.j2_eff,
                                     self.wtype)
        # Plot
        if self.verbose >= 2:
            self.structure.plot(self.STRUCTURE_FIG_LABEL, self.SCALING_FIG_LABEL)
            self.cumulants.plot(self.CUMUL_FIG_LABEL)
            self.spectrum.plot()
            self.plot_signal(signal)
            self.plt.show()

    def plot_signal(self, signal):

        plt.figure("Original Signal")
        plt.imshow(signal, cmap=plt.get_cmap('gray'))
        #plt.grid()
        #plt.xlabel('h(q)')
        #plt.ylabel('D(q)')
        #plt.suptitle(self.name + ' - multifractal spectrum')
        plt.draw()

    def compute_hurst(self, signal):
        """
        Estimate the Hurst exponent using the wavelet structure function for q=2
        """

        # Verify parameters
        self._set_and_verify_parameters()

        # Clear previously computed data
        self.wavelet_coeffs = None
        self.wavelet_leaders = None
        self.structure = None
        self.cumulants = None

        # Compute wavelet coefficients and wavelet leaders
        self._wavelet_analysis(signal)


        structure_dwt = StructureFunction(self.wavelet_coeffs,
                                          np.array([2.0]),
                                          self.j1,
                                          self.j2_eff,
                                          self.wtype)
        self.structure = structure_dwt

        log2_Sj_2 = np.log2(structure_dwt.values[0,:])  # function log2(S(j, 2))
        self.hurst_structure = log2_Sj_2

        hurst  = structure_dwt.zeta[0]/2

        if self.verbose >= 2:
            structure_dwt.plot(self.STRUCTURE_FIG_LABEL, self.SCALING_FIG_LABEL)
            self.plt.show()


        return hurst


    def bivariate_analysis(self, signal_1, signal_2):
        """
        Bivariate multifractal analysis according to:
            https://www.irit.fr/~Herwig.Wendt/data/Wendt_EUSIPCO_talk_V01.pdf

        Signals are cropped to have the same length = min(len(signal_1), len(signal_2))
        """

        # Adjust lengths
        length = min(len(signal_1), len(signal_2))
        signal_1 = signal_1[:length]
        signal_2 = signal_2[:length]

        # Verify parameters
        self._set_and_verify_parameters()

        # Clear previously computed data
        self.wavelet_coeffs = None
        self.wavelet_leaders = None

        # Compute multiresolution quantities from signals 1 and 2
        self._wavelet_analysis(signal_1)
        wavelet_coeffs_1 = self.wavelet_coeffs
        wavelet_leaders_1 = self.wavelet_leaders

        self._wavelet_analysis(signal_2)
        wavelet_coeffs_2 = self.wavelet_coeffs
        wavelet_leaders_2 = self.wavelet_leaders


        # Choose quantities according to the formalism
        if self.formalism == 'wcmf':
            mrq_1 = wavelet_coeffs_1
            mrq_2 = wavelet_coeffs_2
        elif self.formalism == 'wlmf' or self.formalism == 'p-leader':
            mrq_1 = wavelet_leaders_1
            mrq_2 = wavelet_leaders_2


        # Compute structure functions
        self.bi_structure = BiStructureFunction(mrq_1, 
                                                mrq_2, 
                                                self.q, 
                                                self.q, 
                                                self.j1, 
                                                self.j2_eff, 
                                                self.wtype)

        # Compute cumulants
        self.bi_cumulants = BiCumulants(mrq_1,
                                        mrq_2,
                                        1, 
                                        self.j1, 
                                        self.j2_eff, 
                                        self.wtype)

        warnings.warn("Bivariate cumulants (class BiCumulants) are only implemented for C10(j), C01(j) and C11(j).\
         All moments are already computed (BiCumulants.moments), we need to implement only the relation between multivariate cumulants and moments.")


        if self.verbose >= 2:
            self.bi_structure.plot()
            self.bi_cumulants.plot(self.CUMUL_FIG_LABEL)
            self.plt.show()



    def _test(self, signal):
        """
        Used for development purposes
        """
        self._set_and_verify_parameters()
        self.wavelet_coeffs = None
        self.wavelet_leaders = None
        self.structure = None
        self.cumulants = None   
            
        # Compute wavelet coefficients and wavelet leaders
        self._wavelet_analysis(signal)

        self.spectrum = MultifractalSpectrum(self.wavelet_leaders,
                                             self.q,
                                             self.j1,
                                             self.j2_eff,
                                             self.wtype)

  #   %===============================================================================
  #   %- ANCILLARY SUBROUTINES
  #   %===============================================================================
  #  function [sansv] = compute_leader_sans_voisin(coef, sansv, nc, p, j)
  #  % Computes leaders without neighbours from coefs at current scale (j) and
  #  % leaders sans_voisin at the previous scale.
  #  % nc indicates the numer of coefficients at curr scale that will be used
#
  #      tmp(:, :, 1) = abs (coef(1 : nc(1), 1 : nc(2)));
  #      if p ~= inf    % p-leaders
  #          tmp(:, :, 1) = 2 ^ (2 * j) .* tmp(:, :, 1) .^ p;
  #      end
  #      tmp(:, :, 2) = sansv(1 : 2 : 2*nc(1) , 1 : 2 : 2*nc(2));
  #      tmp(:, :, 3) = sansv(2 : 2 : 2*nc(1) , 1 : 2 : 2*nc(2));
  #      tmp(:, :, 4) = sansv(1 : 2 : 2*nc(1) , 2 : 2 : 2*nc(2));
  #      tmp(:, :, 5) = sansv(2 : 2 : 2*nc(1) , 2 : 2 : 2*nc(2));
#
  #      if p == inf
  #          sansv = max (tmp, [], 3);  % Trailing singleton dim is squeezed out
  #      else
  #          sansv = sum (tmp, 3);  % Trailing singleton dim is squeezed out
  #      end
#
  #  end  % compute_leader_sans_voisin



    def _compute_leader_sans_voisin(self, coef, sansv, max_index, j):

        tmp = [np.abs(coef[0:max_index[0], 0:max_index[1]])]

        if self.formalism == "p-leader":
            tmp[0] = 2**(2*j)*tmp[0]**self.p

        tmp.append(sansv[0:2*max_index[0]-1:2, 1:2*max_index[1]:2])        
        tmp.append(sansv[1:2*max_index[0]:2, 1:2*max_index[1]:2])        
        tmp.append(sansv[0:2*max_index[0]-1:2, 0:2*max_index[1]-1:2])        
        tmp.append(sansv[1:2*max_index[0]:2, 0:2*max_index[1]-1:2])        

        if self.formalism == 'p-leader': 
            return sum(tmp)
        else:
            return np.max(tmp, axis=0)


  #       %-------------------------------------------------------------------------------
  #  function [leader] = compute_leader_from_neigbourhood (sansv, p)
  #  % Computes each leader from all leaders_sans_voisin in the neighbourhood
#
  #      si = size (sansv);
  #      ls = zeros (2 + si(1), 2 + si(2));
  #      ls(2 : end-1, 2 : end-1) = sansv;
  #      tmp(:, :, 1) = ls(1 : end - 2, 1 : end - 2);
  #      tmp(:, :, 2) = ls(1 : end - 2, 2 : end - 1);
  #      tmp(:, :, 3) = ls(1 : end - 2, 3 : end    );
  #      tmp(:, :, 4) = ls(2 : end - 1, 1 : end - 2);
  #      tmp(:, :, 5) = ls(2 : end - 1, 2 : end - 1);
  #      tmp(:, :, 6) = ls(2 : end - 1, 3 : end    );
  #      tmp(:, :, 7) = ls(3 : end    , 1 : end - 2);
  #      tmp(:, :, 8) = ls(3 : end    , 2 : end - 1);
  #      tmp(:, :, 9) = ls(3 : end    , 3 : end    );
#
  #      if p == inf    % wavelet leaders
  #          leader = max (tmp, [], 3);
  #      else    % p-leaders
  #          leader = sum (tmp, 3);
  #      end
#
  #  end
  #  %---------
    def _compute_leader_from_neigbourhood(self, sansv):

        ls = np.zeros(np.array(sansv.shape)+2)
        ls [1:-1, 1:-1] = sansv
        tmp = []

        for i in range(3):
            for j in range(3):
                endi = ls.shape[0]-(2-i)
                endj = ls.shape[1]-(2-j)
                tmp.append(ls[i:endi, j:endj])

        if self.formalism == 'p-leader': 
            return sum(tmp)
        else:
            return np.max(tmp, axis=0)
