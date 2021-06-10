import numpy as np
from numpy import matlib
from scipy.special import expn

'''
Constants
'''
# 1) Parameters of Short Time Fourier Analysis:
Fs_ref = 16e3		# 1.1) Reference Sampling frequency
M_ref = 512		# 1.2) Size of analysis window
#Mo_ref = 0.75*M_ref	# 1.3) Number of overlapping samples in consecutive frames
Mo_ref = 352

# 3) Parameters of a Priori Probability for Signal-Absence Estimate
alpha_xi_ref = 0.7	# 3.1) Recursive averaging parameter
w_xi_local = 1 	# 3.2) Size of frequency local smoothing window function
w_xi_global = 15 	# 3.3) Size of frequency local smoothing window function
f_u = 10e3 		# 3.4) Upper frequency threshold for global decision
f_l = 50 		# 3.5) Lower frequency threshold for global decision
P_min = 0.005 		# 3.6) Lower bound constraint
xi_lu_dB = -5 	# 3.7) Upper threshold for local decision
xi_ll_dB = -10 	# 3.8) Lower threshold for local decision
xi_gu_dB = -5 	# 3.9) Upper threshold for global decision
xi_gl_dB = -10 	# 3.10) Lower threshold for global decision
xi_fu_dB = -5 	# 3.11) Upper threshold for local decision
xi_fl_dB = -10 	# 3.12) Lower threshold for local decision
xi_mu_dB = 10 	# 3.13) Upper threshold for xi_m
xi_ml_dB = 0 		# 3.14) Lower threshold for xi_m
q_max = 0.998 		# 3.15) Upper limit constraint

# 4) Parameters of "Decision-Directed" a Priori SNR Estimate
alpha_eta_ref = 0.95	# 4.1) Recursive averaging parameter
eta_min_dB = -18	# 4.2) Lower limit constraint

# 5) Flags
broad_flag = 1               # broad band flag   # new version
tone_flag = 0                # pure tone flag   # new version
nonstat = 'medium'                #Non stationarity  # new version

Fs = Fs_ref
M = int(M_ref)
Mo = int(Mo_ref)
Mno = int(M-Mo)
alpha_eta = alpha_eta_ref
alpha_xi = alpha_xi_ref

alpha_d_long = 0.99
eta_min = 10**(eta_min_dB/10)
G_f = eta_min**0.5	   # Gain floor


##b_xi_local = hanning(2*w_xi_local+1)
#b_xi_local = b_xi_local/sum(b_xi_local)  # normalize the window function
b_xi_local = np.array([0, 1, 0])
#b_xi_global = hanning(2*w_xi_global+1)
#b_xi_global = b_xi_global/sum(b_xi_global)   # normalize the window function
b_xi_global = np.array([0, 0.000728, 0.002882, 0.006366, 0.011029, 0.016667, 0.023033, 0.029849, 0.036818, 0.043634, 0.050000, 0.055638, 0.060301, 0.063785, 0.065938, 0.066667, 0.065938, 0.063785, 0.060301, 0.055638, 0.050000, 0.043634, 0.036818, 0.029849, 0.023033, 0.016667, 0.011029, 0.006366, 0.002882, 0.000728, 0
])


M21 = int(M/2+1)
k_u = round(f_u/Fs*M+1)  # Upper frequency bin for global decision
k_l = round(f_l/Fs*M+1)  # Lower frequency bin for global decision
k_u = min(k_u,M21)
k2_local=round(500/Fs*M+1)
k3_local = round(3500/Fs*M+1)

class SuppressionGain(object):
    def update(self, features):
        pass

class WienerGain(SuppressionGain):
    def update(self, features):
        '''
        ksi : a priori snr
        '''
        gain = features.ksi / (1 + features.ksi) 
        return gain

class OmlsaGain(SuppressionGain):
    def __init__(self, sample_rate, fft_size):
        self.fs = sample_rate
        self.fft_size = fft_size
        self.M21 = int(fft_size/2+1)
        self.eta_2term = np.ones(M21) 
        self.xi = np.ones(M21) 
        self.xi_frame = 0
        self.xi_m_dB = 0

    def update(self, features):
        Ya2 = features['signal_power']
        lambda_d = features['noise_power']
        
        gamma = Ya2 / np.maximum(lambda_d, 1e-10) #post_snr
        eta = alpha_eta*self.eta_2term + (1-alpha_eta)*np.maximum(gamma-1,0)  #prior_snr
        eta = np.maximum(eta,eta_min)  
        v = gamma*eta/(1+eta)

        # A Priori Probability for Signal-Absence Estimate
        self.xi = alpha_xi * self.xi + (1-alpha_xi) * eta
        xi_local = np.convolve(self.xi, b_xi_local)
        xi_local = xi_local[w_xi_local:self.M21+w_xi_local]
        xi_global = np.convolve(self.xi, b_xi_global)
        xi_global = xi_global[w_xi_global:self.M21+w_xi_global]
        dxi_frame = self.xi_frame
        self.xi_frame = np.mean(self.xi[k_l:k_u])
        dxi_frame = self.xi_frame - dxi_frame

        xi_local_dB = np.zeros(len(xi_local))
        xi_global_dB = np.zeros(len(xi_global))

        for i in range(len(xi_local)) :
            if xi_local[i] > 0 :
                xi_local_dB[i] = 10*np.log10(xi_local[i])  
            else : 
                xi_local_dB[i] = -100 

        for i in range(len(xi_global)) :
            if xi_global[i] >0 : 
                xi_global_dB[i] = 10*np.log10(xi_global[i]) 
            else :
                xi_global_dB[i] = -100

        if self.xi_frame >0 :
            xi_frame_dB = 10*np.log10(self.xi_frame) 
        else :
            xi_frame_dB = -100

        P_local = np.ones(M21)
        for idx in range(M21) :
            if xi_local_dB[idx] <= xi_ll_dB:
                P_local[idx] = P_min
            if xi_local_dB[idx] > xi_ll_dB  and xi_local_dB[idx] < xi_lu_dB :
                P_local[idx] = P_min + (xi_local_dB[idx]-xi_ll_dB) / (xi_lu_dB-xi_ll_dB) * (1-P_min)

        P_global = np.ones(M21)
        for idx in range(M21) :
            if xi_global_dB[idx] <= xi_gl_dB:
                P_global[idx] = P_min
            if xi_global_dB[idx] >xi_gl_dB  and xi_global_dB[idx] <xi_gu_dB :
                P_global[idx] = P_min + (xi_global_dB[idx]-xi_gl_dB)/(xi_gu_dB-xi_gl_dB)*(1-P_min)

        m_P_local = np.mean(P_local[2:(k2_local+k3_local-3)])    # average probability of speech presence
        if m_P_local < 0.25 :
            P_local[k2_local:k3_local] = P_min    # reset P_local (frequency>500Hz) for low probability of speech presence

        if xi_frame_dB <= xi_fl_dB :
            P_frame = P_min
        elif dxi_frame >= 0 :
            self.xi_m_dB = min(max(xi_frame_dB,xi_ml_dB),xi_mu_dB)
            P_frame = 1
        elif xi_frame_dB >= self.xi_m_dB + xi_fu_dB :
            P_frame = 1
        elif xi_frame_dB <= self.xi_m_dB + xi_fl_dB :
            P_frame = P_min
        else :
            P_frame = P_min+(xi_frame_dB-self.xi_m_dB-xi_fl_dB)/(xi_fu_dB-xi_fl_dB)*(1-P_min)

        #     q=1-P_global.*P_local*P_frame   # new version
        if broad_flag :  # new version
            q = 1 - P_global * P_local * P_frame   # new version
        else :  # new version
            q = 1 - P_local * P_frame   ##ok<UNRCH> # new version

        q = np.minimum(q, q_max)
        gamma = np.zeros(M21)
        gamma = Ya2 / np.maximum(lambda_d, 1e-10)
        eta = alpha_eta * self.eta_2term + (1-alpha_eta) * np.maximum(gamma-1,0)
        eta = np.maximum(eta, eta_min)
        v = gamma*eta/(1+eta)
        PH1 = np.zeros(M21)
        idx = [i for i, v in enumerate(q) if v<0.9]
        PH1[idx] = 1 / ( 1+q[idx] / (1-q[idx]) * (1+eta[idx]) * np.exp(-v[idx]) )

        # Spectral Gain
        GH1 = np.ones(M21)

        idx = [i for i, val in enumerate(v) if val>5 ]
        GH1[idx] = eta[idx] / (1+eta[idx])
        idx = [i for i, val in enumerate(v) if val<=5 and val>0]
        GH1[idx] = eta[idx] / (1+eta[idx]) * np.exp(0.5 * expn(1, v[idx]))

        GH0 = G_f  

        G = GH1**PH1 * GH0**(1 - PH1)
        self.eta_2term = GH1**2 * gamma
        return G

    def get_eta(self):
        return self.eta_2term