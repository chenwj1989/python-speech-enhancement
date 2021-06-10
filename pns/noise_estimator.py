import numpy as np


'''
Constants
'''
# 1) Parameters of Short Time Fourier Analysis:
Fs_ref = 16e3		# 1.1) Reference Sampling frequency
M_ref = 512		# 1.2) Size of analysis window
#Mo_ref = 0.75*M_ref	# 1.3) Number of overlapping samples in consecutive frames
Mo_ref = 352

# 2) Parameters of Noise Spectrum Estimate
w = 1			# 2.1)  Size of frequency smoothing window function = 2*w+1
alpha_s_ref = 0.9	# 2.2)  Recursive averaging parameter for the smoothing operation
Nwin = 8 	# 2.3)  Resolution of local minima search
Vwin = 15
delta_s = 1.67		# 2.4)  Local minimum factor
Bmin = 1.66
delta_y = 4.6		# 2.4)  Local minimum factor
delta_yt = 3
alpha_d_ref = 0.85	# 2.7)  Recursive averaging parameter for the noise

# 3) Parameters of a Priori Probability for Signal-Absence Estimate
alpha_xi_ref = 0.7	# 3.1) Recursive averaging parameter

# 4) Parameters of "Decision-Directed" a Priori SNR Estimate
alpha_eta_ref = 0.95	# 4.1) Recursive averaging parameter
eta_min_dB = -18	# 4.2) Lower limit constraint

# 5) Flags
nonstat = 'medium'                #Non stationarity  # new version

Fs = Fs_ref
M = int(M_ref)
Mo = int(Mo_ref)
Mno = int(M-Mo)
alpha_s = alpha_s_ref
alpha_d = alpha_d_ref
alpha_eta = alpha_eta_ref
alpha_xi = alpha_xi_ref

alpha_d_long = 0.99
eta_min = 10**(eta_min_dB/10)

#b = hanning(2*w+1)
#b = b/sum(b)     # normalize the window function
b = np.array([0, 1, 0])

M21 = int(M/2+1)

class NoiseEstimator(object):
    def update(self, features):
        pass

class ImcraNoiseEstimator(NoiseEstimator):
    def __init__(self):    
        self.l = 0   #count of frame
        self.l_mod_lswitch = 0
        self.S = np.zeros(M21)              
        self.St = np.zeros(M21)                
        self.Sy = np.zeros(M21)       
        self.Smin = np.zeros(M21)        
        self.Smint = np.zeros(M21)      
        self.SMact = np.zeros(M21)    
        self.SMactt = np.zeros(M21)    
        self.SW = np.zeros((M21,Nwin))
        self.SWt = np.zeros((M21,Nwin))
        self.lambda_d = np.zeros(M21)   
        self.lambda_dav = np.zeros(M21)   

    def update(self, features):
        Ya2 = features['signal_power']
        self.eta_2term = features['eta_2term']
        
        self.l = self.l + 1
        gamma = Ya2 / np.maximum(self.lambda_d, 1e-10) #post_snr
        eta = alpha_eta*self.eta_2term + (1-alpha_eta)*np.maximum(gamma-1,0)  #prior_snr
        eta = np.maximum(eta,eta_min)  
        v = gamma*eta/(1+eta)

        # 2.1. smooth over frequency
        Sf = np.convolve(b, Ya2)  # smooth over frequency
        Sf = Sf[w:M21+w]
        #         if l==1   
        if self.l == 1 :    
            self.Sy = Ya2
            self.S = Sf
            self.St = Sf
            self.lambda_dav = Ya2
        else :
            self.S = alpha_s * self.S + (1-alpha_s) * Sf     # smooth over time

        if self.l < 15 :    
            self.Smin = self.S
            self.SMact = self.S
        else :
            self.Smin = np.minimum(self.Smin, self.S)
            self.SMact = np.minimum(self.SMact, self.S)

        # Local Minima Search
        I_f = np.zeros(M21)
        for i in range(M21) :
            I_f[i] = Ya2[i]<delta_y*Bmin*self.Smin[i] and self.S[i]<delta_s*Bmin*self.Smin[i] and 1
        conv_I = np.convolve(b, I_f)
        conv_I = conv_I[w:M21+w]
        Sft = self.St
        idx = [i for i, v in enumerate(conv_I) if v>0] 
        if len(idx)!=0 :
            if w :
                conv_Y = np.convolve(b, I_f*Ya2)
                conv_Y = conv_Y[w:M21+w]
                Sft[idx] = conv_Y[idx]/conv_I[idx]
            else :
                Sft[idx] = Ya2[idx]

        if self.l < 15 :
            self.St = self.S
            self.Smint = self.St
            self.SMactt = self.St
        else : 
            self.St[:] = alpha_s * self.St + (1-alpha_s) * Sft
            self.Smint[:] = np.minimum(self.Smint, self.St)
            self.SMactt[:] = np.minimum(self.SMactt, self.St)

        qhat = np.ones(M21)
        phat = np.zeros(M21)

        if nonstat  == 'low' : 
            gamma_mint = Ya2/Bmin/np.maximum(self.Smin,1e-10) 
            zetat = self.S/Bmin/np.maximum(self.Smin,1e-10)     
        else : 
            gamma_mint = Ya2/Bmin/np.maximum(self.Smint,1e-10)   
            zetat = self.S/Bmin/np.maximum(self.Smint,1e-10)    

        for idx in range(M21) :
            if gamma_mint[idx]>1 and gamma_mint[idx]<delta_yt and zetat[idx]<delta_s :
                qhat[idx] = (delta_yt-gamma_mint[idx])/(delta_yt-1)
                phat[idx] = 1/(1+qhat[idx]/(1-qhat[idx])*(1+eta[idx])*np.exp(-v[idx]))
            if gamma_mint[idx]>delta_yt  or  zetat[idx]>=delta_s :
                phat[idx] = 1
        
        self.l_mod_lswitch = self.l_mod_lswitch + 1
        if self.l_mod_lswitch == Vwin :
            self.l_mod_lswitch = 0

            if self.l == Vwin : 
                for i in range(Nwin):
                    self.SW[:,i] = self.S
                    self.SWt[:, i] = self.St
            else :
                self.SW[:,:Nwin-1] = self.SW[:,1:Nwin]
                self.SW[:,Nwin-1] = self.SMact
                self.Smin = self.SW.min(1)
                self.SMact = self.S
                self.SWt[:,:Nwin-1] = self.SWt[:,1:Nwin]
                self.SWt[:,Nwin-1] = self.SMactt
                self.Smint = self.SWt.min(1)
                self.SMactt = self.St

        alpha_dt = alpha_d + (1-alpha_d)*phat
        self.lambda_dav = alpha_dt * self.lambda_dav + (1-alpha_dt)*Ya2
        if self.l < 15 :
            self.lambda_dav_long = self.lambda_dav
        else :
            alpha_dt_long = alpha_d_long + (1-alpha_d_long)*phat
            self.lambda_dav_long = alpha_dt_long * self.lambda_dav_long + (1-alpha_dt_long)*Ya2

        # 2.4. Noise Spectrum Estimate
        if nonstat == 'high' :
            self.lambda_d = 2 * self.lambda_dav  
        else :
            self.lambda_d = 1.4685 * self.lambda_dav  

        return self.lambda_d


