#!/usr/bin/python

import numpy as np
from .noise_estimator import ImcraNoiseEstimator
from .suppression_gain import OmlsaGain

'''
Constants
'''
# zero_thres is a threshold for discriminating between zero and nonzero sample.
zero_thres = 1e-10    


'''
Class
'''
class NoiseSuppressor(object):
    def __init__(self, sample_rate, frame_size, fft_size, overlap_size):
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.overlap_size = overlap_size
        self.fft_size = fft_size
        self.win =np.hamming(fft_size)
        self.in_buffer = np.zeros(fft_size)
        self.out_buffer = np.zeros(fft_size)
        self.noise_estimator = ImcraNoiseEstimator()
        self.suppression_gain = OmlsaGain(sample_rate, fft_size)
        self.fnz_flag = 0     # flag for the first frame which is non-zero  

    def stft_analyze(self, audio):
        M = self.fft_size
        M21 = int(M/2+1)
        Mno = int(M - self.overlap_size)

        self.in_buffer[:M-Mno] = self.in_buffer[Mno:M]    # update the frame of data
        self.in_buffer[M-Mno:M] = audio 
        signal_spec = np.zeros(M)
        signal_power = np.zeros(M21)

        if ((self.fnz_flag==0 and abs(self.in_buffer[1])>zero_thres)) or \
             (self.fnz_flag==1 and any(abs(self.in_buffer)>zero_thres)) :     
            self.fnz_flag = 1   
            # 1. Short Time Fourier Analysis
            signal_spec = np.fft.fft(self.win * self.in_buffer)
            signal_power = abs(signal_spec[:M21])**2

        return signal_spec, signal_power

    #def stft_synthesize(self, audio): 

    def process_frame(self, frame_data):

        M = self.fft_size
        M21 = int(M/2+1)
        Mno = int(M - self.overlap_size)

        #0 STFT Analysis
        signal_spec, signal_power = self.stft_analyze(frame_data)
        yout = np.zeros(Mno)

        if self.fnz_flag == 1 :  
            #1 rough noise estimation
            #2 rough a priori and posteri snr estimation
            #3 speech presence prabability estimation
            #4 precise noise estimation
            #5 a priori and posteri snr estimation
            features= {'signal_power': signal_power, 
                        'eta_2term': self.suppression_gain.get_eta()}
            noise_power = self.noise_estimator.update(features)
            
            #6 Update suppression gain
            features= {'signal_power': signal_power, 
                        'noise_power': noise_power}
            gain = self.suppression_gain.update(features)

            #7 STFT Synthesis
            X = gain * signal_spec[:M21]
            x = self.win *np.fft.irfft(X)
            self.out_buffer = self.out_buffer + x

            yout = self.out_buffer[:Mno] * 1.0
            self.out_buffer[:M-Mno] = self.out_buffer[Mno:M]   # update output frame
            self.out_buffer[M-Mno:M] = np.zeros(Mno)   # update output frame
        
        return yout