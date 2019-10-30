#!/usr/bin/python
from __future__ import division

import numpy as np
from .suppression_gain import ns_gain, default_gain_parameters
from .compute_snr import compute_snr, default_snr_parameters
from .noise_estimation import noise_estimation, default_noise_parameters
from .speech_precense_probability import spp, default_spp_parameters


class NoiseSuppression:
    def __init__(self, gain_method, noise_method, frame_size, win_size, nFFT):
        self.frame_size = frame_size
        self.overlap_size = win_size - frame_size
        self.win_size = win_size
        self.nFFT = nFFT
        self.qk = np.ones(nFFT) * 0.5
        self._init_win()

        self.noise_mu2 = np.zeros(nFFT)
        self.ksi = np.ones(nFFT)
        self.in_data = np.zeros(win_size)
        self.out_data = np.zeros(win_size)

        self.gain_parameters = default_gain_parameters()
        self.noise_parameters = default_noise_parameters()
        self.spp_parameters = default_spp_parameters()
        self.snr_parameters = default_snr_parameters()
        self.gain_parameters['gain_method'] = gain_method
        self.noise_parameters['noise_method'] = noise_method


    def _init_win(self):
        win = np.hanning(self.win_size)
        self.win = win * self.frame_size / np.sum(win)

    def set_priori_noise(self, noise_power):
        np.noise_mu2 = noise_power

    def set_priori_snr(self, ksi):
        self.ksi = ksi

    def get_parameters(self):
        parameters = {'gain_parameters':    self.gain_parameters,
                      'noise_parameters':   self.noise_parameters,
                      'spp_parameters':     self.spp_parameters,
                      'snr_parameters':     self.snr_parameters}
        return parameters

    def set_parameters(self, parameters):
        self.gain_parameters = parameters['gain_parameters']
        self.noise_parameters = parameters['noise_parameters']
        self.spp_parameters = parameters['spp_parameters']
        self.snr_parameters = parameters['snr_parameters']
        return

    def process_frame(self, frame_data):

        #0 windowing and stft
        self.in_data[:self.overlap_size] = self.in_data[self.frame_size:]
        self.in_data[self.overlap_size:] = frame_data

        if(len(self.win) == 0 ):
            self._init_win()

        win_data = self.win * self.in_data
        spec = np.fft.fft(win_data, self.nFFT, axis=0)
        sig = np.absolute(spec)
        sig2 = sig ** 2
        #1 rough noise estimation
        #currenly use estimated noise of previous frame

        #2 rough a priori and posteri snr estimation
        gamma, ksi = compute_snr(sig2, self.noise_mu2, self.ksi, self.snr_parameters)
        self.ksi = ksi

        #3 speech presence prabability estimation
        self.spp_parameters['ksi'] = ksi
        self.spp_parameters['gamma'] = gamma
        pSpeech = spp(self.qk, self.spp_parameters)

        #4precise noise estimation
        self.noise_parameters['P_speech'] = pSpeech
        self.noise_parameters['ksi'] = ksi
        self.noise_parameters['gamma'] = gamma
        self.noise_mu2 = noise_estimation(sig2, self.noise_mu2, self.noise_parameters)

        #5 a priori and posteri snr estimation
        gamma, ksi = compute_snr(sig2, self.noise_mu2, self.ksi, self.snr_parameters)
        self.ksi = ksi

        #6 gain
        self.gain_parameters['ksi'] = ksi
        self.gain_parameters['gamma'] = gamma
        gain = ns_gain(self.gain_parameters)

        #xi_w = np.fft.ifft(spec * gain * pSpeech, self.nFFT, axis=0)
        xi_w = np.fft.ifft(spec * gain, self.nFFT, axis=0)
        xi_w = np.real(xi_w)

        self.out_data[:self.overlap_size] = self.out_data[self.frame_size:]
        self.out_data[self.overlap_size:] = np.zeros(self.frame_size)
        self.out_data = self.out_data + xi_w[:self.win_size]

        return self.out_data[:self.frame_size]