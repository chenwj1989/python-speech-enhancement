#!/usr/bin/python
from __future__ import division

from suppression_gain import suppression_gain, default_gain_parameters
from compute_snr import compute_snr, default_snr_parameters
from noise_estimation import noise_estimation, default_noise_parameters
from speech_precense_probability import spp, sap, default_spp_parameters

def ns_init():
    parameters = {'gain_parameters': default_gain_parameters(),
                  'noise_parameters': default_noise_parameters(),
                  'spp_parameters': default_spp_parameters(),
                  'snr_parameters': default_snr_parameters()}

    return parameters

def process_frame(sframe, parameters):

    #0 windowing and stft
    #1 rough noise estimation

    #2 rough a priori and posteri snr estimation
    gamma, ksi =  compute_snr(sframe, parameters['noise_prev'], parameters['prev_ksi'])
    #3 rough noise estimation
    #noise = noise(parameters['noise_parameters'])
    #4 speech presence prabability estimation
    spp = spp(parameters['spp_parameters'])
    #3 precise noise estimation
    noise = noise_estimation(sframe, parameters['noise_parameters'])
    #5 a priori and posteri snr estimation
    gamma, ksi = compute_snr(sframe, noise, parameters['prev_ksi'])
    #6 gain
    gain = suppression_gain(parameters['gain_parameters'], len(sframe), method='logmmse')

    sfinal = sframe * gain * spp

    return sfinal
