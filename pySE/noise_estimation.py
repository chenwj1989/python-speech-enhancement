#!/usr/bin/python

from __future__ import division
import numpy as np
from scipy.special import *

def noise_vad(speech, noise_prev, parameters):
    mu = parameters['alpha']
    ksi = parameters['ksi']
    gamma = parameters['gamma']

    log_sigma_k= gamma * ksi/ (1+ ksi)- np.log(1+ ksi)
    vad_decision= np.mean( log_sigma_k)
    if (vad_decision < 0.15): # noise only frame found
        noise= mu * noise_prev+ (1- mu) * speech
    else:
        noise = noise_prev
    return noise

def noise_default(speech, noise_prev, parameters):
    alpha = parameters['alpha']
    p_speech = parameters['P_speech']
    alpha_d = alpha + (1 - alpha) * p_speech

    noise = alpha_d * noise_prev + (1 - alpha_d) * speech
    return noise

def default_noise_parameters():
    parameters = {'noise_method': 'vad', 'alpha': 0.98,  'P_speech': 0}
    return parameters

def noise_estimation(speech, noise_prev, parameters):
    method = parameters['noise_method']

    if method == "mcra":
        #nosie =  mcra(parameters)
        print("error, method not supported")
    elif method == "imcra":
        #nosie =  imcra(parameters)
        print("error, method not supported")
    elif method == "vad":
        nosie =  noise_vad(speech, noise_prev, parameters)
    else:
        nosie = noise_default(speech, noise_prev, parameters)

    return nosie