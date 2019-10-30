#!/usr/bin/python

from __future__ import division
import numpy as np

def default_snr_parameters():
    parameters = {'gamma_max':float('inf'),  "ksi_min":0, 'dd_nr_snr': 0.98}
    return parameters

def compute_snr(sig_power, noise_power, prev_ksi, parameters):
    gamma_max = parameters["gamma_max"]
    ksi_min = parameters["ksi_min"]
    DD_PR_SNR = parameters["dd_nr_snr"]

    gamma = np.minimum(sig_power / (noise_power), gamma_max)
    ksi = DD_PR_SNR * prev_ksi + (1.0 - DD_PR_SNR) * np.maximum(gamma - 1, 0)
    ksi = np.maximum(ksi_min, ksi)
    return gamma, ksi


