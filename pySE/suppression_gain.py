#!/usr/bin/python

from __future__ import division
import numpy as np
import scipy.special as sp


def spec_sub_gain(parameters=None):
    """ calculate suppression gain by spectral subtraction method
    """
    ksi = parameters['ksi']
    gain = np.sqrt( ksi / (1+ ksi)) # gain function
    return gain


def wiener_gain(parameters=None):
    """ calculate suppression gain by wiener filtering
    """
    ksi = parameters['ksi']
    gain = ksi / (1+ ksi) # gain function
    return gain


def mmse_stsa_gain(parameters=None):
    """ calculate suppression gain by mmse-stsa(MMSE short-time spectral amplitude) method
    """
    gamma = parameters['gamma']
    ksi = parameters['ksi']

    vk = ksi * gamma / (1 + ksi)
    j0 = sp.j0(vk / 2)
    j1 = sp.j1(vk / 2)

    A = np.sqrt(np.pi * vk) / 2 / gamma
    B = (1 + vk) * j0 + vk * j1
    C = np.exp(-0.5 * vk)
    gain = A * B * C
    return gain


def mmse_sqr_gain(parameters=None):
    """ calculate suppression gain by MMSE magnitude squared method
    """
    gamma = parameters['gamma']
    ksi = parameters['ksi']
    vk = ksi * gamma / (1 + ksi)

    A = ksi / (1 + ksi)
    B = (1 + vk) / gamma
    gain = np.sqrt(A * B)
    return gain


def logmmse_gain(parameters=None):
    """ calculate suppression gain by MMSE log spectral amplitude method
    """
    gamma = parameters['gamma']
    ksi = parameters['ksi']
    A = ksi / (1 + ksi)
    vk = A * gamma
    ei_vk = 0.5 * sp.expn(1, vk)
    gain = A * np.exp(ei_vk)
    return gain


def ml_gain(parameters=None):
    """ calculate suppression gain by maximum likelihood method
    """
    ksi = parameters['ksi']
    gain = 0.5 + 0.5 * np.sqrt(ksi / (1.0 + ksi))
    return gain


def map_joint_gain(parameters=None):
    """ calculate suppression gain by MAP method combining spectra amplitude and phase
    """
    gamma = parameters['gamma']
    ksi = parameters['ksi']

    eps = 1e-6
    A = ksi**2 + 2*(1.0 + ksi)* ksi/ (gamma + eps)
    B = 2.0 *(1.0 + ksi)
    gain = (ksi + np.sqrt(A)) / B
    return gain


def map_sa_gain(parameters=None):
    """ calculate suppression gain by MAP method using spectra amplitude only
    """
    gamma = parameters['gamma']
    ksi = parameters['ksi']

    eps = 1e-6
    A = ksi**2 + (1.0 + ksi)* ksi/ (gamma + eps)
    B = 2.0 *(1.0 + ksi)
    gain = (ksi + np.sqrt(A)) / B
    return gain


def default_gain_parameters():
    parameters = { 'gain_method':'logmmse', 'gamma':0,  "ksi":0}
    return parameters


def ns_gain(parameters):
    """
    calculate suppression gain by different methods

    Args:
        parameters: suppression gain parameters
            parameters['method']: the method name
            parameters['gamma']: the posteriori snr on each frequency bin
            parameters['ksi']: the priori snr on each frequency bin

    Returns:
        suppression gain of each frequency bin
    """
    method = parameters['gain_method']

    if method == "spectral_subtraction":
        gain = spec_sub_gain(parameters)
    elif method == "wiener":
        gain = wiener_gain(parameters)
    elif method == "mmse" or  method == "mmse_stsa" :
        gain = mmse_stsa_gain(parameters)
    elif method == "mmse_sqr" :
        gain = mmse_sqr_gain(parameters)
    elif method == "logmmse":
        gain = logmmse_gain(parameters)
    elif method == "max_likelihood":
        gain = ml_gain(parameters)
    elif method == "map" or method == "map_joint" :
        gain = map_joint_gain(parameters)
    elif method == "map_sa":
        gain = map_sa_gain(parameters)
    else:
        print("error, method not supported")
        gain = np.ones(len(parameters['gamma']))

    return gain


