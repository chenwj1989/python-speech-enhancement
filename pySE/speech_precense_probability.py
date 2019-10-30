#!/usr/bin/python

from __future__ import division
import numpy as np
from scipy.signal import lfilter

# hard-decision: Soon et al.
# qk_old: (1,n)
def sap_priori_hard_decision(qk_old, gamma, ksi, c = 0.1):

    p = np.exp(-ksi) * np.i0(2 * np.sqrt(gamma*ksi))
    b = np.ones(p.shape)
    b[p > 1] = 0
    qk = c * b + (1-c) * qk_old
    return qk

# soft-decision: Soon et al.
#qk_old: (1,n)
def sap_priori_soft_decision(qk_old, gamma, ksi, c = 0.1):

    p0 = 1 / (1 + np.exp(-ksi) * np.i0(2 * np.sqrt(gamma*ksi)) )
    p0[p0 > 1] = 1
    qk = c * p0 + (1-c) * qk_old
    return qk

# Malah et al. (1999)
# qk_old: (1,n)
def sap_priori_vad(qk_old, gamma, ksi, c = 0.95, gamma_th = 0.8):
    idx = int(np.floor( len(gamma)/2 ))
    if np.mean(gamma[1:idx]) > 2.4 :#VAD
        I = np.ones(gamma.shape)
        index = np.where(gamma > gamma_th)
        I[index] = 0
        qk = c * qk_old + (1 - c) * I
    else:
        qk = qk_old
    return qk

# Cohen (2002)
def smoothing (x,N):
    Xlen = len(x)
    win = np.hanning(2*N+1)
    win1 = win[0:N+1]
    win2 = win[N+1:2*N+1]

    y1 = lfilter(np.flipud(win1),1,x)

    x2 = np.zeros(Xlen)
    x2[0:Xlen-N] = x[N:Xlen]

    y2 = lfilter(np.flipud(win2), 1, x2)

    y = (y1+y2)/np.linalg.norm(win,2)
    return y

# qk_old: (1,n)
def sap_priori_cohen(qk_old, gamma, ksi, beta = 0.7):

    qLen = len(qk_old)
    len2 = int(qLen / 2 + 1)
    zetak = np.zeros(len2)
    zeta_fr_old = 1000
    z_peak = 0

    zetak = beta * zetak + (1 - beta) * ksi[0:len2]

    z_min = 0.1
    z_max = 0.3162
    C = np.log10(z_max / z_min)
    zp_min = 1
    zp_max = 10
    zeta_local = smoothing(zetak, 1)
    zeta_global = smoothing(zetak, 15)

    Plocal = np.zeros(len2) # estimate P_local
    imax = np.where(zeta_local > z_max)
    Plocal[imax] = 1
    id1 = zeta_local > z_min
    id2 = zeta_local < z_max
    id = id1 & id2
    ibet = np.where(id)
    Plocal[ibet] = np.log10(zeta_local[ibet] / z_min) / C

    Pglob = np.zeros(len2) # estimate P_global
    imax = np.where(zeta_global > z_max)
    Pglob[imax] = 1
    id1 = zeta_global > z_min
    id2 = zeta_global < z_max
    id = id1 & id2
    ibet = np.where(id)
    Pglob[ibet] = np.log10(zeta_global[ibet] / z_min) / C

    zeta_fr = np.mean(zetak) #estimate Pframe
    if zeta_fr > z_min:
        if zeta_fr > zeta_fr_old:
            Pframe = 1
            z_peak = min(max(zeta_fr, zp_min), zp_max)
        else:
            if zeta_fr <= z_peak * z_min:
                Pframe=0
            elif zeta_fr >= z_peak * z_max:
                Pframe = 1
            else:
                Pframe = np.log10(zeta_fr / z_peak / z_min) / C
    else:
        Pframe = 0

    zeta_fr_old = zeta_fr
    qk2 = 1 - Plocal * Pglob * Pframe #estimate SAP

    qk2 = np.minimum(0.95, qk2)
    qk = np.append(qk2, np.flipud(qk2[1: len2-1]))

    return qk

def sap_priori(qk_old, parameters):

    ksi = parameters['ksi']
    gamma = parameters['gamma']
    method = parameters['sap_priori_method']
    c = parameters['sap_priori_c']

    if method == 'hard-decision':
       qk = sap_priori_hard_decision(qk_old, gamma, ksi, c)

    elif method == 'soft-decision':
       qk = sap_priori_soft_decision(qk_old, gamma, ksi, c)

    elif method == 'vad':
       gamma_th = parameters['sap_priori_gamma_th']
       qk = sap_priori_vad(qk_old, gamma, ksi, c, gamma_th)

    elif method == 'cohen':
       qk = sap_priori_cohen(qk_old, gamma, ksi, c)

    else:
        qk = qk_old

    return qk

def spp(qk_old, parameters):
    ksi = parameters['ksi']
    gamma = parameters['gamma']
    A = ksi / (1 + ksi)
    vk = A * gamma
    qk = sap_priori(qk_old, parameters) #qk: a-priori speech absence probability
    pSPP = (1 - qk)/ (1 - qk + qk * (1 + ksi)* np.exp(-1*vk)) # P(H1 | Yk)

    return pSPP


def sap(qk_old, parameters):
    ksi = parameters['ksi']
    gamma = parameters['gamma']
    A = ksi / (1 + ksi)
    vk = A * gamma
    qk = sap_priori(qk_old, parameters) #qk: a-priori speech absence probability
    pSPP = 1 - (1 - qk)/ (1 - qk + qk * (1 + ksi)* np.exp(-vk)) # P(H1 | Yk)

    return pSPP

def default_spp_parameters():
    parameters = {'ksi': 0, 'gamma': 0, 'sap_priori_method': 'soft-decision',
                  'sap_priori_c': 0.1, 'sap_priori_gamma_th': 0.8}
    return parameters

