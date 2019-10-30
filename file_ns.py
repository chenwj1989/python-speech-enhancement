#!/usr/bin/python
from __future__ import division
# import

import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf

from pySE.ns_process import NoiseSuppression
from pySE.suppression_gain import ns_gain
from pySE.compute_snr import compute_snr
from pySE.noise_estimation import noise_estimation
from pySE.speech_precense_probability import spp

def main():
    input_file = "test/sp04_babble_sn10.wav"
    output_file = "test/out.wav"

    x, Srate = sf.read(input_file)

    # ====== set parameters ========
    interval = 0.02 #frame interval = 0.02s
    Slen = int(np.floor(interval * Srate))
    if Slen % 2 == 1:
        Slen = Slen + 1
    PERC = 50  #window overlap in percent of frame size
    len1 = int(np.floor(Slen * PERC / 100))
    len2 = int(Slen - len1)
    nFFT = 2 * Slen

    #Noise magnitude calculations - assuming that the first 6 frames is  noise/silence
    win = np.hanning(Slen)
    win = win * len2 / np.sum(win)
    noise_mean =np.zeros(nFFT)
    plt.specgram(x, NFFT=Slen, Fs=Srate, noverlap=len2, cmap='jet')
    plt.ylim((0, 4000))
    plt.show()

    j=1
    noise_frames = 6
    for m in range(1, noise_frames, 1):
        noise_mean = noise_mean + np.absolute(np.fft.fft(win * x[j:j + Slen], nFFT, axis=0))
        j=j + Slen
    noise_mu2 = (noise_mean / noise_frames) ** 2

    #=============================  User NoiseSuppression C=======================================================
    #initialize
    Nframes = int(np.floor(len(x) / len2) - np.floor(Slen / len2))
    xfinal = np.zeros(Nframes * len2)

    ns = NoiseSuppression('logmmse', 'vad', len2, Slen, nFFT)
    ns_parameters = ns.get_parameters()

    ns_parameters["noise_parameters"]['alpha'] = 0.98
    ns_parameters["spp_parameters"]['sap_priori_method'] = 'cohen'
    ns_parameters["snr_parameters"]['gamma_max'] = 40
    ns_parameters["snr_parameters"]['ksi_min'] = 10 ** (-25 / 10)
    ns_parameters["snr_parameters"]['dd_nr_snr'] = 0.98

    prev_ksi = np.ones(nFFT)
    ns.set_parameters(ns_parameters)
    ns.set_priori_noise(noise_mu2)
    ns.set_priori_snr(prev_ksi)

    #=============================  Start Processing =======================================================
    for k in range(0, Nframes*len2, len2):
        xfinal[k:k + len2] = ns.process_frame(x[k:k + len2])

    sf.write(output_file, xfinal, Srate)
    plt.figure(2)
    plt.specgram(xfinal, NFFT=Slen, Fs=Srate, noverlap=len2, cmap='jet')
    plt.ylim((0, 4000))
    plt.show()


if __name__=="__main__":
    main()


