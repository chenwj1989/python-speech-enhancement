
import numpy as np
import soundfile as sf
from pesq import pesq

from pns.noise_suppressor import NoiseSuppressor

def test():

    # Prepare Data 
    clean_files = ["data/sp02.wav", "data/sp04.wav", "data/sp06.wav", "data/sp09.wav"]

    input_files = ["data/sp02_train_sn5.wav", 
                   "data/sp04_babble_sn10.wav", 
                   "data/sp06_babble_sn5.wav", 
                   "data/sp09_babble_sn10.wav"]

    output_files = ["data/sp02_train_sn5_processed.wav", 
                    "data/sp04_babble_sn10_processed.wav",
                    "data/sp06_babble_sn5_processed.wav", 
                    "data/sp09_babble_sn10_processed.wav"]
    
    for i in range(len(input_files)) :
        clean_file = clean_files[i]
        input_file = input_files[i]
        output_file = output_files[i]

        clean_wav, _  = sf.read(clean_file)
        noisy_wav, fs = sf.read(input_file)

        # Initialize
        noise_suppressor = NoiseSuppressor(fs)

        x = noisy_wav
        frame_size = noise_suppressor.get_frame_size()
        xfinal = np.zeros(len(x))

        # Start Processing
        k = 0
        while k + frame_size < len(x):
            frame = x[k : k + frame_size]
            xfinal[k : k + frame_size] =  noise_suppressor.process_frame(frame)
            k += frame_size

        # Save Results
        xfinal = xfinal / max(np.abs(xfinal))
        sf.write(output_file, xfinal, fs)

        # Performance Metrics
        print("")
        print(input_file)
        pesq_nb = pesq(ref=clean_wav, deg=noisy_wav, fs=fs, mode='nb')
        print("input pesq nb: ", "%.4f" % pesq_nb)
        pesq_nb = pesq(ref=clean_wav, deg=xfinal, fs=fs, mode='nb')
        print("output pesq nb: ", "%.4f" % pesq_nb)

        if fs > 8000:
            pesq_wb = pesq(ref=clean_wav, deg=noisy_wav, fs=fs, mode='wb')
            print("input pesq wb: ", "%.4f" % pesq_wb)
            pesq_wb = pesq(ref=clean_wav, deg=xfinal, fs=fs, mode='wb')
            print("output pesq wb: ", "%.4f" % pesq_wb)


def denoise_file(input_file, output_file):
    noisy_wav, fs = sf.read(input_file)
    channels = noisy_wav.shape[1] if noisy_wav.ndim > 1 else 1
    print("Input file: ", input_file)
    print("Sample rate: ", fs, "Hz")
    print("Num of channels: ", channels)
    print("Output file: ", output_file)

    if channels > 1 :
        xfinal = np.zeros(noisy_wav.shape)

        for ch in range(channels):
            noise_suppressor = NoiseSuppressor(fs)
            x = noisy_wav
            frame_size = noise_suppressor.get_frame_size()

            # Start Processing
            k = 0
            while k + frame_size < len(x):
                frame = x[k : k + frame_size, ch]
                xfinal[k : k + frame_size, ch] =  noise_suppressor.process_frame(frame)
                k += frame_size

            # Save Results
            xfinal[:, ch] = xfinal[:, ch] / max(np.abs(xfinal[:, ch]))

    else:
        # Initialize
        noise_suppressor = NoiseSuppressor(fs)
        x = noisy_wav
        frame_size = noise_suppressor.get_frame_size()
        xfinal = np.zeros(len(x))

        # Start Processing
        k = 0
        while k + frame_size < len(x):
            frame = x[k : k + frame_size]
            xfinal[k : k + frame_size] =  noise_suppressor.process_frame(frame)
            k += frame_size

        # Save Results
        xfinal = xfinal / max(np.abs(xfinal))
    
    sf.write(output_file, xfinal, fs)

if __name__=="__main__":
    denoise_file("data/sp02_train_sn5.wav", "data/sp02_train_sn5_processed.wav")
    # test()


