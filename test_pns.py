
import numpy as np
import soundfile as sf
from pesq import pesq

from pns.noise_suppressor import NoiseSuppressor


def main():

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
        x = noisy_wav
        frame_size = 160
        fft_size = 512
        overlap_size = 352
        num_frames = int(np.floor(len(x) / frame_size) - np.floor(fft_size / frame_size))
        xfinal = np.zeros(num_frames * frame_size)

        noise_suppressor = NoiseSuppressor(fs, frame_size, fft_size, overlap_size)

        # Start Processing
        for i in range(num_frames):
            k = range(i*frame_size, (i + 1)*frame_size)
            frame = x[k]
            xfinal[k] =  noise_suppressor.process_frame(frame)

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

if __name__=="__main__":
    main()


