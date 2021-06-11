# Python Speech Enhancement
A python library for speech enhancement.

![Noise Suppression Flow Diagram](https://wjchen.net/static/posts/ns_flow.png)

## Usage

The example test_pns.py shows how to do noise suppression on wav files. The python-pesq package should be installed in order to evaluate the output.
```
pip install pesq
python test_pns.py
```

Major steps of using the noise suppression library are shown below. The NoiseSuppressor processes audio data block by block.
```python
# Initialize
fs = 16000
noise_suppressor = NoiseSuppressor(fs)
frame_size = noise_suppressor.get_frame_size()

# Process
x = noisy_wav  
xfinal = np.zeros(len(x))

# Start Processing
k = 0
while k + frame_size < len(x):
    frame = x[k : k + frame_size]
    xfinal[k : k + frame_size] =  noise_suppressor.process_frame(frame)
    k += frame_size
```

## Features
- [x] STFT Analysis and Synthesis
- [x] Support sample rate 16000
- [x] IMCRA Noise Estimation, according to [Cohen’s implementation](https://israelcohen.com/software/)
- [x] OMLSA Suppression Gain, according to [Cohen’s implementation](https://israelcohen.com/software/)
- [x] Wiener Suppression Gain

- [ ] Support sample rate 8000, 32000, 44100, 48000
- [ ] MCRA Noise Estimation
- [ ] Histogram Noise Estimation

## Reference
- I. Cohen and B. Berdugo, Speech Enhancement for Non-Stationary Noise Environments, Signal Processing, Vol. 81, No. 11, Nov. 2001, pp. 2403-2418.
- I. Cohen, Noise Spectrum Estimation in Adverse Environments: Improved Minima Controlled Recursive Averaging, IEEE Trans. Speech and Audio Processing, Vol. 11, No. 5, Sep. 2003, pp. 466-475.
- Loizou, Philipos. (2007). Speech Enhancement: Theory and Practice. 10.1201/b14529. 