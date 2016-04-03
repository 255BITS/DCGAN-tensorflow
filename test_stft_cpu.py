import scipy, pylab
import scipy.signal
import numpy as np
import tensorflow_wav
from math import sqrt

def fft(x):
    n = x.shape[0]
    return scipy.fft(x)


def stft(x, fftsize=126, overlap=4):   
    hop = fftsize // overlap
    w = scipy.signal.tukey(fftsize+1)[:-1]      # better reconstruction with this trick +1)[:-1]  
    return np.array([np.fft.rfft(w*x[i:i+fftsize]) for i in range(0, len(x)-fftsize, hop)])

def istft(X, overlap=4):   
    fftsize=(X.shape[1]-1)*2
    hop = fftsize // overlap
    w = scipy.signal.tukey(fftsize+1)[:-1]
    x = scipy.zeros(X.shape[0]*hop)
    wsum = scipy.zeros(X.shape[0]*hop) 
    for n,i in enumerate(range(0, len(x)-fftsize, hop)): 
        x[i:i+fftsize] += scipy.real(np.fft.irfft(X[n])) * w   # overlap-add
        wsum[i:i+fftsize] += w ** 2.
    pos = wsum != 0
    x[pos] /= wsum[pos]
    return x

#def stft(x, fs, framesz, hop):
#    #print("STFT got", x, fs, framesz, hop)
#    framesamp = int(framesz*fs)
#    hopsamp = int(hop*fs)
#    w = scipy.signal.tukey(framesamp)
#    def do_fft(w,x,i,framesamp):
#        #print("Running FFT for ", i, framesamp)
#        return fft(w*x[i:i+framesamp])
#    X = scipy.array([do_fft(w,x,i,framesamp) 
#                     for i in range(0, len(x)-framesamp, hopsamp)])
#    #print("X SHAPE IS", len(X), len(X[0]))
#    return X
#
#def ifft(x):
#    n = x.shape[0]
#    return scipy.ifft(x)
#def istft(X, fs, T, hop):
#    x = scipy.zeros(T*fs)
#    framesamp = X.shape[1]
#    print('fsamp',framesamp)
#    hopsamp = int(hop*fs)
#    for n,i in enumerate(range(0, len(x)-framesamp, hopsamp)):
#        if(n>=X.shape[0]): 
#            break
#        #print("setting i to i+framesamp", n, i, framesamp, i+framesamp, len(X), len(x))
#        x[i:i+framesamp] += scipy.real(ifft(X[n]))
#    #print(x.shape)
#    return x
#
#

if __name__ == '__main__':
    wav_path="input.wav"
    wav= tensorflow_wav.get_wav(wav_path)
    fs=44100
    T=20
    data = wav['data'][:fs*T, 0]
    data_right = wav['data'][:fs*T, 1]

    print("data, max, mean, stddev",data.min(), data.max(), np.mean(data), np.std(data))
    #data['sampwidth']
    framesz=1
    print("fs is ", fs)
    print("Framesz is", framesz)
    hop=0.25
    print("Hop is", hop)
    framesamp = int(framesz*fs)
    hopsamp = int(hop*fs)
    print("framesamp is", framesamp)
    print("hopsamp is", hopsamp, hop, fs)
    print('size is',data-framesamp, 'stride', hopsamp)

    s = stft(data)
    s_right = stft(data_right)
    print(np.shape(s))

    si = istft(s)
    si_right = istft(s_right)
    print("simin, max, mean, stddev",si.min(), si.max(), np.mean(si), np.std(si))
    #print('wavdata is ', len(wav['data']))

    print(np.shape(si))
    si = np.reshape(si, [-1, 1])
    si_right = np.reshape(si_right, [-1, 1])
    wav['data'] = np.array(np.concatenate( [si, si_right], 1))
    print('min/max', s.min(), s.max())
    #print(wav)
    #print('data is ',len(data))
    #print("s is ", len(s))
    res= tensorflow_wav.save_wav(wav, "stft_cpu_sanity.wav")
    print("Wrote to stft_cpu_sanity.wav")
