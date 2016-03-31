import scipy, pylab
import numpy as np
import tensorflow_wav
from math import sqrt

def fft(x):
    n = x.shape[0]
    return scipy.fft(x)
def stft(x, fs, framesz, hop):
    #print("STFT got", x, fs, framesz, hop)
    framesamp = int(framesz*fs)
    hopsamp = int(hop*fs)
    w = scipy.hanning(framesamp)
    def do_fft(w,x,i,framesamp):
        #print("Running FFT for ", i, framesamp)
        return fft(w*x[i:i+framesamp])
    X = scipy.array([do_fft(w,x,i,framesamp) 
                     for i in range(0, len(x)-framesamp, hopsamp)])
    #print("X SHAPE IS", len(X), len(X[0]))
    return X

def ifft(x):
    n = x.shape[0]
    return scipy.ifft(x)
def istft(X, fs, T, hop):
    x = scipy.zeros(T*fs)
    framesamp = X.shape[1]
    print('fsamp',framesamp)
    hopsamp = int(hop*fs)
    for n,i in enumerate(range(0, len(x)-framesamp, hopsamp)):
        if(n>=X.shape[0]): 
            break
        #print("setting i to i+framesamp", n, i, framesamp, i+framesamp, len(X), len(x))
        x[i:i+framesamp] += scipy.real(ifft(X[n]))
    #print(x.shape)
    return x



if __name__ == '__main__':
    wav_path="input.wav"
    wav= tensorflow_wav.get_wav(wav_path)
    fs=2048
    T=1
    data = wav['data'][:fs*T]
    print("data, max, mean, stddev",data.min(), data.max(), np.mean(data), np.std(data))
    #data['sampwidth']
    framesz=(64./2048)
    print("fs is ", fs)
    print("Framesz is", framesz)
    hop=1./64
    print("Hop is", hop)
    framesamp = int(framesz*fs)
    hopsamp = int(hop*fs)
    print("framesamp is", framesamp)
    print("hopsamp is", hopsamp, hop, fs)
    print('size is',data-framesamp, 'stride', hopsamp)

    s = stft(data, fs, framesz, hop)
    #print(s)

    T=1
    si = s
    si = istft(s, fs,T, hop)
    print("simin, max, mean, stddev",si.min(), si.max(), np.mean(si), np.std(si))
    #print('wavdata is ', len(wav['data']))

    #print(si)
    wav['data']=si
    print('min/max', s.min(), s.max())
    #print(wav)
    #print('data is ',len(data))
    #print("s is ", len(s))
    res= tensorflow_wav.save_wav(wav, "stft_cpu_sanity.wav")
    print("Wrote to stft_cpu_sanity.wav")
