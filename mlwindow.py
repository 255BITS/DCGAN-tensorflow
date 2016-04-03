def fft(x):
    n = x.shape[0]
    return scipy.fft(x)
def to_freq_form(x, fs, framesz, hop):
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
def to_pcm(X, fs, T, hop):
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


