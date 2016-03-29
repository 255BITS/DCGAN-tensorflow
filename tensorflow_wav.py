import wave
import numpy as np
import tensorflow as tf
from scipy.io.wavfile import read, write
import scipy
import ops
import math
import pickle

FRAME_SIZE=(64/2048)
HOP=(2048-64)/(2048*64)


def stft(input, fs, framesz, hop):
    #input = tf.reshape(input, [-1])
    #return tf.fft(input)
    #return input
    #return tf.fft(input)
    input =tf.reshape(input,[64,64])
    framesamp = int(framesz*fs)
    hopsamp = int(hop*fs)
    #print('hopsamp is', hopsamp, hop, fs)
    #print('framesamp is', framesamp)
    #print("elemes", input.get_shape()[-1]-framesamp, (int(input.get_shape()[-1])-framesamp)/hopsamp)
    w = scipy.hanning(framesamp)

    def do_fft(w, input, i, n):
        #print("adding fft node for ", i, framesamp)
        slice = tf.slice(input, [i], [framesamp])
        slice = fft(slice*w)
        return slice
    X = [do_fft(w, input, i, i+framesamp)
                     for i in range(0, input.get_shape()[-1]-framesamp, hopsamp)]
    return tf.concat(0,X)

def istft(X, fs, hop):
    height  = int(X.get_shape()[1])
    width = int(X.get_shape()[0])
    T=1
    length = T*fs
    output = tf.zeros([fs*T], dtype='complex64')

    hopsamp = int(hop*fs)
    def do_ifft(X, n,i):
        #print("BUILDING SLICE", n,i)
        res = tf.slice(X, [n, 0], [1, height])
        res = ifft(tf.reshape(res, [-1]))
        pre = tf.zeros([i], dtype='complex64')
        post = tf.zeros([length-i-height], dtype='complex64')
        to_add = tf.concat(0, [pre, res*(1+0j), post])
        return tf.add(output, to_add)
    iterator = enumerate(range(0, length-height, hopsamp))
    for n,i in iterator:
        output = do_ifft(X,n,i)
    output= tf.reshape(output, [-1])
    print(output.get_shape())
    return output

# Returns the file object in complex64
def get_wav(path):

    wav = wave.open(path, 'rb')
    rate, data = read(path)
    results={}
    results['rate']=rate
    results['channels']=wav.getnchannels()
    results['sampwidth']=wav.getsampwidth()
    results['framerate']=wav.getframerate()
    results['nframes']=wav.getnframes()
    results['compname']=wav.getcompname()
    # process fft in tf
    processed = np.array(data).astype(np.complex64, copy=False)
    results['data']=processed
    return results

def save_wav(in_wav, path):

    wav = wave.open(path, 'wb')
    wav.setnchannels(in_wav['channels'])
    wav.setsampwidth(in_wav['sampwidth'])

    wav.setframerate(in_wav['framerate'])

    wav.setnframes(in_wav['nframes'])

    wav.setcomptype('NONE', 'processed')
    # process ifft in tf

    processed = np.array(in_wav['data'], dtype=np.int16)
    wav.writeframes(processed)

def save_stft(in_wav, path):
    f = open(path, "wb")
    try:
        pickle.dump(in_wav, f, pickle.HIGHEST_PROTOCOL)
        print("DUMPED")
    finally:
        f.close()
def get_stft(filename):
    f = open(filename, "rb")
    data = pickle.load(f)
    f.close()
    return data


def decompose(input, rank=3):
    real, imag = tf.split(rank, 2, input)
    complex = tf.complex(real, imag)
    return complex
def compose(input, rank=3):
    real = tf.real(input)
    imag = tf.imag(input)
    return tf.concat(rank, [real, imag])

def encode(input,bitrate=4096):
    output = input

    output = tf.reshape(output, [-1, 64,64,1])
    output = compose(output)
    return output

def scale_up(input):
    output = tf.nn.tanh(input)
    return decompose(input)*2e4

def build_fft_graph(input):
    return tf.ifft(input)

