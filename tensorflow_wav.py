import wave
import numpy as np
import tensorflow as tf
from scipy.io.wavfile import read, write
import scipy
import ops
import math

FRAME_SIZE=(64/2048.)
HOP=(2048-64)/(2048*64)

def stft(input, fs, framesz, hop):
    input = tf.reshape(input, [-1])
    framesamp = int(framesz*fs)
    hopsamp = int(hop*fs)
    #print('hopsamp is', hopsamp, hop, fs)
    #print('framesamp is', framesamp)
    #print("elemes", input.get_shape()[-1]-framesamp, (int(input.get_shape()[-1])-framesamp)/hopsamp)
    w = scipy.hanning(framesamp)

    def do_fft(w, input, i, n):
        #print("adding fft node for ", i, framesamp)
        slice = tf.slice(input, [i], [framesamp])
        #slice = fft(slice*w)
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
        res = tf.reshape(res, [-1])
        #res = ifft(res)
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
    processed = np.array(in_wav['data'], dtype=np.int16)
    # process ifft in tf
    wav.writeframes(processed)


def decompose(input, rank=3):
    #real,imag = tf.unpack(input, 2)
    #tf.unpack(input, 2)#
    #real = tf.slice(input, [0,0,0,0,0], [-1,-1,-1,-1,1])
    #imag = tf.slice(input, [0,0,0,0,1], [-1,-1,-1,-1,1])
    real, imag = tf.split(rank, 2, input)
    complex = tf.complex(real, imag)
    return complex
def compose(input, rank=3):
    real = tf.real(input)
    imag = tf.imag(input)
    return tf.concat(rank, [real, imag])

def fft(input):
    return tf.fft(input)
    #shape = input.get_shape()
    #output = tf.fft2d(input)

    #n=int(tf.reshape(input, [-1]).get_shape()[0])
    #with tf.variable_scope('fft', reuse=True):
    #    stored_n = tf.get_variable("fft_n", [1], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
    #    print('shape is', n)
    #    tf.assign(stored_n, [n])
    #    output=compose(output, rank=0)
    #    output = output * (1.0/n)
    #    output=decompose(output, rank=0)
    #    tf.reshape(output, shape)
    #    return output

def ifft(input):
    return tf.ifft(input)
    #return input#tf.fft2d(input)
    #shape = input.get_shape()
    #output = tf.ifft2d(input)

    #with tf.variable_scope('fft', reuse=True):
    #    stored_n = tf.get_variable("fft_n", [1], dtype=tf.float32)

    #    output=compose(output, rank=0)
    #    output = tf.div(output, tf.div(1.0,tf.sqrt(stored_n)))
    #    output=decompose(output, rank=0)
    #    tf.reshape(output, shape)
    #    return output

def encode(input,bitrate=2048):
    output = input

    #with tf.variable_scope('fft', reuse=None):
    #    stored_n = tf.get_variable("fft_n", [1], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
    #results = []
    #print("SHAPE IS", input.get_shape())
    #output = tf.reshape(output, [-1, bitrate])
    #for i in range(int(input.get_shape()[0])):
    #    print("Setting up sftf layer ", i)
    #    result = tf.slice(output, [i, 0], [1, -1])
    #    result = stft(result,bitrate,FRAME_SIZE, HOP)
    #    #result = tf.reshape(result, [1,64,64,1])
    #    results += [result]

    #output = tf.concat(0, results)
    output = tf.reshape(output, [-1, 32,64,1])
    output = compose(output)
    return output
def decode(input, bitrate=2048):
    output = input
    output = decompose(output)
    output = tf.reshape(output, [-1, 32,64])
    #results = []

    #for i in range(input.get_shape()[0]):
    #    print("stft decode layer", i)
    #    result = tf.slice(output, [i, 0, 0], [1, -1, -1])
    #    result = tf.reshape(result, [64,64]) 
    #    result = istft(result, bitrate, HOP)
    #    results += [tf.reshape(result, [-1])]

    #output = tf.concat(0, results)
            
    #output = tf.reshape(output, [-1])
    #output = tf.ifft(output)
    #print(output.get_shape())
    return output

def scale_up(input):
    with tf.variable_scope("scale"):

        real = tf.nn.tanh(input)
        sign = tf.sign(real)

        #w = tf.get_variable('scale_w', [1], initializer=tf.constant_initializer(10))
        return tf.exp(tf.abs(11.09*real))*sign
