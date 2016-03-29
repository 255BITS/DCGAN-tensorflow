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


def decode_sampler(input, bitrate=4096):
    output = input
    output = tf.reshape(output, [-1, 4096])
    results = []
    for i in range(int(output.get_shape()[0])):
        print("Setting up sftf layer ", i, output.get_shape())
        result = tf.slice(output, [i, 0], [1, -1])
        #result = tf.reshape(result, [64,64])
        result = tf.reshape(result, [-1])
        result = tf.ifft(result)
        #result = tf.reshape(result, [-1])
        #result = tf.reshape(result,[-1])
        results += [result]

    output = tf.concat(0, results)
    return output


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
    #return tf.ifft(input)
    return tf.fft2d(input)
    #shape = input.get_shape()
    #output = tf.ifft2d(input)

    #with tf.variable_scope('fft', reuse=True):
    #    stored_n = tf.get_variable("fft_n", [1], dtype=tf.float32)

    #    output=compose(output, rank=0)
    #    output = tf.div(output, tf.div(1.0,tf.sqrt(stored_n)))
    #    output=decompose(output, rank=0)
    #    tf.reshape(output, shape)
    #    return output

def encode(input,bitrate=4096):
    output = input

    #with tf.variable_scope('fft', reuse=None):
    #    stored_n = tf.get_variable("fft_n", [1], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
    #results = []
    #print("SHAPE IS", input.get_shape())
    #output = tf.fft(tf.reshape(output,[-1]))
    #output = tf.reshape(output, [-1, bitrate])
    #for i in range(int(input.get_shape()[0])):
    #    print("Setting up sftf layer ", i)
    #    result = tf.slice(output, [i, 0], [1, -1])
    #    result = stft(result,bitrate,FRAME_SIZE, HOP)
    #    result = tf.reshape(result, [1,64,64,1])
    #    results += [result]

    #output = tf.concat(0, results)
    output = tf.reshape(output, [-1, 64,64,1])
    output = compose(output)
    return output
def decode(input, bitrate=4096):
    output = input
    output = decompose(output)
    output = tf.reshape(output, [-1, 64,64])
    results = []

    for i in range(input.get_shape()[0]):
        print("stft decode layer", i)
        result = tf.slice(output, [i, 0, 0], [1, -1, -1])
        result = tf.reshape(result, [64,64]) 
        result = istft(result, bitrate, HOP)
        results += [tf.reshape(result, [-1])]

    output = tf.concat(0, results)
            
    #output = tf.reshape(output, [-1])
    #output = tf.ifft(output)
    #print(output.get_shape())
    return output

def scale_up(input):

    with tf.variable_scope('scale'):
        tf.get_variable_scope().reuse_variables()
        output = tf.nn.tanh(input)
        real, imag = tf.split(3, 2, output)
        #sign_real = tf.get_variable('sign_real', real.get_shape(), initializer=tf.constant_initializer(1))
        #sign_imag = tf.get_variable('sign_imag', imag.get_shape(), initializer=tf.constant_initializer(1))
        #tf.assign(sign_real, tf.sign(real))
        #tf.assign(sign_imag, tf.sign(imag))
        #imag_sign = tf.sign(imag)*1
        #real = tf.abs(1/tf.exp(real*(4.4*math.pi)))*sign_real#-min.real
        #imag = tf.abs(1/tf.exp(imag*(4.4*math.pi)))*sign_imag#-min.imag
        real = (tf.pow(real, 3.))*1e7#*1e7
        imag = (tf.pow(imag, 3.))*1e7#*1e7

        complex = tf.complex(real, imag)
        return tf.concat(3, [complex])
        #return 1/(tf.exp(output*4*math.pi))
    #min = 32000+12000j
    #max = 65000+32000j
    #max = 50000+20000j

    output = decompose(output)
    max = 1000000+1000000j
    return output*max#decompose(output*max)
    #max = 70000+30000j
    #max =  23000+0j

def build_fft_graph(input):
    return tf.ifft(input)

