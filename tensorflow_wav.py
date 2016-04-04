import wave
import numpy as np
import tensorflow as tf
from scipy.io.wavfile import read, write
import scipy
import scipy.signal
import ops
import math
import pickle
import mdct
import pywt

FRAME_SIZE=(64/2048)
HOP=(2048-64)/(2048*64)

def do_imdct(row):
    return mdct.imdct(row, len(row))


def do_istft(X, overlap=4):
    fftsize=(X.shape[1]-1)*2
    hop = fftsize // overlap
    w = scipy.signal.hamming(fftsize+1)[:-1]
    x = scipy.zeros(X.shape[0]*hop)
    wsum = scipy.zeros(X.shape[0]*hop) 
    for n,i in enumerate(range(0, len(x)-fftsize, hop)): 
        x[i:i+fftsize] += scipy.real(np.fft.irfft(X[n])) * w   # overlap-add
        wsum[i:i+fftsize] += w ** 2.
    pos = wsum != 0
    x[pos] /= wsum[pos]
    return scipy.real(x)


def do_idwt(rows):
    def idwt(elems):
        main = elems[:,0]
        detail = elems[:,1]
        return pywt.idwt(main, detail, 'db1')
    results = [idwt(rows[i][:]) for i in range(0, len(rows), 1)]
    return results

def convert_mlaudio_to_wav(mlaudio):
    data = np.array(mlaudio['data'])
    # We split the audio stream into 2, one for each speaker
    #if the dimensions change so will this function
    audio, audio_right = np.split(data, 2, 2)
    #print("Running idwi on data", data.shape)
    data = do_idwt(audio)
    data_right = do_idwt(audio_right)

    # combine left and right streams, wav uses [-1, channels] as the output format
    data = np.reshape(data, [-1, 1])
    data_right = np.reshape(data_right, [-1, 1])
    result =np.array(np.concatenate( [data, data_right], 1))
    mlaudio['data'] = result
    return mlaudio


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
    processed = np.array(data).astype(np.int16, copy=False)
    results['data']=processed
    return results

def save_wav(in_wav, path):

    print("Saving to ", path)
    wav = wave.open(path, 'wb')
    wav.setnchannels(in_wav['channels'])
    wav.setsampwidth(in_wav['sampwidth'])

    wav.setframerate(in_wav['framerate'])

    wav.setnframes(in_wav['nframes'])

    wav.setcomptype('NONE', 'processed')

    processed = np.array(in_wav['data'], dtype=np.int16)
    wav.writeframes(processed)

def save_pre(in_wav, path):
    f = open(path, "wb")
    try:
        pickle.dump(in_wav, f, pickle.HIGHEST_PROTOCOL)
        print("DUMPED")
    finally:
        f.close()
def get_pre(filename):
    f = open(filename, "rb")
    data = pickle.load(f)
    f.close()
    return data


def compose(input, rank=3):
    return input

def encode(input,bitrate=4096):
    output = input

    output = tf.reshape(output, [-1, 64,64,1])
    output = compose(output)
    return output

def ff_nn(input, name):
    with tf.variable_scope("ff_nn"):
        input_shape = input.get_shape() 
        input_dim = int(input.get_shape()[1])
        W = tf.get_variable(name+'w',[input_dim, input_dim], initializer=tf.random_normal_initializer(0, 0.02))

        # Initialize b to zero
        b = tf.get_variable(name+'b', [input_dim], initializer=tf.constant_initializer(0))

        output = tf.nn.tanh(tf.matmul(tf.reshape(input, [-1,input_dim]),W) + b)
        output = tf.reshape(output, input_shape)
        return output



def scale_up(input):
    with tf.variable_scope("scale"):

        output = tf.nn.tanh(input)
        w = w2 = 0.00002
        #w = tf.get_variable('g_scale_w', [1], dtype=tf.float32, initializer=tf.constant_initializer(0.00002))
        #w2 = tf.get_variable('g_scale_w2', [1], dtype=tf.float32, initializer=tf.constant_initializer(0.00002))
        l_main, l_det, r_main, r_det = tf.split(3, 4, output)
        l_main = l_main / w
        r_main = r_main / w
        l_det = l_det / w2
        r_det = r_det / w2
        return tf.concat(3, [l_main, l_det, r_main, r_det])

        raw_output, fft_real_output= tf.split(3, 2, output)
        sign = tf.sign(raw_output)

        #raw = tf.exp(tf.abs(10.8*raw_output))*sign
        raw = raw_output * 32768

        #new_fft = ff_nn(fft_output, 'fft')
        #complex = tf.complex(fft_real_output, fft_imag_output)
        #fft = complex / w
        fft = fft_real_output / w

        return tf.concat(3, [raw, fft])
