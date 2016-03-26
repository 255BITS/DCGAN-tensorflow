import wave
import numpy as np
import tensorflow as tf
from scipy.io.wavfile import read, write

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
    processed = np.array(data)
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


def decompose(input):
    #real,imag = tf.unpack(input, 2)
    #tf.unpack(input, 2)#
    #real = tf.slice(input, [0,0,0,0,0], [-1,-1,-1,-1,1])
    #imag = tf.slice(input, [0,0,0,0,1], [-1,-1,-1,-1,1])
    return input
def compose(input):
    #real = tf.real(input)
    #imag = tf.imag(input)
    #return tf.concat(3, [real, imag])
    return input
def encode(input, inner_shape=[-1,64,64,1], shape=[-1,64,64,1]):
    output = input
    #output = tf.reshape(output, [-1, 64])
    output = tf.reshape(output, [-1])
    #output = tf.fft(output)
    #output = tf.fft2d(output)
    output = tf.reshape(output, inner_shape)
    output = compose(output)
    output = tf.reshape(output, shape)
    return output
def decode(input, inner_shape=[-1,64,64,1], shape=[-1,64,64,1]):
    output = input
    #output = tf.reshape(output, inner_shape)
    #output = decompose(output)
    #output = tf.reshape(output, [-1, 64])
    output = tf.reshape(output, [-1])
    #output = tf.ifft(output)
    #output = tf.ifft2d(output)
    return output

def scale_up(input):
    real = 65535*input#tf.pow((65535/2), real)
    return real
