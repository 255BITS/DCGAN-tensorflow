import wave
import numpy as np
import tensorflow as tf
from scipy.io.wavfile import read, write

# Returns the file object in complex64
def get_wav(path):

    wav = wave.open(path, 'rb')
    _, data = read(path)
    results={}
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
    processed = np.array(in_wav['data'])
    # process ifft in tf
    wav.writeframes(processed)



def decompose(input):
    #real,imag = tf.unpack(input, 2)
    #tf.unpack(input, 2)#
    #real = tf.slice(input, [0,0,0,0,0], [-1,-1,-1,-1,1])
    #imag = tf.slice(input, [0,0,0,0,1], [-1,-1,-1,-1,1])
    real, imag = tf.split(4, 2, input)
    complex = tf.complex(real, imag)
    return complex
def compose(input):
    real = tf.real(input)
    imag = tf.imag(input)
    return tf.concat(4, [real, imag])
def encode(input, inner_shape=[-1, 64,64,64,1], shape=[-1, 64,64,64,2]):
    output = input
    output = tf.reshape(output, [-1])
    output = tf.fft(output)
    output = tf.reshape(output, inner_shape)
    output = compose(output)
    output = tf.reshape(output, shape)
    return output
def decode(input, shape):
    output = input
    output = decompose(output)
    output = tf.reshape(output, [-1])
    output = tf.ifft(output)
    print(output.get_shape())
    output = tf.reshape(output, shape)
    return output

