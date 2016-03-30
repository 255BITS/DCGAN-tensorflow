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

    wav = wave.open(path, 'wb')
    wav.setnchannels(in_wav['channels'])
    wav.setsampwidth(in_wav['sampwidth'])

    wav.setframerate(in_wav['framerate'])

    wav.setnframes(in_wav['nframes'])

    wav.setcomptype('NONE', 'processed')

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


def compose(input, rank=3):
    return input
    real = tf.real(input)
    imag = tf.imag(input)
    return tf.concat(rank, [real, imag])

def encode(input,bitrate=4096):
    output = input

    output = tf.reshape(output, [-1, 64,64,1])
    output = compose(output)
    return output

def scale_up(input):
    output = tf.nn.tanh(input)*(65535)
    return output

