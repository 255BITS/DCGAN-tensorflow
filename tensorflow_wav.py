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


def tensorflow_fft_graph(input):
    output = tf.fft(input)
    return output
def tensorflow_ifft_graph(input):
    output = tf.ifft(input)
    return output

