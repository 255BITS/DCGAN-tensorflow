from scipy.io.wavfile import read, write
from scipy.fftpack import rfft, irfft
from numpy.fft import hfft, ihfft, fft, ifft
import numpy as np
import math


def best_length_fft(n):
    return int(int(n/2)*2)
    #return math.pow(2, max(factorint(n)))
def loadfft(wavfile):
    rate, input = read(wavfile)
    max_size = len(input)#best_length_fft(len(input))
    print("input is", input)
    rfftx = fft(input)
    transformed_raw = np.array(rfftx)
    transformed = transformed_raw / transformed_raw.max(axis=0)
    sanity = ifft(rfftx)
    uintout = sanity.astype('int16')
    write("sanity-2.wav", 44100, uintout)

    return {
            "transformed": transformed,
            "raw": transformed_raw,
            "rate": rate
            }

def sanity(wavobj):
    transformed = wavobj['transformed']
    transformed_raw = wavobj['raw']
    rate = wavobj['rate']
    output = irfft(transformed)
    san_output = irfft(transformed)* transformed_raw.max(axis=0)

    write('sanity.wav', rate, np.array(san_output, dtype=np.int16))
    print("Sanity.wav written")

def savefft(wavfile, wavobj, filtered):
    transformed = wavobj['transformed']
    transformed_raw = wavobj['raw']
    rate = wavobj['rate']
    data = filtered * transformed_raw.max(axis=0)
    output = ifft(data)
    write(wavfile, rate, np.array(output, dtype=np.int16))

def loadfft2(wav):
    return loadfft(wav)

def savefft2(wav, wavobj, filtered):
    return savefft(wav, wavobj, filtered)
