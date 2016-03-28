
# Create sample wav file from util
# Ensure correct dimensionality
# should be [w, h, z] where w=441, h=?(0.25 seconds?), and z=2(complex, real)

# Sanity test load and save

import tensorflow as tf
import numpy as np
from numpy.fft import fft, ifft


from tensorflow_wav import get_wav, save_wav, encode, decode

wav_path="input.wav"
wav_size=64

with tf.Session() as sess:
    wav= get_wav(wav_path)
    print('data is', wav['data'])
    print(wav)
    fs = 2048
    T=10
    raw_data = tf.placeholder(tf.complex64, [1,fs*T])

    #print("WAV IS", wav['data'].tolist())
    data = wav['data'][:int(fs*T)]
    data = data.reshape([1,int(fs*T)])
    #data = tf.reshape(raw_data[:64*64*64], [-1])
    encoded = encode(raw_data, bitrate=fs)
    #print(encoded)
    decoded = decode(encoded, bitrate=fs)

    noop = decoded
    #noop = encoded
    wav['data'] = sess.run(noop, {raw_data: data})
    #print('data is now', wav['data'])
    wav['data'] =wav['data']
    res= save_wav(wav, "sanity.wav")
