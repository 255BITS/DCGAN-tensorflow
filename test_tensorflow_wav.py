
# Create sample wav file from util
# Ensure correct dimensionality
# should be [w, h, z] where w=441, h=?(0.25 seconds?), and z=2(complex, real)

# Sanity test load and save

import tensorflow as tf
import numpy as np
from numpy.fft import fft, ifft
from features import mfcc


from tensorflow_wav import get_wav, save_wav, encode, decode

wav_path="input.wav"
wav_size=64

with tf.Session() as sess:
    wav= get_wav(wav_path)
    print('data is', wav['data'])
    raw_data = tf.placeholder(tf.complex64, [len(wav['data'])])

    print("WAV IS", wav['data'].tolist())
    data = tf.reshape(raw_data[:64*64], [-1,64,64,1])
    #data = tf.reshape(raw_data[:64*64*64], [-1])
    print("calling encoded")
    encoded = encode(data)
    print(encoded)
    decoded = decode(encoded )

    noop = decoded
    #noop = encoded
    print(len(wav['data']))
    wav['data'] = sess.run(noop, {raw_data: wav['data']})
    print('data is now', wav['data'])
    flattened_item = np.array(wav['data']).flatten()
    mfccd = mfcc(flattened_item, wav['rate'], 0.025, 0.015625)
    print("flat", np.shape(flattened_item), np.shape(mfccd))
    #wav['data'] = fft(ifft(wav['data']))
    wav['data'] = np.array(wav['data'])
    res= save_wav(wav, "sanity.wav")
