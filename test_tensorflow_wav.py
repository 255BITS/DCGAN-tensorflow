
# Create sample wav file from util
# Ensure correct dimensionality
# should be [w, h, z] where w=441, h=?(0.25 seconds?), and z=2(complex, real)

# Sanity test load and save

import tensorflow as tf
import numpy as np
from numpy.fft import fft, ifft


from tensorflow_wav import get_wav, save_wav, tensorflow_fft_graph, tensorflow_ifft_graph

wav_path="input.wav"
wav_size=64

with tf.Session() as sess:
    wav= get_wav(wav_path)
    raw_data = tf.placeholder(tf.complex64, [len(wav['data'])])

    print("WAV IS", wav)
    noop = tensorflow_fft_graph(tensorflow_ifft_graph(raw_data))

    wav['data'] = sess.run(noop, {raw_data: wav['data']})
    #wav['data'] = fft(ifft(wav['data']))
    wav['data'] = np.array(wav['data'], dtype=np.int16)
    res= save_wav(wav, "sanity.wav")
