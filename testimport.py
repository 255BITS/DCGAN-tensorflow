
# Create sample wav file from util
# Ensure correct dimensionality
# should be [w, h, z] where w=441, h=?(0.25 seconds?), and z=2(complex, real)

# Sanity test load and save


from utils import *

import tensorflow as tf

with tf.Session() as sess:
    wav_path="output.wav"
    wav_size=64

    print(sess)
    wav= get_wav(wav_path, wav_size, is_crop=True)
    wav['data'] = sess.run(fft_transform, {raw_data: })

    print("WAV IS", wav)

    res= save_wav(wav, wav_size, "sanity.wav")
