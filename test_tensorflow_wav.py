
# Create sample wav file from util
# Ensure correct dimensionality
# should be [w, h, z] where w=441, h=?(0.25 seconds?), and z=2(complex, real)

# Sanity test load and save

import tensorflow as tf


from tensorflow_wav import get_wav, save_wav

wav_path="training/02-overworld-01.wav"
wav_size=64

with tf.Session() as sess:
    wav= get_wav(sess, wav_path)

    print("WAV IS", wav)

    res= save_wav(wav, "sanity.wav")
