import os
import numpy as np
import tensorflow as tf

from glob import glob
from model import DCGAN
from utils import pp, visualize, to_json
import tensorflow_wav


dataset="audio"
wav_size=64
is_crop=False
batch_size=10
checkpoint_dir="checkpoint"
bitrate=2048

with tf.Session() as sess:
    with tf.device('/cpu:0'):
        dcgan = DCGAN(sess, wav_size=wav_size, batch_size=batch_size,
                dataset_name=dataset, is_crop=is_crop, checkpoint_dir=checkpoint_dir)
        dcgan.load(checkpoint_dir)

        data = glob(os.path.join("./training", "*.wav"))
        sample_file = data[0]
        sample =tensorflow_wav.get_wav(sample_file)

        full_audio = []
        for i in range(120):
            audio = dcgan.sample()

            audio = np.reshape(audio,[-1])
            print("Audio shape", np.shape(audio))
            full_audio += audio[:bitrate].tolist()
            print("Full audio shape", np.shape(full_audio))

        samplewav = sample.copy()
        samplewav
        samplewav['data']=np.array(full_audio)
        print("samplewav shape", np.shape(samplewav['data']))

        filename = "./compositions/song.wav.stft"
        tensorflow_wav.save_stft(samplewav, filename )



