import os
import numpy as np
import tensorflow as tf

from glob import glob
from model import DCGAN
from utils import pp, visualize, to_json
import tensorflow_wav


dataset="video-game2"
wav_size=64
is_crop=False
batch_size=150
checkpoint_dir="checkpoint"
bitrate=4096
song_seconds=10
song_step=1.0


with tf.Session() as sess:
    with tf.device('/cpu:0'):
      dcgan = DCGAN(sess, wav_size=wav_size, batch_size=batch_size,
        dataset_name=dataset, is_crop=is_crop, checkpoint_dir=checkpoint_dir)
      dcgan.load(checkpoint_dir)

      data = glob(os.path.join("./training", "*.wav"))
      sample_file = data[0]
      sample =tensorflow_wav.get_wav(sample_file)

      full_audio = []
      second = 0.0
      while(second < song_seconds):
        batch_z = np.random.uniform(-1, 1, [batch_size, z_dim]) \
                    .astype(np.float32)
        second += song_step
        batch_z[0]=float(second)/song_seconds
        audio = dcgan.sample()
        print("Audio shape", np.shape(audio))

        audio = np.reshape(audio,[-1, 2])
        print("WAV shape", np.shape(audio[:,0]))
        full_audio += audio[:,0].tolist()
        print("Full audio shape", np.shape(full_audio))

      samplewav = sample.copy()
      samplewav
      samplewav['data']=full_audio
      print("samplewav shape", np.shape(samplewav['data']))

      filename = "./compositions/song.wav"
      tensorflow_wav.save_wav(samplewav, filename )



