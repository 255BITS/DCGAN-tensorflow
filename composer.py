import os
import numpy as np
import tensorflow as tf

from glob import glob
from model import DCGAN
from utils import pp, visualize, to_json
import tensorflow_wav


dataset="dcttime"
wav_size=64
is_crop=False
batch_size=128
checkpoint_dir="checkpoint"
bitrate=4096
song_seconds=20
song_step=1.0
z_dim=64

DIMENSIONS=1


with tf.Session() as sess:
    with tf.device('/cpu:0'):
      dcgan = DCGAN(sess, wav_size=wav_size, batch_size=batch_size,
        dataset_name=dataset, is_crop=is_crop, checkpoint_dir=checkpoint_dir)
      dcgan.load(checkpoint_dir)

      data = glob(os.path.join("./training", "*.wav"))
      sample_file = data[0]
      sample =tensorflow_wav.get_wav(sample_file)
      samplewav = sample.copy()

      full_audio = []
      second = 0.0
      batch_z = np.random.uniform(-1, 1, [batch_size, z_dim]) \
                .astype(np.float32)
      while(second < song_seconds):
        second += song_step
        batch_z[0]=float(second)/song_seconds
        audio = dcgan.sample(batch_z)
        print("Audio shape", np.shape(audio))

        audio = np.reshape(audio,[-1, DIMENSIONS])
        print("WAV shape", np.shape(audio))
        audio = audio.tolist()[:bitrate]
        samplewav['data']=audio
        converted = tensorflow_wav.convert_mlaudio_to_wav(samplewav, DIMENSIONS, wav_size)
        converted['data'] = np.reshape(converted['data'], [-1])
        full_audio += audio
        print("Full audio shape", np.shape(full_audio))

      samplewav['data']=full_audio
      print(samplewav['rate'], bitrate, len(full_audio))
      samplewav['data'] = np.reshape(samplewav['data'], [-1])

      filename = "./compositions/song.wav"
      tensorflow_wav.save_wav(samplewav, filename )



