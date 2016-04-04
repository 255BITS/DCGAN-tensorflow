import os
import numpy as np
import tensorflow as tf

from glob import glob
from model import DCGAN
from utils import pp, visualize, to_json
import tensorflow_wav


dataset="vg-dwt-lstm4"
wav_size=64
is_crop=False
batch_size=128
checkpoint_dir="checkpoint"
bitrate=4096*2
song_seconds=1
song_step=1.0
z_dim=64

DIMENSIONS=4


with tf.Session() as sess:
    with tf.device('/cpu:0'):
      dcgan = DCGAN(sess, wav_size=wav_size, batch_size=batch_size,
        dataset_name=dataset, is_crop=is_crop, checkpoint_dir=checkpoint_dir)
      dcgan.load(checkpoint_dir)

      data = glob(os.path.join("./training", "*.mlaudio"))
      sample_file = data[0]
      sample =tensorflow_wav.get_pre(sample_file)
      print(sample_file, 'sample')
      samplewav = sample.copy()

      full_audio = []

      second = 0.0
      while(second < song_seconds):
        batch_z = np.random.uniform(-1, 1, [batch_size, z_dim]) \
                    .astype(np.float32)
        second += song_step
        #batch_z[0]=float(second)/song_seconds
        audio = dcgan.sample()
        print("Audio shape", np.shape(audio))

        #audio = np.swapaxes(audio, 1, 2)
        full_audio.append(audio)
      #  print("Full audio shape", np.shape(full_audio))
      #  audio = np.reshape(audio,[-1, DIMENSIONS])
      #  print("WAV shape", np.shape(audio))
      #  audio = audio.tolist()[:bitrate]
      full_audio = np.concatenate(full_audio, 0)
      print("FA shape", np.shape(full_audio))
      print("Stats min/max/mean/stddev", np.min(audio), np.max(audio), np.mean(audio), np.std(audio))

      batch_wavs = full_audio
      print('FA', np.min(batch_wavs), np.max(batch_wavs))
      full_audio = batch_wavs


      samplewav['data']=np.reshape(full_audio,[-1, 64, DIMENSIONS])
      converted = tensorflow_wav.convert_mlaudio_to_wav(samplewav)

      print('converted', np.shape(converted['data']), np.min(batch_wavs), np.max(batch_wavs))
      samplewav['data']=converted['data']
      print(samplewav['rate'], bitrate, len(full_audio))
      #samplewav['data'] = np.reshape(samplewav['data'], [-1])

      filename = "./compositions/song.wav"
      print("Saving with", samplewav)
      tensorflow_wav.save_wav(samplewav, filename )



