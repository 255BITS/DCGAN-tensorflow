import os
import numpy as np
import tensorflow as tf

from glob import glob
from model import DCGAN
from utils import pp, visualize, to_json
import hwav
import tensorflow_wav


dataset="lstm-wavelet-2"
batch_size=256
checkpoint_dir="checkpoint"
bitrate=4096*2
z_dim=64

LENGTH=22

COUNT=664112//batch_size

with tf.Session() as sess:
    with tf.device('/cpu:0'):
      dcgan = DCGAN(sess, batch_size=batch_size,
        dataset_name=dataset, checkpoint_dir=checkpoint_dir)
      dcgan.load(checkpoint_dir)

      data = glob(os.path.join("./training", "*.mlaudio"))
      sample_file = data[0]
      sample =tensorflow_wav.get_pre(sample_file)
      print(sample_file, 'sample')
      samplewav = sample.copy()

      full_audio = []

      i=0
      leaves=[]
      leaves_right=[]
      while(i < COUNT):
        batch_z = np.random.uniform(-1, 1, [batch_size, z_dim]) \
                    .astype(np.float32)
        audio = dcgan.sample()

        leaves.append(audio[0::2])
        #print("Stats min/max/mean/stddev", np.min(audio), np.max(audio), np.mean(audio), np.std(audio))
        leaves_right.append(audio[1::2])
        if(i % 1000 == 0):
            print(i)
        i+=1
      print("tree, tr", np.shape(leaves), np.shape(leaves_right), COUNT*batch_size//2, LENGTH)
      leaves = np.reshape(leaves, [COUNT*batch_size//2, LENGTH])
      leaves_right = np.reshape(leaves_right, [COUNT*batch_size//2, LENGTH])
      print("tree, tr", np.shape(leaves), np.shape(leaves_right))
      tree = hwav.reconstruct_tree(leaves)
      tree_right = hwav.reconstruct_tree(leaves_right)
      samplewav['wavdec']=[tree, tree_right]


      converted = tensorflow_wav.convert_mlaudio_to_wav(samplewav)

      print('converted', np.shape(converted['wavdec']), np.min(batch_wavs), np.max(batch_wavs))

      filename = "./compositions/song.wav"
      tensorflow_wav.save_wav(converted, filename )



