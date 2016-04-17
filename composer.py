import os
import numpy as np
import tensorflow as tf
import scipy

from glob import glob
from model import DCGAN
from utils import pp, visualize, to_json
import hwav
import tensorflow_wav


dataset="drums"
batch_size=8
checkpoint_dir="checkpoint"
bitrate=4096*2
z_dim=64

LENGTH=20
Y_DIM=512

COUNT=131072//(batch_size*Y_DIM/2.0)

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
        t = dcgan.coordinates(dcgan.t_dim)
        position = i / COUNT
        stepsize = 1 / COUNT
        t = np.array(t, dtype=np.float32)
        t *= 0.5*stepsize
        t += position
        t *= 20
        scale = 3
        z = np.zeros([dcgan.batch_size, dcgan.z_dim])
        z =  (np.random.uniform(-1,1.0,(dcgan.batch_size, dcgan.z_dim))*scale)
        #z[:, i] = 3
        #z[:, 0] = 1
        print("Z is ", z)
        #z[:, :(i-1)] = -3
        print(i)
        audio = dcgan.sample(t,z)
        audio = np.reshape(audio, (-1, LENGTH))
        print("shape is", np.shape(audio))

        leaves.append(audio[0::2])
        #print("Stats min/max/mean/stddev", np.min(audio), np.max(audio), np.mean(audio), np.std(audio))
        leaves_right.append(audio[1::2])
        if(i % 1000 == 0):
            print(i)
        i+=1
      print("tree, tr", np.shape(leaves), np.shape(leaves_right), COUNT*batch_size//2, LENGTH)
      #scipy.misc.imsave("visualize/output-t-"+str(i)+".png", leaves[:1])
      leaves = np.reshape(leaves, [-1, LENGTH])
      leaves_right = np.reshape(leaves_right, [-1, LENGTH])
      print("tree, tr", np.shape(leaves), np.shape(leaves_right))
      scipy.misc.imsave("visualize/output-"+str(i)+".png", leaves[:60])
      tree = hwav.reconstruct_tree(leaves)
      tree_right = hwav.reconstruct_tree(leaves_right)
      samplewav['wavdec']=[tree, tree_right]


      converted = tensorflow_wav.convert_mlaudio_to_wav(samplewav)

      batch_wavs = converted['wavdec']
      #print('converted', np.shape(converted['wavdec']), np.min(batch_wavs), np.max(batch_wavs))

      filename = "./compositions/song.wav"
      tensorflow_wav.save_wav(converted, filename )



