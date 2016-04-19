import os
import numpy as np
import tensorflow as tf
import scipy

from glob import glob
from model import DCGAN
from utils import pp, visualize, to_json
import hwav
import tensorflow_wav


dataset="vg"
batch_size=128
checkpoint_dir="checkpoint"
z_dim=64

LENGTH=1024

COUNT=10

with tf.Session() as sess:
    with tf.device('/cpu:0'):
      dcgan = DCGAN(sess, batch_size=batch_size,
        dataset_name=dataset, checkpoint_dir=checkpoint_dir)
      dcgan.load(checkpoint_dir)

      data = glob(os.path.join("./training", "*.wav"))
      sample_file = data[0]
      sample =tensorflow_wav.get_wav(sample_file)
      print(sample_file, 'sample')
      samplewav = sample.copy()

      full_audio = []

      i=0
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
        scale = 4
        #z = np.zeros([dcgan.batch_size, dcgan.z_dim])
        z =  (np.random.uniform(-1,1.0,(dcgan.batch_size, dcgan.z_dim))*scale)
        
        #z = np.random.uniform(-1, 1, [batch_size, dcgan.z_dim]) \
        #                                    .astype(np.float32)
        # = np.random.normal(0,2,(dcgan.batch_size, dcgan.z_dim))
        #z = np.zeros_like(z)
        #z = np.ones_like(z)*-4
        #z[:, min(i, 63)] = 1
        #z[:, i] = 1
        #z[:, 0] = 5
        #z[:, 1] = 1
        print("Z is ", z)
        #z[:, :(i-1)] = -3
        print(i)
        audio = dcgan.sample(t,z)
        audio = np.reshape(audio, (-1, 1, LENGTH))
        audio = np.swapaxes(audio, 1, 2)
        full_audio.append(audio)
        print("shape is", np.shape(audio))

        if(i % 1000 == 0):
            print(i)
        i+=1
      #scipy.misc.imsave("visualize/output-t-"+str(i)+".png", leaves[:1])
      
      samplewav['data']=np.reshape(full_audio, [-1, 2])
      #print('converted', np.shape(converted['wavdec']), np.min(batch_wavs), np.max(batch_wavs))

      filename = "./compositions/song.wav"
      tensorflow_wav.save_wav(samplewav, filename )



