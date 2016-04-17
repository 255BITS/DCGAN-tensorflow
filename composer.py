import os
import numpy as np
import tensorflow as tf
import scipy

from glob import glob
from model import DCGAN
from utils import pp, visualize, to_json
import hwav
import tensorflow_wav


dataset="wnn"
batch_size=1024
checkpoint_dir="checkpoint"
bitrate=4096*2
z_dim=64

LENGTH=512

COUNT=4

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
        scale = 2
        z = np.zeros([dcgan.batch_size, dcgan.z_dim])
        z =  (np.random.uniform(-1,1.0,(dcgan.batch_size, dcgan.z_dim))*scale)
        #z[:, i] = 3
        #z[:, 0] = 1
        print("Z is ", z)
        #z[:, :(i-1)] = -3
        print(i)
        audio = dcgan.sample(t,z)
        audio = np.reshape(audio, (-1, 2, LENGTH))
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



