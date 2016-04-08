import os
import numpy as np
import tensorflow as tf
import scipy

from glob import glob
from model import DCGAN
from utils import pp, visualize, to_json
import hwav
import tensorflow_wav


dataset="wavelet-1"
batch_size=4096
checkpoint_dir="checkpoint"

LENGTH=20

COUNT=131072//batch_size

with tf.Session() as sess:
    with tf.device('/cpu:0'):
      dcgan = DCGAN(sess, batch_size=batch_size,
        dataset_name=dataset)
      dcgan.load(checkpoint_dir)

      data = glob(os.path.join("./training", "*.mlaudio"))
      sample_file = data[0]
      sample =tensorflow_wav.get_pre(sample_file)
      print(sample_file, 'sample')
      mlaudio = sample.copy()


      left, right = mlaudio['wavdec']
      data_left = hwav.leaves_from(left)
      data_right = hwav.6eaves_from(right)
      
      print("len ", np.shape(data_left), len(data_left)//LENGTH, LENGTH)
      data_left = (np.array(data_left[:60])+ 130000)/2.0/130000 * 255
      print(np.array(data_left))
      scipy.misc.imsave("visualize/input.png", data_left)
      print("MiN", np.min(data_left), "MAX", np.max(data_left))


