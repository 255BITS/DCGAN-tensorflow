"""
Some codes from https://github.com/Newmu/dcgan_code
"""
import math
import json
import random
import pprint
import scipy.misc
import numpy as np
from time import gmtime, strftime
from cmath import polar
import tensorflow_wav

from wav import *

WAV_HEIGHT=64
pp = pprint.PrettyPrinter()

get_stddev = lambda x, k_h, k_w: 1/math.sqrt(k_w*k_h*x.get_shape()[-1])


def imread(path):
    return scipy.misc.imread(path).astype(np.float)

def merge_wavs(wavs, size):
    return inverse_transform(wavs)

def merge(wavs, size):
    h, w = wavs.shape[1], wavs.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))

    for idx, wav in enumerate(wavs):
        i = idx % size[1]
        j = idx / size[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = wav

    return img

def imsave(wavs, size, path):
    return scipy.misc.imsave(path, merge(wavs, size))

def center_crop(x, crop_h, crop_w=None, resize_w=64):
    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[:2]
    j = int(round((h - crop_h)/2.))
    i = int(round((w - crop_w)/2.))
    return scipy.misc.imresize(x[j:j+crop_h, i:i+crop_w],
                               [resize_w, resize_w])

def transform(wav, npx=64, is_crop=True):
    # npx : # of pixels width/height of wav
    if is_crop:
        cropped_wav = center_crop(wav, npx)
    else:
        cropped_wav = wav
    return np.array(cropped_wav)/127.5 - 1.

def inverse_transform(wavs):
    return (wavs+1.)/2.


def to_json(output_path, *layers):
    with open(output_path, "w") as layer_f:
        lines = ""
        for w, b, bn in layers:
            layer_idx = w.name.split('/')[0].split('h')[1]

            B = b.eval()

            if "lin/" in w.name:
                W = w.eval()
                depth = W.shape[1]
            else:
                W = np.rollaxis(w.eval(), 2, 0)
                depth = W.shape[0]

            biases = {"sy": 1, "sx": 1, "depth": depth, "w": ['%.2f' % elem for elem in list(B)]}
            if bn != None:
                gamma = bn.gamma.eval()
                beta = bn.beta.eval()

                gamma = {"sy": 1, "sx": 1, "depth": depth, "w": ['%.2f' % elem for elem in list(gamma)]}
                beta = {"sy": 1, "sx": 1, "depth": depth, "w": ['%.2f' % elem for elem in list(beta)]}
            else:
                gamma = {"sy": 1, "sx": 1, "depth": 0, "w": []}
                beta = {"sy": 1, "sx": 1, "depth": 0, "w": []}

            if "lin/" in w.name:
                fs = []
                for w in W.T:
                    fs.append({"sy": 1, "sx": 1, "depth": W.shape[0], "w": ['%.2f' % elem for elem in list(w)]})

                lines += """
                    var layer_%s = {
                        "layer_type": "fc", 
                        "sy": 1, "sx": 1, 
                        "out_sx": 1, "out_sy": 1,
                        "stride": 1, "pad": 0,
                        "out_depth": %s, "in_depth": %s,
                        "biases": %s,
                        "gamma": %s,
                        "beta": %s,
                        "filters": %s
                    };""" % (layer_idx.split('_')[0], W.shape[1], W.shape[0], biases, gamma, beta, fs)
            else:
                fs = []
                for w_ in W:
                    fs.append({"sy": 5, "sx": 5, "depth": W.shape[3], "w": ['%.2f' % elem for elem in list(w_.flatten())]})

                lines += """
                    var layer_%s = {
                        "layer_type": "deconv", 
                        "sy": 5, "sx": 5,
                        "out_sx": %s, "out_sy": %s,
                        "stride": 2, "pad": 1,
                        "out_depth": %s, "in_depth": %s,
                        "biases": %s,
                        "gamma": %s,
                        "beta": %s,
                        "filters": %s
                    };""" % (layer_idx, 2**(int(layer_idx)+2), 2**(int(layer_idx)+2),
                             W.shape[0], W.shape[3], biases, gamma, beta, fs)
        layer_f.write(" ".join(lines.replace("'","").split()))

def make_gif(wavs, fname, duration=2, true_wav=False):
  return
  #import moviepy.editor as mpy

  #def make_frame(t):
  #  try:
  #    x = wavs[int(len(wavs)/duration*t)]
  #  except:
  #    x = wavs[-1]

  #  if true_wav:
  #    return x.astype(np.uint8)
  #  else:
  #    return ((x+1)/2*255).astype(np.uint8)

  #clip = mpy.VideoClip(make_frame, duration=duration)
  #clip.write_gif(fname, fps = len(wavs) / duration)

def visualize(sess, dcgan, config, option):
  in_wav = tensorflow_wav.get_wav('input.wav')
  if option == 0:
    z_sample = np.random.uniform(-0.5, 0.5, size=(config.batch_size, dcgan.z_dim))
    samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
    in_wav['data']=samples
    tensorflow_wav.save_wav(in_wav, './samples/test.wav' )
  elif option == 1:
    values = np.arange(0, 1, 1./config.batch_size)
    for idx in range(100):
      print(" [*] %d" % idx)
      z_sample = np.zeros([config.batch_size, dcgan.z_dim])
      for kdx, z in enumerate(z_sample):
        z[idx] = values[kdx]

      samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
      in_wav['data']=samples
      tensorflow_wav.save_wav(in_wav, './samples/test_arange_%s.wav'%(idx) )
  elif option == 2:
    values = np.arange(0, 1, 1./config.batch_size)
    for idx in [random.randint(0, 99) for _ in range(100)]:
      print(" [*] %d" % idx)
      z = np.random.uniform(-0.2, 0.2, size=(dcgan.z_dim))
      z_sample = np.tile(z, (config.batch_size, 1))
      #z_sample = np.zeros([config.batch_size, dcgan.z_dim])
      for kdx, z in enumerate(z_sample):
        z[idx] = values[kdx]

      samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
      make_gif(samples, './samples/test_gif_%s.gif' % (idx))
  elif option == 3:
    values = np.arange(0, 1, 1./config.batch_size)
    for idx in range(100):
      print(" [*] %d" % idx)
      z_sample = np.zeros([config.batch_size, dcgan.z_dim])
      for kdx, z in enumerate(z_sample):
        z[idx] = values[kdx]

      samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
      make_gif(samples, './samples/test_gif_%s.gif' % (idx))
  elif option == 4:
    wav_set = []
    values = np.arange(0, 1, 1./config.batch_size)

    for idx in range(100):
      print(" [*] %d" % idx)
      z_sample = np.zeros([config.batch_size, dcgan.z_dim])
      for kdx, z in enumerate(z_sample): z[idx] = values[kdx]

      wav_set.append(sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample}))
      make_gif(wav_set[-1], './samples/test_gif_%s.gif' % (idx))

    new_wav_set = [merge(np.array([wavs[idx] for wavs in wav_set]), [10, 10]) \
        for idx in range(64) + range(63, -1, -1)]
    make_gif(new_wav_set, './samples/test_gif_merged.gif', duration=8)
