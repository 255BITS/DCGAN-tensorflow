fs=2048
framesz=(64/2048)
hop=(2048-64)/(2048*64)
import glob
from test_stft_cpu import stft, istft
import os
import sys
import tensorflow_wav
import numpy as np
import math
if(len(sys.argv)<2):
    print("You have to pick a file")
    no_file_Picked_Exception

files = glob.glob(sys.argv[1])

if(len(files)==0):
    print("Who you trying to kid?")

for file in files:
    print("Reading stft for "+file, ' at rate ', fs)
    wav= tensorflow_wav.get_stft(file)
    nframes = wav['nframes']
    print('shape', wav['data'].shape)
    time = np.shape(np.reshape(wav['data'],[-1]))[0]/4096
    print(np.shape(wav['data']))
    #print(np.min(wav['data']), np.max(wav['data']), np.mean(wav['data']), np.std(wav['data']))
    #wav['data'] = np.exp(wav['data'])
    data = istft(wav['data'],fs, time, hop)
    wav['data'] = data*6
    #print(wav)
    #wav['data'] = np.sign(wav['data'])*np.sqrt(wav['data'])
    res= tensorflow_wav.save_wav(wav, file+".istft")
    print(file+".istft"+" is written")
