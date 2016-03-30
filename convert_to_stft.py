fs=2048
framesz=(64/2048)
hop=(2048-64)/(2048*64)
import glob
from test_stft_cpu import stft
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
    wav= tensorflow_wav.get_wav(file)
    data = stft(wav['data'],fs, framesz, hop)
    print(wav)
    wav['data']=data.real
    print(wav['data'])
    #wav['data'] = np.sign(wav['data'])*np.power(wav['data'], 2)
    print(np.min(wav['data']), np.max(wav['data']))
    res= tensorflow_wav.save_stft(wav, file+".stft")
    print(file+".stft"+" is written")
