import glob
import os
import tensorflow_wav
import numpy as np
import scipy.signal
import scipy.fftpack
import argparse
import mdct


parser = argparse.ArgumentParser(description='Converts data to mlaudio format.')
parser.add_argument('--sanity', action='store_true')

DIMENSIONS=2

args = parser.parse_args()
print(args)

def do(command):
    print("Running " + command)
    print(os.system(command))


def do_mdct(raw):
    output = mdct.mdct(raw, len(raw))
    return output
def do_dct(raw):
    dct = np.array(scipy.fftpack.dct(raw, norm='ortho'), dtype=np.float32)
    return dct
def do_stft(x, fftsize=126, overlap=4):
    hop = fftsize // overlap
    w = scipy.signal.tukey(fftsize+1)[:-1]      # better reconstruction with this trick +1)[:-1]  
    print(range(0, len(x)-fftsize, hop))
    return np.array([np.fft.rfft(w*x[i:i+fftsize]) for i in range(0, len(x)-fftsize, hop)])

def do_fft(raw):
    zeros = np.zeros_like(raw, dtype=np.float32)
    real = np.array(np.fft.rfft(raw, norm='ortho'), dtype=np.float32)
    zeros[0:len(real)] += real
    return zeros
def preprocess(output_file):
    wav = tensorflow_wav.get_wav(output_file)

    #raw = np.array(wav['data'])
    #raw = raw[:int(raw.shape[0]/BITRATE)*BITRATE]
    #raw = np.reshape(raw, [-1, WAV_X])
    #mdct = [do_mdct(row) for row in raw]
    if(len(wav['data'].shape) > 1):
        data = wav['data'][:44100*20*10, 0]
        data_right = wav['data'][:44100*20*10, 1]
    else:
        data = wav['data'][:44100*20*10]
        data_right = wav['data'][:44100*20*10]

    print("stft")
    stft = do_stft(data)
    stft_right = do_stft(data_right)
    row_length = stft.shape[1]
    print("/stft", stft.shape, row_length)
    wav['stft_row_length']=row_length
    #dct = np.zeros_like(mdct)
    #dct = [do_dct(row) for row in raw]
    #fft = np.swapaxes(fft, 0, 1)

    data = np.concatenate([[stft], [stft_right]])#, [dct]])
    #carefully change the format to [-1, WAV_X, 3] 
    data = np.reshape(data, [2, -1, row_length]) 
    data = np.swapaxes(np.swapaxes(data, 0, 1), 1, DIMENSIONS)
    print("New data shape is ", data.shape)
    wav['data']=data
    print("save")
    tensorflow_wav.save_pre(wav, output_file+".mlaudio")
    print("/save")


def add_to_training(dir):
    files = glob.glob(dir+"/*.wav")
    files += glob.glob(dir+"/*.mp3")
    print(dir+'/wav')
    #files = files[:1]
    for file in files:
        print("converting " +file)
        #-bufsize 4096 -b:v 4096
        fname = file.split("/")[-1]
        ext = fname.split(".")[-1]
        fname = fname.split(".")[0]
        fname+='.wav'
        #print("fname", fname)
        process_file=  "training/processed/"+fname
        silent_file = "training/silence_removed/"+fname
        output_file=  "training/"+fname
        do("ffmpeg -loglevel panic -y -i \""+file+"\" -ar 44100 \""+process_file+"\"")
        do("ffmpeg -loglevel panic -y -i \""+process_file+"\" -ac 2 \""+silent_file+"\"")
        do("sox \""+silent_file+"\" \""+output_file+"\" silence 1 0.1 0.1% reverse silence 1 0.1 0.1% reverse")
        preprocess(output_file)
        #remove silence
        #do("ffmpeg -i \""+file+"-4k-1-chan.wav\" -af silenceremove=1:0:-30dB:-1:0:0 \""+file+"-4k-mono-silent.wav\"")
        #do("rm \""+file+"-4k-1-chan.wav\"")


def sanity_test(input_wav):
    processed = preprocess(input_wav)
    mlaudio = tensorflow_wav.get_pre(input_wav+".mlaudio")
    out = tensorflow_wav.convert_mlaudio_to_wav(mlaudio)
    outfile = input_wav+".sanity.wav"
    tensorflow_wav.save_wav(out, outfile)

if(args.sanity):
    sanity_test("input.wav")
else:
    do("rm training/*.wav")
    do("rm training/*.mlaudio")
    #add_to_training("datasets/one-large")
    #add_to_training("datasets/youtube-drums-2)
    #add_to_training("datasets/youtube-drums-3")
    add_to_training('datasets/drums2')
    #add_to_training('datasets/videogame')

    #add_to_training("datasets/youtube-drums-120bpm-1")
    #add_to_training("youtube/5")
    #add_to_training("youtube/1")
