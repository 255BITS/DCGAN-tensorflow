import glob
import os
import tensorflow_wav
import numpy as np

WAV_X=64

def do(command):
    print("Running " + command)
    print(os.system(command))


def do_fft(raw):
    complex = np.fft.fft(raw, norm='ortho')
    output = np.array([np.real(complex), np.imag(complex)])
    return output
def preprocess(output_file):
    wav = tensorflow_wav.get_wav(output_file)

    raw = np.array(wav['data'])
    raw = raw[:int(raw.shape[0]/WAV_X)*WAV_X]
    raw = np.reshape(raw, [-1, WAV_X])
    fft = [do_fft(row) for row in raw]
    fft = np.swapaxes(fft, 0, 1)
    
    print(np.shape(fft), raw.shape)
    data = np.concatenate([[raw], [fft[0]], [fft[1]]])
    #carefully change the format to [-1, WAV_X, 3]
    print(np.shape(data), data[0].shape, data[1].shape, data[2].shape)
    data = np.reshape(data, [3, -1, WAV_X])
    data = np.swapaxes(np.swapaxes(data, 0, 1), 1, 2)
    wav['data']=data
    tensorflow_wav.save_pre(wav, output_file+".mlaudio")


def add_to_training(dir):
    files = glob.glob(dir+"/*.wav")
    #files = files[:1]
    for file in files:
        print("converting " +file)
        #-bufsize 4096 -b:v 4096
        fname = file.split("/")[-1]
        #print("fname", fname)
        process_file=  "training/processed/"+fname
        silent_file = "training/silence_removed/"+fname
        output_file=  "training/"+fname
        do("ffmpeg -loglevel panic -y -i \""+file+"\" -ar 8192 \""+process_file+"\"")
        do("ffmpeg -loglevel panic -y -i \""+process_file+"\" -ac 1 \""+silent_file+"\"")
        do("sox \""+silent_file+"\" \""+output_file+"\" silence 1 0.1 0.1% reverse silence 1 0.1 0.1% reverse")
        try:
            preprocess(output_file)
        except Exception as e:
            print("Error preprocessing", output_file, e)
        #remove silence
        #do("ffmpeg -i \""+file+"-4k-1-chan.wav\" -af silenceremove=1:0:-30dB:-1:0:0 \""+file+"-4k-mono-silent.wav\"")
        #do("rm \""+file+"-4k-1-chan.wav\"")

do("rm training/*.wav")
do("rm training/*.mlaudio")
#add_to_training("datasets/one-large")
#add_to_training("datasets/youtube-drums-2)
#add_to_training("datasets/youtube-drums-3")
add_to_training('datasets/videogame')
#add_to_training("datasets/youtube-drums-120bpm-1")
#add_to_training("youtube/5")
#add_to_training("youtube/1")
