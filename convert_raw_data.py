import glob
import os
import tensorflow_wav
import numpy as np
import scipy.signal
import scipy.fftpack
import argparse
import mdct
import pywt


parser = argparse.ArgumentParser(description='Converts data to mlaudio format.')
parser.add_argument('--sanity', action='store_true')
parser.add_argument('--insanity', action='store_true')

args = parser.parse_args()
print(args)
BITRATE = 8192

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
    result = np.array([np.fft.rfft(w*x[i:i+fftsize]) for i in range(0, len(x)-fftsize, hop)])
    return np.real(result)

def do_fft(raw):
    zeros = np.zeros_like(raw, dtype=np.float32)
    real = np.array(np.fft.rfft(raw, norm='ortho'), dtype=np.float32)
    zeros[0:len(real)] += real
    return zeros

#breaks down a list of raw data into Nxsize chunks,
#where size is the length of the raw pcm data to encode(usually divisible into the bitrate)
# the first 2 64 entries in the dimension are the main wavelet, followed by 2 64 detail wavelets
#and N is the length of the data
def do_dwt(data, size=4096):
    rows = data.reshape([-1, size])
    def breakdown(row):
        result = pywt.dwt(row, 'db1')
        return result
    x = [breakdown(row) for row in rows]
    return np.array(x)


def resize_multiple(data, multiple):
    by = len(data)//multiple
    data = data[:by*multiple]
    return data
def preprocess(output_file):
    wav = tensorflow_wav.get_wav(output_file)

    #raw = np.array(wav['data'])
    #raw = raw[:int(raw.shape[0]/BITRATE)*BITRATE]
    #raw = np.reshape(raw, [-1, WAV_X])
    #mdct = [do_mdct(row) for row in raw]
    length = -1#BITRATE*5
    if(len(wav['data'].shape) > 1):
        data = wav['data'][:length, 0]
        data_right = wav['data'][:length, 1]
    else:
        raise("MONO NOT SUPPORTED")
        data = wav['data'][:length]
        data_right = wav['data'][:length]
    
    data = resize_multiple(data, BITRATE)
    data_right = resize_multiple(data, BITRATE)
    #stft = do_stft(data)
    #stft_right = do_stft(data_right)
    dwt = do_dwt(data)
    dwt_right = do_dwt(data_right)
    #dct = np.zeros_like(mdct)
    #dct = [do_dct(row) for row in raw]
    #fft = np.swapaxes(fft, 0, 1)

    dwt = np.reshape(dwt, [1, -1, 4096])
    dwt_right = np.reshape(dwt_right, [1, -1, 4096])
    data = np.concatenate([dwt, dwt_right], 0)#, [dct]])
    # the data is now in the form [2, -1, 4096]
    #carefully change the format to [-1, 4096, 2]
    data = np.swapaxes(np.swapaxes(data, 0, 1), 1, 2)
    wav['data']=data
    print("Data is of the form", np.shape(data))
    tensorflow_wav.save_pre(wav, output_file+".mlaudio")
    audio = wav['data']


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
        do("ffmpeg -loglevel panic -y -i \""+file+"\" -ar "+str(BITRATE)+" \""+process_file+"\"")
        do("ffmpeg -loglevel panic -y -i \""+process_file+"\" -ac 2 \""+silent_file+"\"")
        do("sox \""+silent_file+"\" \""+output_file+"\" silence 1 0.1 0.1% reverse silence 1 0.1 0.1% reverse")
        try:
            preprocess(output_file)
        except:
            print("Oops that broke")
        #remove silence
        #do("ffmpeg -i \""+file+"-4k-1-chan.wav\" -af silenceremove=1:0:-30dB:-1:0:0 \""+file+"-4k-mono-silent.wav\"")
        #do("rm \""+file+"-4k-1-chan.wav\"")


def sanity_test(input_wav):
    processed = preprocess(input_wav)
    mlaudio = tensorflow_wav.get_pre(input_wav+".mlaudio")
    out = tensorflow_wav.convert_mlaudio_to_wav(mlaudio)
    outfile = input_wav+".sanity.wav"
    tensorflow_wav.save_wav(out, outfile)

def insanity_test(input_wav):

    wav = tensorflow_wav.get_wav(input_wav)
    wavdata = wav['data']

    converted = pywt.wavedec(np.reshape(wavdata, [-1]), 'db1')
    print(np.shape(converted), len(converted[24]), len(converted[23]))

    c_lv = 9617408
    converted[24] = np.zeros(c_lv)
    converted[23] = np.zeros(c_lv/2)
    converted[22] = np.zeros(c_lv/4)
    converted[21] = np.zeros(c_lv/8)
    converted_data = pywt.waverec(converted, 'db1')

    wav['data'] = converted_data
    tensorflow_wav.save_wav(wav, "insanity.wav")


if(args.sanity):
    sanity_test("input.wav")
if(args.insanity):
    insanity_test("input.wav")
else:
    do("rm training/*.wav")
    do("rm training/*.mlaudio")
    #add_to_training("datasets/one-large")
    #add_to_training("datasets/youtube-drums-2)
    #add_to_training("datasets/youtube-drums-3")
    #add_to_training('datasets/drums2')
    add_to_training('datasets/youtube-drums-120bpm-1')
    #add_to_training('datasets/videogame')

    #add_to_training("datasets/youtube-drums-120bpm-1")
    #add_to_training("youtube/5")
    #add_to_training("youtube/1")
