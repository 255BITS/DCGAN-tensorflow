import glob
import os

def do(command):
    print("Running " + command)
    print(os.system(command))


files = glob.glob("training/*.wav")
for file in files:
    if '4k' in file:
        print("skipping "+file)
        continue
    print("converting " +file)
    #-bufsize 4096 -b:v 4096
    do("ffmpeg -i \""+file+"\" -ar 4096 \""+file+"-4k.wav\"")
    do("ffmpeg -i \""+file+"-4k.wav\" -ac 1 \""+file+"-4k-1-chan.wav\"")
    #remove silence
    #do("ffmpeg -i \""+file+"-4k-1-chan.wav\" -af silenceremove=1:0:-30dB:-1:0:0 \""+file+"-4k-mono-silent.wav\"")
    do("mv \""+file+"\" training/processed")
    do("rm \""+file+"-4k.wav\"")
    #do("rm \""+file+"-4k-1-chan.wav\"")

