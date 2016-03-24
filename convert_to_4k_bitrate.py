import glob
import os

def do(command):
    print("Running " + command)
    print(os.system(command))


files = glob.glob("training/*.wav")
for file in files:
    #-bufsize 4096 -b:v 4096
    do("ffmpeg -i \""+file+"\" -ar 4096 "+file+"-4k.wav")
    do("ffmpeg -i \""+file+"-4k.wav\" -ac 1 "+file+"-4k-1-chan.wav")
    do("mv \""+file+"\" training/processed")
    do("rm \""+file+"-4k.wav\"")

