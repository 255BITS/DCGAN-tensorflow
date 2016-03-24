import glob
import os

def do(command):
    print("Running " + command)
    print(os.system(command))


files = glob.glob("training/*.wav")
for file in files:
    do("ffmpeg -i \""+file+"\" -bufsize 4096 -b:v 4096 -ar 4096 "+file+"-4k.wav")
    do("mv \""+file+"\" training/processed")

