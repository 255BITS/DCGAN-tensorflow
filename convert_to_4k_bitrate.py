import glob
import os

def do(command):
    print("Running " + command)
    print(os.system(command))


files = glob.glob("training/*.wav")
for file in files:
    do("ffmpeg -i \""+file+"\" -bufsize 4k -b:v 4k "+file+"-4k.wav")
    do("rm \""+file+"\"")

