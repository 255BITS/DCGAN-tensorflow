import os
import sys
import glob
import time

def do(command):
    print("Running " + command)
    print(os.system(command))

i = 0
if(len(sys.argv) > 1):
    do("cd training/to_process && scdl -c -a -l "+sys.argv[1])

    for file in glob.glob('training/to_process/**/*.mp3'):
        wav_out = 'training/wav'+str(i)+'-'+str(time.time())+'.wav'
        do("ffmpeg -i \""+file+"\" -ac 1 -bufsize 4k -b:v 4k "+wav_out)
        #do("rm \""+file+"\"")
        i+=1

else:
    print("Usage: " + sys.argv[0]+" [link to soundcloud playlist]")
