import glob
import os

def do(command):
    print("Running " + command)
    print(os.system(command))


def add_to_training(dir):
    files = glob.glob(dir+"/*.wav")
    for file in files:
        if '4k' in file:
            print("skipping "+file)
            continue
        print("converting " +file)
        #-bufsize 4096 -b:v 4096
        fname = file.split("/")[-1]
        print("fname", fname)
        process_file=  "training/processed/"+fname
        silent_file = "training/silence_removed/"+fname
        output_file=  "training/"+fname
        do("ffmpeg -y -i \""+file+"\" -ar 4096 \""+process_file+"\"")
        do("ffmpeg -y -i \""+process_file+"\" -ac 1 \""+silent_file+"\"")
        do("sox \""+silent_file+"\" \""+output_file+"\" silence 1 0.1 0.1% reverse silence 1 0.1 0.1% reverse")
        #remove silence
        #do("ffmpeg -i \""+file+"-4k-1-chan.wav\" -af silenceremove=1:0:-30dB:-1:0:0 \""+file+"-4k-mono-silent.wav\"")
        #do("rm \""+file+"-4k-1-chan.wav\"")

add_to_training("youtube/1")
add_to_training("youtube/2")
add_to_training("youtube/3")
#add_to_training("youtube/5")
#add_to_training("youtube/1")
