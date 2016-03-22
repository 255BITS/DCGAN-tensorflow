
# Create sample wav file from util
# Ensure correct dimensionality
# should be [w, h, z] where w=441, h=?(0.25 seconds?), and z=2(complex, real)

# Sanity test load and save


from utils import *

wav_path="training/02-overworld-01.wav"
wav_size=441

wav, raw = get_wav(wav_path, wav_size, is_crop=True)

#print("WAV IS", wav)

res,res2 = save_wav(wav, wav_size, "sanity.wav")
print(wav[0][0][0], res[0], wav[0][0][0]==res[0])
print(raw[0], res2[0], raw[0]==res2[0])
