fs=2048
T = 1
framesz=(64/2048)
hop=(T*2048-64)/(2048*64)

for file in files:
    s = read_stft(file)
    write(file, is)
