import wave

# Returns the file object in complex64
def get_wav(sess, path):

    wav = wave.open(path, 'rb')
    data = wav.readframes(wav.getnframes())
    results={}
    results['channels']=wav.getnchannels()
    results['sampwidth']=wav.getsampwidth()
    results['framerate']=wav.getframerate()
    results['nframes']=wav.getnframes()
    results['compname']=wav.getcompname()
    # process fft in tf
    processed = data.astype(complex, copy=False)
    results['data']=processed
    return results

def save_wav(wav, path):

    wav = wave.open(path, 'wb')
    wav.setnchannels(wav.getnchannels())
    wav.setsampwidth(wav.getsampwidth())

    wav.setframerate(wav.getframerate())

    wav.setnframes(wav.getnframes())

    wav.setcomptype(None, 'processed')
    processed = wav['data']
    # process ifft in tf
    wav.writeframes(processed)
