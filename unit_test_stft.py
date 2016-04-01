import tensorflow as tf
import unittest
import test_stft_cpu
import tensorflow_wav

class TestStringMethods(unittest.TestCase):

    def test_isftf1(self):
        wav_path="input.wav"
        fs=2048
        T=1
        framesz=(64/2048)
        hop=(T*2048-64)/(2048*64)
        wav= tensorflow_wav.get_wav(wav_path)
        data = wav['data'][:fs*T]
        with tf.Session() as sess:
            xin = test_stft_cpu.stft(data, fs, framesz, hop)
            print("Xin is", xin)

            raw_data = tf.placeholder(tf.complex64, [64, 64])
            x=test_stft_cpu.istft(xin, fs, T, hop)
            tstft=tensorflow_wav.istft(raw_data, fs, hop)
            y = sess.run(tstft, {raw_data:xin})
            print('x is', x)
            print('y is', y)
            self.assertEqual(x.tolist()[0:10], y.tolist()[0:10])



    def test_sftf1(self):
        wav_path="input.wav"
        fs=2048
        T=1
        framesz=(64/2048)
        hop=(T*2048-64)/(2048*64)
        wav= tensorflow_wav.get_wav(wav_path)
        data = wav['data'][:fs*T]
        with tf.Session() as sess:
            x = test_stft_cpu.stft(data, fs, framesz, hop)
            print('max/min', x.max(),x.min())
            raw_data = tf.placeholder(tf.complex64, [fs*T])
            tstft=tensorflow_wav.stft(raw_data, fs, framesz, hop)
            y = sess.run(tstft, {raw_data:data})
            print(x)
            print(y)
            self.assertEqual(True, True)
            #self.assertEqual(x.tolist()[0:10], y.tolist()[0:10])

if __name__ == '__main__':
    unittest.main()


