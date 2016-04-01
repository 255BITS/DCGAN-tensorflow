with tf.Session() as sess:
    with tf.device('/cpu:0'):
            checkpoint_dir = "checkpoint/lstm"

            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                ckpt_name = ckpt.model_checkpoint_path
                saver.restore(self.sess, "checkpoint/"+ckpt_name)
                return True
            else:
                return False

        data = glob(os.path.join("./training", "*.wav"))
        sample_file = data[0]
        sample =tensorflow_wav.get_wav(sample_file)
        print(sample)

        full_audio = []
        for i in range(1):
            audio = dcgan.sample()

            audio = np.reshape(audio,[-1])
            print("Audio shape", np.shape(audio))
            full_audio += audio[:bitrate*batch_size].tolist()
            print("Full audio shape", np.shape(full_audio))

        samplewav = sample.copy()
        samplewav
        print("Generated stats 'min', 'max', 'mean', 'stddev'", np.min(full_audio), np.max(full_audio), np.mean(full_audio), np.std(full_audio))
        samplewav['data']=np.reshape(np.array(full_audio), [-1, 64])
        print("samplewav shape", np.shape(samplewav['data']))

        filename = "./compositions/song.wav.stft"
        tensorflow_wav.save_stft(samplewav, filename )


def generate():
    X_batch = np.random.normal(0,0.5,(seq_length, batch_size, number_of_items))

    feed_dict = {enc_inp[t]: X_batch[t] for t in range(seq_length)}
    logits_batch = sess.run(logits, feed_dict)
    #print(logits_batch)
    return logits_batch


