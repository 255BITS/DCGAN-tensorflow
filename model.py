import os
import time
from glob import glob
import tensorflow as tf

from ops import *
from utils import *
import tensorflow_wav

WAV_SIZE=64
WAV_HEIGHT=64
BITRATE=4096  # this is not the bitrate with stft
class DCGAN(object):
    def __init__(self, sess, wav_size=WAV_SIZE, is_crop=True,
                 batch_size=64, sample_size = 2, wav_shape=[WAV_SIZE, WAV_HEIGHT, 1],
                 y_dim=None, z_dim=64, gf_dim=64, df_dim=64,
                 gfc_dim=1024, dfc_dim=1024, c_dim=1, dataset_name='default',
                 checkpoint_dir='checkpoint'):
        """

        Args:
            sess: TensorFlow session
            batch_size: The size of batch. Should be specified before training.
            y_dim: (optional) Dimension of dim for y. [None]
            z_dim: (optional) Dimension of dim for Z. [100]
            gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
            df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
            gfc_dim: (optional) Dimension of gen untis for for fully connected layer. [1024]
            dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
            c_dim: (optional) Dimension of wav color. [3]
        """
        self.sess = sess
        self.is_crop = is_crop
        self.batch_size = batch_size
        self.wav_size = wav_size
        self.sample_size = sample_size
        self.wav_shape = wav_shape

        self.y_dim = y_dim
        self.z_dim = z_dim

        self.gf_dim = gf_dim
        self.df_dim = df_dim

        self.gfc_dim = gfc_dim
        self.dfc_dim = dfc_dim

        self.c_dim = c_dim

        # batch normalization : deals with poor initialization helps gradient flow
        self.d_bn1 = batch_norm(batch_size, name='d_bn1')
        self.d_bn2 = batch_norm(batch_size, name='d_bn2')

        if not self.y_dim:
            self.d_bn3 = batch_norm(batch_size, name='d_bn3')

        self.g_bn0 = batch_norm(batch_size, name='g_bn0')
        self.g_bn1 = batch_norm(batch_size, name='g_bn1')
        self.g_bn2 = batch_norm(batch_size, name='g_bn2')

        if not self.y_dim:
            self.g_bn3 = batch_norm(batch_size, name='g_bn3')

        self.dataset_name = dataset_name
        self.checkpoint_dir = checkpoint_dir
        self.build_model()

    def build_model(self):


        with tf.variable_scope('scale'):
            sign_real = tf.get_variable('sign_real', [self.batch_size, WAV_HEIGHT, WAV_SIZE,1 ], initializer=tf.constant_initializer(1))
            sign_imag = tf.get_variable('sign_imag', [self.batch_size, WAV_HEIGHT, WAV_SIZE, 1], initializer=tf.constant_initializer(1))
        if self.y_dim:
            self.y= tf.placeholder(tf.float32, [None, self.y_dim], name='y')

        self.wavs = tf.placeholder(tf.float32, [self.batch_size, BITRATE],
                                    name='real_wavs')

        self.z = tf.placeholder(tf.float32, [None, self.z_dim],
                                name='z')

        self.encoded_wavs=tensorflow_wav.encode(self.wavs)
        self.encoded_wavs = tf.reshape(self.encoded_wavs, [self.batch_size]+self.wav_shape)
        #self.z_sum = tf.histogram_summary("z", self.z)

        self.G = self.generator(self.z)
        print("G is", self.G.get_shape(), self.encoded_wavs.get_shape())
        self.D = self.discriminator(self.encoded_wavs, reuse=None)

        self.sampler = self.sampler(self.z)
        self.sampler = tf.reshape(self.sampler,[-1])
        #self.sampler = tensorflow_wav.decode(self.sampler)
        encoded_G = self.G#tensorflow_wav.encode(self.G)
        self.D_ = self.discriminator(encoded_G, reuse=True)
        

        #self.d_sum = tf.histogram_summary("d", self.D)
        #self.d__sum = tf.histogram_summary("d_", self.D_)

        self.d_loss_real = binary_cross_entropy_with_logits(tf.ones_like(self.D), self.D)
        self.d_loss_fake = binary_cross_entropy_with_logits(tf.zeros_like(self.D_), self.D_)
        self.g_loss = binary_cross_entropy_with_logits(tf.ones_like(self.D_), self.D_)

        #self.d_loss_real_sum = tf.scalar_summary("d_loss_real", self.d_loss_real)
        #self.d_loss_fake_sum = tf.scalar_summary("d_loss_fake", self.d_loss_fake)
                                                    
        self.d_loss = self.d_loss_real + self.d_loss_fake

        #self.g_loss_sum = tf.scalar_summary("g_loss", self.g_loss)
        #self.d_loss_sum = tf.scalar_summary("d_loss", self.d_loss)

        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.saver = tf.train.Saver()

    def train(self, config):
        """Train DCGAN"""
        data = glob(os.path.join("./training", "*.stft"))
        print(data)
        #np.random.shuffle(data)

        #print('g_vars', [shape.get_shape() for shape in self.g_vars])
        #print('d_vars', [shape.get_shape() for shape in self.d_vars])
        d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
                          .minimize(self.d_loss, var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
                          .minimize(self.g_loss, var_list=self.g_vars)

        self.saver = tf.train.Saver()
        #self.g_sum = tf.merge_summary([self.z_sum, self.d__sum, 
        #    self.d_loss_fake_sum, self.g_loss_sum])
        #self.d_sum = tf.merge_summary([self.z_sum, self.d_sum, self.d_loss_real_sum, self.d_loss_sum])
        #self.writer = tf.train.SummaryWriter("./logs", self.sess.graph_def)

        sample_z = np.random.uniform(-1, 1, size=(self.batch_size , self.z_dim))
        sample_file = data[0]
        sample =tensorflow_wav.get_stft(sample_file)#get_wav(sample_file, self.wav_size, is_crop=self.is_crop) #[get_wav(sample_file, self.wav_size, is_crop=self.is_crop) for sample_file in sample_files]
        sample_wavs = np.array(sample['data'])

        counter = 1
        start_time = time.time()
        tf.initialize_all_variables().run()

        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
        print('epoch', config.epoch)

        for epoch in range(config.epoch):
            batch_files = glob(os.path.join("./training", "*.stft"))

            def get_wav_content(files):
                for filee in files:
                    print("Yielding ", filee)
                    yield tensorflow_wav.get_stft(filee)

            #print(batch)
            idx=0
            batch_idxs=0
            for wavobj in get_wav_content(batch_files):
                batch_item = wavobj['data']
                print(batch_item, len(batch_item))

                #print(len(batch_item))

                #TODO: review this code to make sure nothing is being deformed
                # Are we properly getting the values?  We can output to a file to be sure 'sanity.wav'

                #pre_fft_batch = batch_item.reshape([-1])
                #print("Computing FFT")
                #data = tf.placeholder(tf.complex64, [pre_fft_batch.shape[0]])
                #post_fft = self.sess.run([tensorflow_wav.build_fft_graph(data)],
                #        feed_dict={ data: pre_fft_batch })
                #print("Done computing FFT")
                max_items = int(len(batch_item)/BITRATE/config.batch_size)*BITRATE * config.batch_size
                batch_item = batch_item[:max_items]
                print("MAX ITEMS IS", max_items, 'to', BITRATE)

                sample_wavs = sample_wavs[:max_items].reshape([-1, config.batch_size, BITRATE])
                batch_idxs+=1
                errD_fake = 0
                errD_real = 0
                errG = 0
                batch_wavs_multiple = batch_item.reshape([-1, config.batch_size, BITRATE])
                for i, batch_wavs in enumerate(batch_wavs_multiple):
                    idx+=1
                    batch_z = np.random.uniform(-1, 1, [config.batch_size, self.z_dim]) \
                                .astype(np.float32)

                    #if(errD_fake > 10):
                    #    errd_range = 3
                    #if(errD_fake > 8):
                    #    errd_range = 2
                    #else:
                    errd_range=1
                    #print('min', 'max', 'mean', 'stddev', batch_wavs.min(), batch_wavs.max(), np.mean(batch_wavs), np.std(batch_wavs))
                    for repeat in range(errd_range):
                        #print("discrim ", errd_range)
                        # Update D network
                        #print("Running discriminator with min/max", batch_wavs.min(), batch_wavs.max())
                        _= self.sess.run([d_optim],
                            feed_dict={ self.wavs: batch_wavs, self.z: batch_z })
                        #self.writer.add_summary(summary_str, counter)

                    # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
                    errg_range=1
                    for repeat in range(errg_range):
                        #print("generating ", errg_range)
                        # Update G network
                        _= self.sess.run([g_optim],
                            feed_dict={ self.z: batch_z })
                        #self.writer.add_summary(summary_str, counter)

                    errD_fake = self.d_loss_fake.eval({self.z: batch_z})
                    errD_real = self.d_loss_real.eval({self.wavs: batch_wavs})
                    errG = self.g_loss.eval({self.z: batch_z})

                    counter += 1
                    print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss_fake %.8f, d_loss: %.8f, g_loss: %.8f" \
                        % (epoch, idx, batch_idxs,
                            time.time() - start_time, errD_fake, errD_real, errG))

                    SAVE_COUNT=10
                    SAMPLE_COUNT=1e10
                    
                    if np.mod(counter, SAVE_COUNT) == SAVE_COUNT-3:
                        print("Saving after next batch")
                    if np.mod(counter,SAMPLE_COUNT) == SAMPLE_COUNT-2:
                        bz = sample_z
                        #bz = np.random.normal(0, 1, [config.batch_size, self.z_dim])
                        #        .astype(np.float32)
                        #print(np.shape(sample_wavs[0]), np.shape(sample_z))
                        samples = self.sess.run(
                            self.sampler,
                            feed_dict={self.z: bz}
                        )
                        samplewav = sample.copy()
                        samplewav['data']=samples
                        print(samplewav)
                        print("[Sample] min, max, avg, mean, stddev", samples.min(), samples.max(), np.average(samples), np.mean(samples), np.std(samples))
                        #print(samples)
                        filename = "./samples/%s_%s_train.png"% (epoch, idx)
                        data = np.array(samplewav['data'])
                        save_data = data.reshape([-1, WAV_SIZE])
                        samplewav['data']=save_data
                        tensorflow_wav.save_stft(samplewav,filename+".stft"  )
                        print("[Sample] saved in "+ filename)

                    if np.mod(counter, SAVE_COUNT) == SAVE_COUNT-2:
                        if(errD_fake == 0 or errD_fake > 23 or errG > 23):
                            print("Refusing to save, error rate above threshold")
                        else:
                            print("Saving !")
                            self.save(config.checkpoint_dir, counter)

    def sample(self, bz=None):
        if(bz == None):
            bz = np.random.normal(0, 1, [self.batch_size, self.z_dim]) 
        result = self.sess.run(
            self.sampler,
            feed_dict={self.z: bz}
        )
        return result


    def discriminator(self, wav, reuse=False, y=None):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        if not self.y_dim:
            print("Discriminator creation")
            print('wav', wav.get_shape())
            h0 = lrelu(conv2d(wav, self.df_dim, name='d_h0_conv'))
            print('h0', h0.get_shape())
            h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, name='d_h1_conv')))
            print('h1', h1.get_shape())
            h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, name='d_h2_conv')))
            print('h2', h2.get_shape())
            h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*8, name='d_h3_conv')))
            print('h3', h3.get_shape())
            h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h3_lin')
            print('h4', h4.get_shape())
            print("End discriminator creation")

            return tf.nn.sigmoid(h4)

    def generator(self, z, y=None):
        if not self.y_dim:
            print("Generator creation")
            print('z', z.get_shape())
            # project `z` and reshape
            self.z_, self.h0_w, self.h0_b = linear(z, self.gf_dim*8*4*4, 'g_h0_lin', with_w=True)
            print('z_', z.get_shape())
            print('self.h0_w', self.h0_w.get_shape())

            self.h0 = tf.reshape(self.z_, [self.batch_size, 4, 4, self.gf_dim * 8])
            h0 = tf.nn.relu(self.g_bn0(self.h0))
            print('h0',h0.get_shape())

            self.h1, self.h1_w, self.h1_b = deconv2d(h0, 
                    [self.batch_size, 8, 8, self.gf_dim*4], name='g_h1', with_w=True)
            h1 = tf.nn.relu(self.g_bn1(self.h1))
            print('h1',h1.get_shape())

            h2, self.h2_w, self.h2_b = deconv2d(h1,
                    [self.batch_size, 16, 16, self.gf_dim*2], name='g_h2', with_w=True)
            h2 = tf.nn.relu(self.g_bn2(h2))
            print('h2',h2.get_shape())

            h3, self.h3_w, self.h3_b = deconv2d(h2,
                    [self.batch_size, 32, 32, self.gf_dim], name='g_h3', with_w=True)
            h3 = tf.nn.relu(self.g_bn3(h3))
            print('h3',h3.get_shape())

            h4= deconv2d(h3,
                    [self.batch_size, WAV_SIZE, WAV_HEIGHT, 1], name='g_h4', with_w=False, no_bias=False)

            print('h4',h4.get_shape())
            tanh = tf.nn.tanh(h4)
            return tensorflow_wav.scale_up(h4)

    def sampler(self, z, y=None):
        tf.get_variable_scope().reuse_variables()

        if not self.y_dim:
            print("Sampler creation")
            # project `z` and reshape
            h0 = tf.reshape(linear(z, self.gf_dim*8*4*4, 'g_h0_lin'),
                            [-1, 4, 4, self.gf_dim * 8])
            h0 = tf.nn.relu(self.g_bn0(h0, train=False))
            print('h0', h0.get_shape())

            h1 = deconv2d(h0, [self.batch_size, 8, 8, self.gf_dim*4], name='g_h1')
            h1 = tf.nn.relu(self.g_bn1(h1, train=False))
            print('h1', h1.get_shape())

            h2 = deconv2d(h1, [self.batch_size, 16, 16, self.gf_dim*2], name='g_h2')
            h2 = tf.nn.relu(self.g_bn2(h2, train=False))
            print('h2', h2.get_shape())

            h3 = deconv2d(h2, [self.batch_size, 32, 32, self.gf_dim], name='g_h3')
            h3 = tf.nn.relu(self.g_bn3(h3, train=False))
            print('h3', h3.get_shape())

            h4 = deconv2d(h3, [self.batch_size, 64, 64, 1], name='g_h4', no_bias=False)
            print('h4', h4.get_shape())

            #tanh = tf.nn.tanh(h4)

            return tensorflow_wav.scale_up(h4)

    def save(self, checkpoint_dir, step):
        model_name = "DCGAN.model"
        model_dir = "%s_%s" % (self.dataset_name, self.batch_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")

        model_dir = "%s_%s" % (self.dataset_name, self.batch_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print("TRUE")
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            print("FALSE")
            return False
