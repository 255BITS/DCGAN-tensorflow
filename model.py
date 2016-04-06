import os
import time
from glob import glob
import tensorflow as tf

from ops import *
from utils import *
import tensorflow_wav
import lstm

WAV_HEIGHT=64
WAV_WIDTH=64
DIMENSIONS=2

class DCGAN(object):
    def __init__(self, sess, is_crop=True,
                 batch_size=64, sample_size = 2, wav_shape=[WAV_WIDTH, WAV_HEIGHT, DIMENSIONS],
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
        self.sample_size = sample_size
        self.wav_shape = wav_shape

        self.net_size_q=512
        self.keep_prob = 0.9
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
            self.d_bn4 = batch_norm(batch_size, name='d_bn4')

        self.g_bn0 = batch_norm(batch_size, name='g_bn0')
        self.g_bn01 = batch_norm(batch_size, name='g_bn01')
        self.g_bn1 = batch_norm(batch_size, name='g_bn1')
        self.g_bn2 = batch_norm(batch_size, name='g_bn2')

        self.g_bn3 = batch_norm(batch_size, name='g_bn3')
        self.g_bn4 = batch_norm(batch_size, name='g_bn4')

        self.dataset_name = dataset_name
        self.checkpoint_dir = checkpoint_dir
        self.build_model()

    def normalize_wav(self,wav):
        maxWavValue = 46340
        wav = (wav+maxWavValue)/(2*maxWavValue)#tf.add(tf.div(wav,(maxWavValue*2)), 0.5)
        return wav

    def build_model(self):

        self.wavs = tf.placeholder(tf.float32, [self.batch_size, WAV_HEIGHT*WAV_WIDTH, DIMENSIONS],
                                    name='real_wavs')
        self.batch_flatten = self.normalize_wav(tf.reshape(self.wavs, [self.batch_size, -1]))


        self.z_mean, self.z_log_sigma_sq = self.encoder()

        # Draw one sample z from Gaussian distribution
        eps = tf.random_normal((self.batch_size, self.z_dim), 0, 1, dtype=tf.float32)
        # z = mu + sigma*epsilon
        self.z = tf.add(self.z_mean, tf.mul(tf.sqrt(tf.exp(self.z_log_sigma_sq)), eps))
        print(eps)

 
        #self.encoded_wavs=tensorflow_wav.encode(self.wavs)
        #self.encoded_wavs = tf.reshape(self.encoded_wavs, [self.batch_size]+self.wav_shape)
        #self.z_sum = tf.histogram_summary("z", self.z)


        print("shapes d_wav", self.wav_shape, self.wavs.get_shape())
        d_wav = tf.reshape(self.wavs, [self.batch_size] + self.wav_shape)
        self.G = self.generator()
        self.batch_reconstruct_flatten = self.normalize_wav(tf.reshape(self.G, [self.batch_size, -1]))

        print("G is", self.G.get_shape())
        self.D = self.discriminator(d_wav, reuse=None)

        self.sampler = self.sampler()
        self.D_ = self.discriminator(self.G, reuse=True)
        
        self.create_vae_loss_terms()

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
        self.q_vars = [var for var in t_vars if'_q_' in var.name]
        self.vae_vars = self.q_vars+self.g_vars

        self.saver = tf.train.Saver()
    def create_vae_loss_terms(self):
      # The loss is composed of two terms:
      # 1.) The reconstruction loss (the negative log probability
      #     of the input under the reconstructed Bernoulli distribution
      #     induced by the decoder in the data space).
      #     This can be interpreted as the number of "nats" required
      #     for reconstructing the input when the activation in latent
      #     is given.
      # Adding 1e-10 to avoid evaluatio of log(0.0)
      reconstr_loss = \
          -tf.reduce_sum(self.batch_flatten * tf.log(1e-10 + self.batch_reconstruct_flatten)
                         + (1-self.batch_flatten) * tf.log(1e-10 + 1 - self.batch_reconstruct_flatten), 1)
      # 2.) The latent loss, which is defined as the Kullback Leibler divergence
      ##    between the distribution in latent space induced by the encoder on
      #     the data and some prior. This acts as a kind of regularizer.
      #     This can be interpreted as the number of "nats" required
      #     for transmitting the the latent space distribution given
      #     the prior.
      latent_loss = -0.5 * tf.reduce_sum(1 + self.z_log_sigma_sq
                                         - tf.square(self.z_mean)
                                         - tf.exp(self.z_log_sigma_sq), 1)
      self.vae_loss = tf.reduce_mean(reconstr_loss + latent_loss) / (WAV_WIDTH * WAV_HEIGHT * DIMENSIONS) # average over batch and pixel

    def encode(self, X):
      """Transform data by mapping it into the latent space."""
      # Note: This maps to mean of distribution, we could alternatively
      # sample from Gaussian distribution
      return self.sess.run(self.z_mean, feed_dict={self.batch: X})


    def encoder(self):
      # Generate probabilistic encoder (recognition network), which
      # maps inputs onto a normal distribution in latent space.
      # The transformation is parametrized and can be learned.
      H1 = tf.nn.dropout(tf.nn.softplus(linear(self.batch_flatten, self.net_size_q, 'vae_q_lin1')), self.keep_prob)
      H2 = tf.nn.dropout(tf.nn.softplus(linear(H1, self.net_size_q, 'vae_q_lin2')), self.keep_prob)
      z_mean = linear(H2, self.z_dim, 'vae_q_lin3_mean')
      z_log_sigma_sq = linear(H2, self.z_dim, 'vae_q_lin3_log_sigma_sq')
      return (z_mean, z_log_sigma_sq)


    def train(self, config):
        """Train DCGAN"""
        data = glob(os.path.join("./training", "*.mlaudio"))
        print(data)

        #print('g_vars', [shape.get_shape() for shape in self.g_vars])
        #print('d_vars', [shape.get_shape() for shape in self.d_vars])
        d_optim = tf.train.AdamOptimizer(config.learning_rate_d, beta1=config.beta1) \
                          .minimize(self.d_loss, var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(config.learning_rate_g, beta1=config.beta1) \
                          .minimize(self.g_loss, var_list=self.vae_vars)
        vae_optim = tf.train.AdamOptimizer(config.learning_rate_v, beta1=config.beta1) \
                          .minimize(self.vae_loss, var_list=self.vae_vars)


        self.saver = tf.train.Saver()
        #self.g_sum = tf.merge_summary([self.z_sum, self.d__sum, 
        #    self.d_loss_fake_sum, self.g_loss_sum])
        #self.d_sum = tf.merge_summary([self.z_sum, self.d_sum, self.d_loss_real_sum, self.d_loss_sum])
        #self.writer = tf.train.SummaryWriter("./logs", self.sess.graph_def)


        counter = 1
        start_time = time.time()
        tf.initialize_all_variables().run()

        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
        print('epoch', config.epoch)

        for epoch in range(config.epoch):
            batch_files = glob(os.path.join("./training", "*.mlaudio"))
            np.random.shuffle(batch_files)

            def get_wav_content(files):
                for filee in files:
                    print("Yielding ", filee)
                    try:
                        yield tensorflow_wav.get_pre(filee)
                    except Exception as e:
                        print("Could not load ", filee, e)

            #print(batch)
            idx=0
            batch_idxs=0
            diverged_count = 0
            for wavobj in get_wav_content(batch_files):
                print('shape is', wavobj['data'].shape)
                wavdata = wavobj['data']

                dims_map = config.batch_size * WAV_HEIGHT*WAV_WIDTH * DIMENSIONS
                print("DIM map is", dims_map)
                flattened = np.reshape(wavdata, [-1])
                max_items = int(flattened.shape[0]/dims_map)*dims_map
                print("MAX items ", max_items)

                batch_item = flattened[:max_items]
                batch_wavs_multiple = batch_item.reshape([-1, config.batch_size, WAV_HEIGHT*WAV_WIDTH, DIMENSIONS])
                batch_idxs+=1
                errD_fake = 0
                errD_real = 0
                errG = 0
                for i, batch_wavs in enumerate(batch_wavs_multiple):
                    idx+=1
                    #print("Min:", np.min(batch_wavs))
                    #print("Max:", np.max(batch_wavs))

                    _ = self.sess.run((vae_optim, self.vae_loss),
                            feed_dict={self.wavs: batch_wavs})
                    #if(errD_fake > 10):
                    #    errd_range = 3
                    #elif(errD_fake > 8):
                    #    errd_range = 2
                    #else:
                    errd_range=1
                    #print('min', 'max', 'mean', 'stddev', batch_wavs.min(), batch_wavs.max(), np.mean(batch_wavs), np.std(batch_wavs))
                    for repeat in range(errd_range):
                        #print("discrim ", errd_range)
                        # Update D network
                        _= self.sess.run([d_optim],
                                feed_dict={ self.wavs: batch_wavs })
                        #self.writer.add_summary(summary_str, counter)

                    # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
                    #if(errG > 10):
                    #    errg_range = 4
                    #if(errG > 5):
                    #    errg_range = 2
                    #else:
                    errg_range=1
                    for repeat in range(errg_range):
                        #print("generating ", errg_range)
                        # Update G network
                        _= self.sess.run([g_optim],
                                feed_dict={ self.wavs: batch_wavs })
                        #self.writer.add_summary(summary_str, counter)

                    errD_fake = self.d_loss_fake.eval({self.wavs: batch_wavs})
                    errD_real = self.d_loss_real.eval({self.wavs: batch_wavs})
                    errG = self.g_loss.eval({self.wavs: batch_wavs})
                    errVAE = self.vae_loss.eval({self.wavs: batch_wavs})
                    rG = self.G.eval({self.wavs: batch_wavs})
                    #H4 = self.h4.eval({self.wavs: batch_wavs})
                    #bf = self.batch_flatten.eval({self.wavs: batch_wavs})
                    #brf = self.batch_reconstruct_flatten.eval({self.wavs: batch_wavs})
                    #z = self.z.eval({self.wavs: batch_wavs})
                    gen_output = self.gen_output.eval({self.wavs: batch_wavs})

                    #print("H4", np.min(H4), np.max(H4))
                    #print("z", np.min(z), np.max(z))
                    #print("bf", np.min(bf), np.max(bf))
                    #print("brf", np.min(brf), np.max(brf))
                    print("rG", np.min(rG), np.max(rG))
                    print("gen_out", np.min(gen_output), np.max(gen_output))

                    counter += 1
                    print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss_fake %.8f, d_loss: %.8f, g_loss: %.8f vae_loss: %.8f" \
                        % (epoch, idx, batch_idxs,
                            time.time() - start_time, errD_fake, errD_real, errG, errVAE))

                    SAVE_COUNT=300
                    
                    #print("Batch ", counter)
                    if np.mod(counter, SAVE_COUNT) == SAVE_COUNT-3:
                        print("Saving after next batch")
                    if(errD_fake == 0 or errD_fake > 23 or errG > 23 or errVAE > 2 or np.isnan(errVAE)):
                        diverged_count += 1
                        print("Error rate above threshold")
                        if(diverged_count > 20):
                            print("Loading from last checkpoint")
                            loaded = self.load(self.checkpoint_dir)
                            diverged_count = 0
                            print("loaded", loaded)

                    else:
                        diverged_count = 0
                        if np.mod(counter, SAVE_COUNT) == SAVE_COUNT-2:
                            print("Saving !")
                            self.save(config.checkpoint_dir, counter)


    def sample(self):
        result = self.sess.run(
            self.sampler,
            feed_dict={self.wavs: np.ones((self.batch_size, WAV_WIDTH* WAV_HEIGHT, DIMENSIONS))}
        )
        print("len res", np.shape(result))
        return result

    def discriminator(self, wav, reuse=False, y=None):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        c2d = tf.nn.relu(conv2d(wav, 64, name='d_h0_conv'))
        c2d = tf.nn.relu(conv2d(c2d, 8, name='d_h1_conv'))
        #c2d = self.d_bn2(c2d)
        lstm_input = tf.reshape(tf.nn.tanh(c2d), [self.batch_size, WAV_HEIGHT*WAV_WIDTH*DIMENSIONS//4])
        lstm_layer = lstm.discriminator(lstm_input,WAV_HEIGHT*WAV_WIDTH*DIMENSIONS )
        bn_input =  tf.reshape(lstm_layer, [self.batch_size, WAV_HEIGHT,WAV_WIDTH,DIMENSIONS])
        bn = bn_input
        #bn = self.d_bn3(bn_input)
        return tf.nn.sigmoid(bn)

    def generator(self, y=None):
        return self.build_generator(True)
    
    def build_generator(self,is_generator):
        if(not is_generator):
            tf.get_variable_scope().reuse_variables()

        print("Generator creation")
        z = self.z
        print('z', z.get_shape())

        lstm_gen = lstm.generator(self.z, WAV_HEIGHT*WAV_WIDTH*DIMENSIONS//128)

        reshaped = tf.reshape(lstm_gen, [self.batch_size, WAV_WIDTH//8, WAV_HEIGHT//8, DIMENSIONS//2])

        batch_lstm = self.g_bn0(reshaped)
        print("batch shape", batch_lstm.get_shape())
        #c2d = conv2d(batch_lstm, 32, name='g_h0_conv')
        c2d_reshape = deconv2d(batch_lstm, [self.batch_size, WAV_HEIGHT//4,WAV_WIDTH//4, DIMENSIONS], name='g_h0_conv')
        c2d_reshape2 = deconv2d(c2d_reshape, [self.batch_size, WAV_HEIGHT//2,WAV_WIDTH//2, DIMENSIONS], name='g_h1_conv')
        c2d_reshape3 = deconv2d(c2d_reshape2, [self.batch_size, WAV_HEIGHT//1,WAV_WIDTH//1, DIMENSIONS], name='g_h2_conv')
        #c2d_reshape = tf.reshape(c2d, [self.batch_size, WAV_HEIGHT,WAV_WIDTH, DIMENSIONS])
        #output = self.g_bn1(c2d_reshape3)
        self.gen_output = output = c2d_reshape3
        return tensorflow_wav.scale_up(output)


    def sampler(self, y=None):
        return self.build_generator(False)

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
