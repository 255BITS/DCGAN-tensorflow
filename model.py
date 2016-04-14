import os
import time
from glob import glob
import scipy
import tensorflow as tf

from ops import *
from utils import *
import tensorflow_wav
import lstm
import hwav

LENGTH = 20
Y_DIM = 4096

class DCGAN(object):
    def __init__(self, sess, is_crop=True,
                 batch_size=64, sample_size = 2, t_dim=LENGTH,
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
        self.t_dim = t_dim

        self.net_size_q=512
        self.keep_prob = 1
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
        maxWavValue = 130000
        wav = (wav+maxWavValue)/(2*maxWavValue)#tf.add(tf.div(wav,(maxWavValue*2)), 0.5)
        return wav

    def build_model(self):

        self.wavs = tf.placeholder(tf.float32, [self.batch_size, Y_DIM, LENGTH],
                                    name='real_wavs')
        self.batch_flatten = self.normalize_wav(tf.reshape(self.wavs, [self.batch_size, -1]))

        self.t_vec = self.coordinates(self.t_dim)

        self.t = tf.placeholder(tf.float32, [self.batch_size, LENGTH])

        self.z_mean, self.z_log_sigma_sq = self.encoder()

        # Draw one sample z from Gaussian distribution
        eps = tf.random_normal((self.batch_size, self.z_dim), 0, 1, dtype=tf.float32)
        # z = mu + sigma*epsilon
        self.z = tf.add(self.z_mean, tf.mul(tf.sqrt(tf.exp(self.z_log_sigma_sq)), eps))
        print(eps)

 

        d_wav = tf.reshape(self.wavs, [self.batch_size, Y_DIM, LENGTH])
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
      self.vae_loss = tf.reduce_mean(reconstr_loss + latent_loss) / (LENGTH*Y_DIM) # average over batch and pixel

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

        #print('g_vars', [shape.get_shape() for shape in self.g_vars])
        #print('d_vars', [shape.get_shape() for shape in self.d_vars])
        d_optim = tf.train.AdamOptimizer(config.learning_rate_d, beta1=config.beta1) \
                          #.minimize(self.d_loss, var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(config.learning_rate_g, beta1=config.beta1) \
                          #.minimize(self.g_loss, var_list=self.vae_vars)
        vae_optim = tf.train.AdamOptimizer(config.learning_rate_v, beta1=config.beta1) \
                          .minimize(self.vae_loss, var_list=self.vae_vars)

        
        self.grad_clip = 5
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.d_loss, self.d_vars),
                                self.grad_clip)
        d_optim = d_optim.apply_gradients(zip(grads, self.d_vars))
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.g_loss, self.g_vars),
                                self.grad_clip)
        g_optim = g_optim.apply_gradients(zip(grads, self.g_vars))


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

            def get_wav_content(files, batch_size):
                for filea in files:
                    try:
                        def get_batch(filee):

                            print("Loading", filee)
                            mlaudio = tensorflow_wav.get_pre(filee)
                            left, right = mlaudio['wavdec']
                            data_left = hwav.leaves_from(left)
                            data_right = hwav.leaves_from(right)

                            batch = np.empty(len(data_left) + len(data_right)).tolist()
                            batch[0::2]=data_left
                            batch[1::2]=data_right
                            batch = np.array([b[:LENGTH] for b in batch])
                            return batch
                        #scipy.misc.imsave("visualize/input-full.png", data_left[:Y_DIM])
                        batches = [get_batch(filea)]#, get_batch(fileb)]
                        splitInto = 1# segments
                        amountNeeded = batch_size * Y_DIM
                        for i in range(0,len(batches[0])-amountNeeded, batch_size * Y_DIM//splitInto): #  window over the song.  every nn sees every entry. * 2 for left / right speaker
                            #if(batch[i][LENGTH-1] == 0.0):
                            #   print("reached end of file?")
                            #   next
                            batcha = np.array(batches[0][i:i+amountNeeded])
                            batcha = np.reshape(batcha, [batch_size, Y_DIM, LENGTH])
                            #scipy.misc.imsave("visualize/input-"+str(i)+".png", batcha[0][0::2])
                            
                            yield [batcha, i/len(batches[0]), 1.0/batch_size]

                    except Exception as e:
                        print("Could not load ", filee, e)

            #print(batch)
            idx=0
            batch_idxs=0
            diverged_count = 0
            for wavobj, position, stepsize in get_wav_content(batch_files, self.batch_size):
                batch_wavs = wavobj
                batch_idxs+=1
                errD_fake = 0
                errD_real = 0
                errG = 0
                t = self.coordinates(self.t_dim)
                t = np.array(t, dtype=np.float32)
                t *= 0.25*stepsize
                t += position
                t *= 20
                print(position)

                idx+=1
                #print("Min:", np.min(batch_wavs))
                #print("Max:", np.max(batch_wavs))

                _ = self.sess.run((vae_optim, self.vae_loss),
                        feed_dict={self.t: t, self.wavs: batch_wavs})
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
                            feed_dict={self.t: t, self.wavs: batch_wavs })
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
                            feed_dict={self.t: t, self.wavs: batch_wavs })
                    #self.writer.add_summary(summary_str, counter)

                errD_fake = self.d_loss_fake.eval({self.t: t, self.wavs: batch_wavs})
                errD_real = self.d_loss_real.eval({self.t: t, self.wavs: batch_wavs})
                errG = self.g_loss.eval({self.t: t, self.wavs: batch_wavs})
                errVAE = self.vae_loss.eval({self.t: t, self.wavs: batch_wavs})
                rG = self.G.eval({self.t: t, self.wavs: batch_wavs})
                rZ = self.z.eval({self.t: t, self.wavs: batch_wavs})

                #H4 = self.h4.eval({self.wavs: batch_wavs})
                #bf = self.batch_flatten.eval({self.wavs: batch_wavs})
                #brf = self.batch_reconstruct_flatten.eval({self.wavs: batch_wavs})
                #z = self.z.eval({self.wavs: batch_wavs})

                #print("H4", np.min(H4), np.max(H4))
                print("z", np.min(rZ), np.max(rZ))
                #print("bf", np.min(bf), np.max(bf))
                #print("brf", np.min(brf), np.max(brf))
                print("rG", np.min(rG), np.max(rG))

                counter += 1
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss_fake %.8f, d_loss: %.8f, g_loss: %.8f vae_loss: %.8f" \
                    % (epoch, idx, batch_idxs,
                        time.time() - start_time, errD_fake, errD_real, errG, errVAE))

                SAVE_COUNT=300
                
                SAMPLE_COUNT=100
                if np.mod(counter, SAMPLE_COUNT) == SAMPLE_COUNT-3:
                    stepsize = 0.03125
                    position = 0.0
                    t = self.coordinates(self.t_dim)
                    t = np.array(t, dtype=np.float32)
                    t *= 0.5*stepsize
                    t += position
                    t *= 20

                    X = 8
                    Y = 8
                    sample_rows = 20

                    np.random.seed(42)
                    audio = self.sample(t)
                    audio = np.reshape(audio[0::2], (-1, LENGTH))
                    audio = np.reshape(audio[:X*Y*sample_rows], [X*LENGTH, Y*sample_rows])


                    np.random.seed(2103123)
                    audiob = self.sample(t)
                    audiob = np.reshape(audiob[0::2], (-1, LENGTH))
                    audiob = np.reshape(audiob[:X*Y*sample_rows], [X*LENGTH, Y*sample_rows])

                    audiof = np.hstack([audio, audiob])

                    scipy.misc.imsave("visualize/samples-%08d-both.png" % counter, audiof)
                    scipy.misc.imsave("visualize/samples-%08d-sub.png" % counter, np.subtract(audio, audiob))

                    

                #print("Batch ", counter)
                if np.mod(counter, SAVE_COUNT) == SAVE_COUNT-3:
                    print("Saving after next batch")
                if(np.isnan(errVAE)):
                    diverged_count += 1
                    print("Error rate above threshold")
                    if(diverged_count > 1):
                        print("Loading from last checkpoint")
                        loaded = self.load(self.checkpoint_dir)
                        diverged_count = 0
                        print("loaded", loaded)

                else:
                    diverged_count = 0
                    if np.mod(counter, SAVE_COUNT) == SAVE_COUNT-2:
                        print("Saving !")
                        self.save(config.checkpoint_dir, counter)


    def sample(self, t):
        wavs = (np.random.uniform(-1,1.0,(self.batch_size, Y_DIM, LENGTH))*40000)
        #wavs = np.ones((self.batch_size, Y_DIM, LENGTH)) * 40000
        rZ = self.z.eval({self.t: t, self.wavs: wavs})
        print("! sample z", np.min(rZ), np.max(rZ))
        result = self.sess.run(
            self.sampler,
            feed_dict={self.wavs: wavs ,
                       self.t: t}
        )
        return result

    def discriminator(self, wav, reuse=False, y=None):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        depth = 4
        network_size = 64
        wav_unroll = tf.reshape(wav, [self.batch_size, Y_DIM*LENGTH])

        #U = fully_connected(wav_unroll, network_size, 'd_0_wav', with_bias= False)
        #print("D U ", U.get_shape())
        
        #H = tf.nn.softplus(U)
        U = wav_unroll
        H = U

        c1_dim=16
        c2_dim=32
        c3_dim=64
        #H = wav
        H = tf.nn.dropout(H, self.keep_prob)
        H =  tf.reshape(H, [self.batch_size, 20,Y_DIM, 1])
        H = tf.nn.tanh(conv2d(H, c1_dim, name="d_conv1", k_w=3, k_h=3))
        H = tf.nn.dropout(H, self.keep_prob)
        H = tf.nn.tanh(conv2d(H, c2_dim, name="d_conv2", k_w=3, k_h=3))
        H = tf.nn.dropout(H, self.keep_prob)
        H = conv2d(H, c3_dim, name="d_conv3", k_w=3, k_h=3)
        H = tf.reshape(H, [self.batch_size, -1])
        #for i in range(1, depth):
        #  H = tf.nn.tanh(fully_connected(H, network_size, 'd_tanh_'+str(i)))
        #  H = tf.nn.dropout(H, self.keep_prob)
        #  print("D H ", H.get_shape())
        output = H
        #output = linear(output, 128, "d_lstm_in")
        #output = tf.nn.tanh(lstm.discriminator(output, network_size, 'd_lstm0'))
        #output = lstm.enerator(self.z, LENGTH)
        output = tf.nn.tanh(output)
        #in_d = tf.matmul(wav_unroll,tf.ones([ wav_unroll.get_shape()[1], 16])) #batch_size, 16
        #in_d = tf.matmul(output,tf.ones([ output.get_shape()[1], 64])) #batch_size, 16
        in_d = linear(output, 64, "d_lstm_lin")
        disc = lstm.discriminator(in_d, 1, 'd_lstm')
        output = linear(output, 1, "d_fc_out")
        print("D OUT", output.get_shape())



        return tf.nn.sigmoid(output*disc)


    def generator(self, y=None):
        return self.build_generator(True)
    
    def build_generator(self,is_generator):
        if(not is_generator):
            tf.get_variable_scope().reuse_variables()
        network_size = 512
        scale = 1.0
        depth = 4

        #z =  (np.random.uniform(-1,1.0,(self.batch_size, network_size))*scale)

        #z_scaled = tf.reshape(self.z, [self.batch_size, 1, self.z_dim]) * \
        #                tf.ones([LENGTH, 1], dtype=tf.float32) #* scale

        #z_unroll = tf.reshape(z_scaled, [self.batch_size, LENGTH*self.z_dim])
        #l_unroll = lstm.generator(z_unroll, LENGTH*self.z_dim)
        #t_unroll = tf.reshape(self.t, [self.batch_size, self.t_dim])

        #U = z_unroll
        #H = U
        #U = fully_connected(z_unroll*scale, network_size, 'g_0_z', with_bias=True)# + \
        #    fully_connected(t_unroll, network_size, 'g_0_t', with_bias = True) 
            #fully_connected(l_unroll, network_size, 'g_0_l', with_bias = False)

        
        #z = z.astype(np.float32)
        #H = tf.nn.softplus(U)
        #H = z_scaled

        x = LENGTH
        y = Y_DIM

        p = 4

        #output = H

        #output = tf.reshape(output, [self.batch_size, -1])
        output = self.z

        output = tf.matmul(output, tf.ones([output.get_shape()[1], (LENGTH//4)*(Y_DIM//4)*16]))
        #output = linear(output, (LENGTH//4)*(Y_DIM//4)*8, 'g_lin_0')
        #output = linear(output, network_size, 'g_lin_0')
        output = tf.reshape(output, [self.batch_size,  LENGTH//4, Y_DIM//4,16])

        output = deconv2d(output, [self.batch_size,  LENGTH//4, Y_DIM//4,8], name='g_d_1', d_h=1, d_w=1)
        output = tf.nn.tanh(output)
        output = tf.nn.dropout(output, self.keep_prob)
        output = deconv2d(output, [self.batch_size, LENGTH//2, Y_DIM//2, 4], name='g_d_2')
        output = tf.nn.tanh(output)
        output = tf.nn.dropout(output, self.keep_prob)
        ##output = tf.nn.dropout(output, self.keep_prob)
        ##output = self.g_bn0(output)
        output = deconv2d(output, [self.batch_size,  LENGTH, Y_DIM,1], name='g_d_15')
        ##output = tf.nn.dropout(output, self.keep_prob)
        ##output = tf.nn.tanh(output)
        #output = linear(output, 20*256, 'g_d_lin')


        #output = self.g_bn1(output)
        #output = tf.nn.tanh(deconv2d(output, [self.batch_size, p*8, p*8, 8], name='g_d_3'))
        #output = tf.nn.dropout(output, self.keep_prob)
        #output = self.g_bn2(output)
        #output = deconv2d(output, [self.batch_size, p*16, p*16, 1], name='g_d_4')
        #output = tf.nn.dropout(output, self.keep_prob)
        #output = self.g_bn3(output)

        scribe = tf.matmul(self.z, tf.ones([self.z.get_shape()[1], 32]))
        filter = tf.get_variable('g_filter', [self.batch_size, 32])
        softmax = tf.nn.softmax(filter)
        #softmax = tf.maximum(softmax, 1)
        scribe = scribe * softmax
        scribbler = linear(scribe, Y_DIM*LENGTH, 'g_lin_scribe')
        scribbler = tf.reshape(scribbler, [self.batch_size, LENGTH, Y_DIM, 1])
        output = output + scribbler
        #output = tf.reshape(output, [self.batch_size, -1])
        #print("Deconv out", output)
        #output = fully_connected(output, network_size, 'g_proj_start2', with_bias=True)
        #output = lrelu(output)
        #output = fully_connected(output, network_size, 'g_proj_start3', with_bias=True)
        #output = lrelu(output)
        #output = fully_connected(output, network_size, 'g_proj_start4', with_bias=True)
        #output = tf.nn.softmax(output)


        #output = fully_connected(output, LENGTH*self.z_dim, "g_z2_out")
        #output = output 
        #output = tf.nn.sigmoid(output)
        #output = tf.nn.tanh(output)

        #output = fully_connected(output, network_size, "g_fc_out")
        output = tf.nn.tanh(output)
        #output = fully_connected(output, Y_DIM*LENGTH, "g_fc_out2")
        #output = tf.nn.dropout(output, self.keep_prob)
        #output = tf.nn.tanh(lstm.generator(output, Y_DIM*LENGTH, 'g_lstm4'))
        #output = tf.nn.relu(output)
        #output = fully_connected(output, Y_DIM*LENGTH, 'g_proj_end', with_bias=False)
        #output = tf.reshape(output, [self.batch_size, Y_DIM, LENGTH])

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
    def coordinates(self, t_dim=64):
        t_range = (np.arange(t_dim)-(t_dim-1)/2.0)/(t_dim-1)/0.5
        t_mat = np.tile(t_range, self.batch_size).reshape(self.batch_size, t_dim)
        return t_mat

