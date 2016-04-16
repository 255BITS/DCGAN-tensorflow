from multiprocessing import Queue, Process
import tensorflow_wav
import hwav
import numpy as np
import time
LENGTH = 20
Y_DIM = 128
queue = Queue()
def load(files, batch_size):
    p = Process(target=add_to_queue, args=([files, batch_size]))
    p.start()
    
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
def add_to_queue(files,batch_size):
   for filea in files:
        #add batch to queue

       #scipy.misc.imsave("visualize/input-full.png", data_left[:Y_DIM])
       batches = [get_batch(filea)]#, get_batch(fileb)]
       splitInto = 8# segments
       amountNeeded = batch_size * Y_DIM
       for i in range(0,len(batches[0])-amountNeeded, batch_size * Y_DIM//splitInto): #  window over the song.  every nn sees every entry. * 2 for left / right speaker
           #if(batch[i][LENGTH-1] == 0.0):
           #   print("reached end of file?")
           #   next
           batcha = np.array(batches[0][i:i+amountNeeded])
           batcha = np.reshape(batcha, [batch_size, Y_DIM, LENGTH])
           #scipy.misc.imsave("visualize/input-"+str(i)+".png", batcha[0][0::2])
           
           while(queue.qsize() > 100):
               time.sleep(0.1)
           queue.put([batcha, i/len(batches[0]), 1.0/batch_size])
       time.sleep(0.1)
   queue.put("DONE")


def next_batch():
    pop = queue.get()
    if(pop == "DONE"):
        return None
    else:
        return pop
