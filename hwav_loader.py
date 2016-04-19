from multiprocessing import Queue, Process
import tensorflow_wav
import hwav
import numpy as np
import time
CHANNELS=1
LENGTH = 1024
queue = Queue()
def load(files, batch_size):
    p = Process(target=add_to_queue, args=([files, batch_size]))
    p.start()
    
def get_batch(filee, batch_size):

   #print("Loading", filee)

   # Size is [-1, 2]
   out = tensorflow_wav.get_wav(filee)
   out = np.reshape(out['data'], [-1])
   out = out[:(len(out)//batch_size//LENGTH//2)*2*LENGTH*batch_size]
   out = np.reshape(out, [-1, batch_size, LENGTH, CHANNELS])
   out = np.swapaxes(out, 2, 3)
   return out
   
def add_to_queue(files,batch_size):
    for filea in files:
        #add batch to queue

       #scipy.misc.imsave("visualize/input-full.png", data_left[:Y_DIM])
       batches = get_batch(filea, batch_size)#, get_batch(fileb)]
       i=0
       for batch in batches:
           while(queue.qsize() > 100):
               time.sleep(0.1)
           queue.put([batch, i/len(batches[0]), 1.0/batch_size])
           i+=1
       time.sleep(0.1)
    queue.put("DONE")


def next_batch():
    pop = queue.get()
    if(pop == "DONE"):
        return None
    else:
        return pop
