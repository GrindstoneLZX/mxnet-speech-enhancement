# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 14:07:22 2016

@author: zxlee
"""

import numpy as np
import h5py
import mxnet as mx
import threading
import Queue
import copy

class simple_batch(object):
    def __init__(self, input_arrays, label_arrays, input_names, label_names):
        self.data = input_arrays
        self.label = label_arrays
        self.data_names = input_names
        self.label_names = label_names
        
    @property
    def provide_data(self):
        return [(n, x.shape) for n, x in zip(self.data_names, self.data)]
#        return [(self.data_names, self.data.shape)]
        
    @property
    def provide_label(self):
        return [(n, x.shape) for n, x in zip(self.label_names, self.label)]
#        return [(self.label_names, self.label.shape)]
    
                
def load_chunk(utt_idx, file_name, splice=2):    
    
    handle = h5py.File(file_name, 'r')    
    fea_len = handle['0'].shape[1]
    
    cur = sum([handle['%d' % utt].shape[0] for utt in utt_idx])
    
    #print cur
    chunk = np.empty(shape=(cur, (2*splice+1)*fea_len), dtype='float32')
    
    cur = 0    
    for utt in utt_idx:
        data = np.pad(handle['%d' % utt][:], ((splice,splice),(0,0)), mode='edge')
        utt_len = data.shape[0] - 2*splice
        for frame in range(0, 2*splice+1):
            chunk[cur:cur+utt_len, frame*fea_len:(frame+1)*fea_len] = data[frame:frame+utt_len, :]
        cur += utt_len
        
    handle.close()
    return chunk                                                              

# multi-thread data iterator with splice 
class multi_thread_iterator(mx.io.DataIter):
    def __init__(self, input_file_name, label_file_name, input_name='noisy_spec', label_name='clean_spec', 
                 splice=2, chunk_size=1000, batch_size=512, shuffle=False, queue_size=5, view_fea_num=False):
        
        self.input_file_name = input_file_name
        self.label_file_name = label_file_name
        self.input_name = input_name
        self.label_name = label_name
        self.splice = splice
        self.chunk_size = chunk_size
        self.batch_size = batch_size

        self.shuffle = shuffle
        
        ''' check utterance and frame number '''
        file_handle = h5py.File(self.input_file_name, 'r')
        self.utt_num = len(file_handle.keys()) if file_handle.get('std') == None \
                        else len(file_handle.keys())-1
        self.input_fea_len = file_handle['0'].shape[1] * (2*self.splice+1)
        self.label_fea_len = file_handle['0'].shape[1]
        if view_fea_num:
            self.fea_num = sum([file_handle['%d' % i].shape[0] for i in range(self.utt_num)])
            print 'Summary of dataset =================='
            print 'utterance number: %d, frame number: %d' % (self.utt_num, self.fea_num)
        file_handle.close()
        
        ''' for mxnet interface '''
        self.provide_data = [(self.input_name, (self.batch_size, self.input_fea_len))]
        self.provide_label = [(self.label_name, (self.batch_size, self.label_fea_len))]
        
        ''' prefetch chunk '''
        self.queue = Queue.Queue(maxsize=queue_size)
        self.sentinel = object()
        self.prefetch_switch = threading.Event()
        self.prefetch_switch.clear()
        
        
        self.chunk_indices = np.arange(self.utt_num)
        if self.shuffle:
            np.random.shuffle(self.chunk_indices)
        self.load_chunk_pair = [np.empty((0,self.input_fea_len), dtype='float32'), \
                                np.empty((0,self.label_fea_len), dtype='float32')]
        self.chunk_begin = 0
        
        def load_chunk_func(self, index, out_id, file_name, splice):
            #index = self.chunk_indices[self.chunk_begin:self.chunk_begin + self.chunk_size]
            self.load_chunk_pair[out_id] = load_chunk(index, file_name, splice)
        
#        self.load_chunk_thread = [threading.Thread(target=load_chunk_func, args=(self, 0, self.input_file_name, self.splice,)), \
#                                  threading.Thread(target=load_chunk_func, args=(self, 1, self.label_file_name, 0,))]
        
        def prefetch_func(self):
            while True:
                self.prefetch_switch.wait()
                
                index = self.chunk_indices[self.chunk_begin:self.chunk_begin + self.chunk_size]
                load_chunk_thread = [threading.Thread(target=load_chunk_func, args=(self, index, 0, self.input_file_name, self.splice,)), \
                                  threading.Thread(target=load_chunk_func, args=(self, index, 1, self.label_file_name, 0,))]
                for i in range(2):    
                    load_chunk_thread[i].start()
                for i in range(2):
                    load_chunk_thread[i].join()

                self.queue.put(copy.deepcopy(self.load_chunk_pair))
                self.chunk_begin += self.chunk_size
                if self.chunk_begin >= self.utt_num:
                    self.queue.put(self.sentinel)
                    self.prefetch_switch.clear()
                    
        self.prefetch_thread = threading.Thread(target=prefetch_func, args=(self,))
        
        ''' get batch from chunk '''
        self.batch_indices = np.arange(0)
        self.take_chunk_pair = [np.empty((0,self.input_fea_len), dtype='float32'), \
                                np.empty((0,self.label_fea_len), dtype='float32')]
        self.batch_begin = 0
        
        ''' start '''
        self.prefetch_thread.setDaemon(True)
        self.prefetch_thread.start()
        
    def reset(self):
        if self.shuffle:
            np.random.shuffle(self.chunk_indices)
        self.load_chunk_pair = [np.empty((0,self.input_fea_len), dtype='float32'), \
                                np.empty((0,self.label_fea_len), dtype='float32')]
        self.chunk_begin = 0
        
        self.batch_indices = np.arange(0)
        self.take_chunk_pair = [np.empty((0,self.input_fea_len), dtype='float32'), \
                                np.empty((0,self.label_fea_len), dtype='float32')]
        self.batch_begin = 0
        self.prefetch_switch.set()
        
    def next(self):
        ''' if batch_begin + batch_size >= this_chunk len, discard this_chunk '''
        if self.batch_begin + self.batch_size >= self.take_chunk_pair[0].shape[0] :
            self.take_chunk_pair = self.queue.get()
            
            if self.take_chunk_pair is self.sentinel:
                raise StopIteration
            else:
                ''' reset indices '''
                self.batch_indices = np.arange(self.take_chunk_pair[0].shape[0])
                if self.shuffle:
                    np.random.shuffle(self.batch_indices)
                self.batch_begin = 0
        
        index = self.batch_indices[self.batch_begin:self.batch_begin + self.batch_size]
        input_ndarray = [mx.nd.array(self.take_chunk_pair[0][index,:])]
        label_ndarray = [mx.nd.array(self.take_chunk_pair[1][index,:])]
        input_names = [self.input_name]
        label_names = [self.label_name]
        batch = simple_batch(input_ndarray, label_ndarray, input_names, label_names)
        
        self.batch_begin += self.batch_size
        
        return batch
        
    def __next__(self):
        return self.next(self)
        
    def __iter__(self):
        return self
        
#    def __del__(self):
#        self.file.close()

    
if __name__ == '__main__':
    import time
    input_file_name = '/home/zxlee/Documents/SpeechEnhance_xuyong/train1_noisy_spec.h5'
    label_file_name = '/home/zxlee/Documents/SpeechEnhance_xuyong/train1_clean_spec.h5'
    iterator = multi_thread_iterator(input_file_name, label_file_name, shuffle=True)
    iterator.reset()
    
    tic = time.time()
    for i, batch in enumerate(iterator):
        data = batch.data
        print (i, data[0].shape)

    print 'escape time: %.3f' % (time.time() - tic)
