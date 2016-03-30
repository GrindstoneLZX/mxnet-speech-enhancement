# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 20:53:56 2016

@author: zxlee
"""

import numpy as np
import h5py
import mxnet as mx
import threading
import Queue
import copy

class simple_batch(object):
    def __init__(self, data_names, data, label_names, label, bucket_key):
        self.data = data
        self.label = label
        self.data_names = data_names
        self.label_names = label_names
        self.bucket_key = bucket_key
        
    @property
    def provide_data(self):
        return [(n, x.shape) for n, x in zip(self.data_names, self.data)]
        
    @property
    def provide_label(self):
        return [(n, x.shape) for n, x in zip(self.label_names, self.label)]


# multi-thread data iterator in sequence     
class multi_thread_iterator(mx.io.DataIter):
    def __init__(self, input_file_name, label_file_name, init_states, buckets, 
                 input_name='noisy_spec', label_name='clean_spec', 
                 parallel_num=10, batch_size=512, shuffle=False, queue_size=100):
        
        self.input_file_name = input_file_name
        self.label_file_name = label_file_name
        self.input_name = input_name
        self.label_name = label_name
        self.parallel_num = parallel_num
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.buckets = buckets
        self.buckets.sort()

        self.default_bucket_key = buckets[0]
        
        ''' check utterances' length '''
        file_handle = h5py.File(self.input_file_name, 'r')
        self.utt_num = len(file_handle.keys()) if file_handle.get('std') == None \
                        else len(file_handle.keys())-1
        self.input_fea_len = file_handle['0'].shape[1]
        self.utt_lens = [file_handle['%d' % i].shape[0] for i in range(self.utt_num)]
        file_handle.close()
        
        file_handle = h5py.File(self.label_file_name, 'r')
        label_utt_num = len(file_handle.keys()) if file_handle.get('std') == None \
                        else len(file_handle.keys())-1
        assert label_utt_num == self.utt_num
        
        self.label_fea_len = file_handle['0'].shape[1]
        file_handle.close()        
        
        ''' prepare utt index in each bucket '''
        # save utt index in each bucket
        self.utt_index_bucket = [[] for _ in self.buckets]
        for iutt,utt_len in enumerate(self.utt_lens):
            for ibucket in range(len(self.buckets)-1, -1, -1):
                if self.buckets[ibucket-1] < utt_len or ibucket == 0:
                    self.utt_index_bucket[ibucket].append(iutt)
                    break
        
        print("Summary of dataset ==================")
        for i in range(len(self.utt_index_bucket)):
            print("bucket of len %3d : %d samples" % (self.buckets[i], len(self.utt_index_bucket[i])))
            
        # random shuffle each bucket
        if shuffle:
            for bucket in self.utt_index_bucket:
                np.random.shuffle(bucket)
                
        ''' for mxnet interface '''
        self.init_states = init_states
        self.init_state_arrays = [mx.nd.zeros(x[1]) for x in init_states]
        self.init_state_names = [x[0] for x in self.init_states]
        self.provide_data = [('%s/%d' % (self.input_name, t), (self.batch_size,))
                             for t in range(self.default_bucket_key)] + init_states
        self.provide_label = [('%s/%d' % (self.label_name, t), (self.batch_size,))
                              for t in range(self.default_bucket_key)]
        
        ''' for prefetch thread '''
        self.queue = Queue.Queue(maxsize=queue_size)
        self.sentinel = object()
        self.prefetch_switch = threading.Event()
        self.prefetch_switch.clear()
        
        self.bucket_indices = np.arange(len(self.buckets))
        self.bucket_fetch_index = 0
        if shuffle:
            np.random.shuffle(self.bucket_indices)
        self.bucket_fetch_finish = 0
        
        self.load_chunk_pair = [np.empty((0, self.input_fea_len*self.parallel_num), dtype='float32'), \
                                np.empty((0, self.label_fea_len*self.parallel_num), dtype='float32')]
                                
        # load one chunk according to index
        def load_chunk_func(self, index, out_id, file_name, bucket):
            assert len(index) <= self.parallel_num
            handle = h5py.File(file_name, 'r')
            
            fea_len = handle['0'].shape[1]
            self.load_chunk_pair[out_id] = np.zeros((bucket, fea_len * self.parallel_num), dtype='float32')
            for i in range(len(index)):
                utt_len = handle['%d'%index[i]].shape[0]
                self.load_chunk_pair[out_id][:utt_len, i*fea_len:(i+1)*fea_len] = handle['%d'%index[i]][:]
            
            handle.close()
            
        def prefetch_func(self):
            while True:
                self.prefetch_switch.wait()
                # utterance indices of this bucket
                this_bucket_utt_indices = self.utt_index_bucket[self.bucket_indices[self.bucket_fetch_index]]
                # bucket size of this bucket
                this_bucket = self.buckets[self.bucket_indices[self.bucket_fetch_index]]
                # utterance index for prefetch
                index = this_bucket_utt_indices[self.bucket_fetch_finish:self.bucket_fetch_finish+self.parallel_num]
                
                load_chunk_thread = [threading.Thread(target=load_chunk_func, args=(self, index, 0, self.input_file_name, this_bucket,)), \
                                     threading.Thread(target=load_chunk_func, args=(self, index, 1, self.label_file_name, this_bucket,))]
                
                for i in range(2):    
                    load_chunk_thread[i].start()
                for i in range(2):
                    load_chunk_thread[i].join()
                self.queue.put(copy.deepcopy(self.load_chunk_pair))
                
                self.bucket_fetch_finish += self.parallel_num
                
                if self.bucket_fetch_finish >= len(this_bucket_utt_indices):
                    self.bucket_fetch_index += 1
                    self.bucket_fetch_finish = 0
                    
                if self.bucket_fetch_index >= len(self.bucket_indices):
                    self.queue.put(self.sentinel)
                    self.prefetch_switch.clear()
                    
        self.prefetch_thread = threading.Thread(target=prefetch_func, args=(self,))
                
        ''' start '''
        self.prefetch_thread.setDaemon(True)
        self.prefetch_thread.start()
        
    def reset(self):        
        if self.shuffle:
            for bucket in self.utt_index_bucket:
                np.random.shuffle(bucket)
            np.random.shuffle(self.bucket_indices)
        self.bucket_fetch_index = 0
        self.bucket_fetch_finish = 0        
        self.load_chunk_pair = [np.empty((0, self.input_fea_len*self.parallel_num), dtype='float32'), \
                                np.empty((0, self.label_fea_len*self.parallel_num), dtype='float32')]
                                
        self.prefetch_switch.set()
        
    def next(self):
        take_chunk_pair = self.queue.get()
        if take_chunk_pair is self.sentinel:
            raise StopIteration
        else:
            bucket_key = take_chunk_pair[0].shape[0]
            data_all = [mx.nd.array(take_chunk_pair[0][t,:].reshape((self.parallel_num, self.input_fea_len))) \
                        for t in range(bucket_key)] + self.init_state_arrays
            label_all = [mx.nd.array(take_chunk_pair[1][t,:].reshape((self.parallel_num, self.label_fea_len))) \
                        for t in range(bucket_key)]
            data_names = ['%s/%d' % (self.input_name, t) for t in range(bucket_key)] + self.init_state_names
            label_names = ['%s/%d' % (self.label_name, t) for t in range(bucket_key)]
            
            batch = simple_batch(data_names, data_all, label_names, label_all, bucket_key)
            return batch
            
    def __next__(self):
        return self.next(self)
        
    def __iter__(self):
        return self

def get_buckets(lens, bucket_num):
    temp_lens = np.array(lens)
    temp_lens.sort()
    sample_index = np.linspace(0, len(temp_lens)-1, num=bucket_num+1, dtype='int')
    bucket = [temp_lens[i] for i in sample_index]
    return bucket[1:]

        
if __name__ == '__main__':
    import time
    input_file_name = '/home/zxlee/Documents/SpeechEnhance_xuyong/test1_noisy_spec.h5'
    label_file_name = '/home/zxlee/Documents/SpeechEnhance_xuyong/test1_clean_spec.h5'
    batch_size = 10
    num_hidden = 100
    num_lstm_layer = 2
    init_c = [('l%d_init_c'%l, (batch_size, num_hidden)) for l in range(num_lstm_layer)]
    init_h = [('l%d_init_h'%l, (batch_size, num_hidden)) for l in range(num_lstm_layer)]
    init_states = init_c + init_h
    
    file_handle = h5py.File(input_file_name, 'r')
    utt_num = len(file_handle.keys()) if file_handle.get('std') == None \
                    else len(file_handle.keys())-1
    utt_lens = [file_handle['%d' % i].shape[0] for i in range(utt_num)]
    file_handle.close()
    
    buckets = get_buckets(utt_lens, 5)
    
    iterator = multi_thread_iterator(input_file_name, label_file_name, shuffle=True,
                                     init_states=init_states, buckets=buckets)

    for epoch in range(2):    
        iterator.reset()   
        tic = time.time()
        for i, batch in enumerate(iterator):
            data = batch.data
            data_names = batch.data_names
            print (i, len(batch.label), batch.data_names[0], batch.data[0].shape)
        print 'escape time: %.3f' % (time.time() - tic)