#!/usr/bin/env python
# coding: utf-8
import torch
from torch.utils import data
import h5py
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
    
class MaxwellBatchLoader:

    """Iterator that counts upward forever."""

    def __init__(self, file_path, train_size, batch_size, shuffle=True, transform=None):
        
        self.file_path = file_path
        self.transform = transform
        self.hdf5file = h5py.File(self.file_path, 'r')
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        if train_size > self.hdf5file['hcal_only']['layers'].shape[0]-1:
            self.train_size = self.hdf5file['hcal_only']['layers'].shape[0]-1
        else:
            self.train_size = train_size
            
        self.train_size = (self.train_size//self.batch_size)*self.batch_size
        self.dataset = np.zeros((self.train_size, 1))

        self.batch_indices = np.arange(start=0, stop=train_size+1, step=batch_size)
        if self.shuffle:
            np.random.shuffle(self.batch_indices)

        self.index = 0
            
    def __len__(self):
        return self.train_size
            
    def get_data(self, i):
        return self.hdf5file['hcal_only']['layers'][i]
    
    def get_energy(self, i):
        return self.hdf5file['hcal_only']['energy'][i]

    def get_data_range(self, i, j):
        return self.hdf5file['hcal_only']['layers'][i:j]
    
    def get_energy_range(self, i, j):
        return self.hdf5file['hcal_only']['energy'][i:j]


    def __iter__(self):
        return self

    def __next__(self):
        # get data
        if self.index+1 >= len(self.batch_indices):
            if self.shuffle:
                np.random.shuffle(self.batch_indices)
            self.index = 0
            raise StopIteration
        
        x = self.get_data_range(self.batch_indices[self.index], self.batch_indices[self.index]+self.batch_size)
        e = self.get_energy_range(self.batch_indices[self.index], self.batch_indices[self.index]+self.batch_size)
        
        if self.transform:
            x = torch.from_numpy(self.transform(x)).float()
        else:
            x = torch.from_numpy(x).float()
        e = torch.from_numpy(e)
        
        self.index += 1


            
        return x, e
    
    
    
    
class MaxwellBatchLoader_Angular:

    """Iterator that counts upward forever."""

    def __init__(self, file_path, train_size, batch_size, shuffle=True, transform=None):
        
        self.file_path = file_path
        self.transform = transform
        self.hdf5file = h5py.File(self.file_path, 'r')
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        if train_size > self.hdf5file['ecal']['layers'].shape[0]-1:
            self.train_size = self.hdf5file['ecal']['layers'].shape[0]-1
        else:
            self.train_size = train_size
            
        self.train_size = (self.train_size//self.batch_size)*self.batch_size
        self.dataset = np.zeros((self.train_size, 1))

        self.batch_indices = np.arange(start=0, stop=train_size+1, step=batch_size)
        if self.shuffle:
            np.random.shuffle(self.batch_indices)

        self.index = 0
            
    def __len__(self):
        return self.train_size
            
    def get_data(self, i):
        return self.hdf5file['ecal']['layers'][i]
    
    def get_energy(self, i):
        return self.hdf5file['ecal']['energy'][i]
    
    def get_theta(self, i):
        return self.hdf5file['ecal']['theta'][i]

    def get_data_range(self, i, j):
        return self.hdf5file['ecal']['layers'][i:j]
    
    def get_energy_range(self, i, j):
        return self.hdf5file['ecal']['energy'][i:j]
    
    def get_theta_range(self, i, j):
        return self.hdf5file['ecal']['theta'][i:j]


    def __iter__(self):
        return self

    def __next__(self):
        # get data
        if self.index+1 >= len(self.batch_indices):
            if self.shuffle:
                np.random.shuffle(self.batch_indices)
            self.index = 0
            raise StopIteration
        
        x = self.get_data_range(self.batch_indices[self.index], self.batch_indices[self.index]+self.batch_size)
        e = self.get_energy_range(self.batch_indices[self.index], self.batch_indices[self.index]+self.batch_size)
        t = self.get_theta_range(self.batch_indices[self.index], self.batch_indices[self.index]+self.batch_size)
        
        if self.transform:
            x = torch.from_numpy(self.transform(x)).float()
        else:
            x = torch.from_numpy(x).float()
        e = torch.from_numpy(e)
        t = torch.from_numpy(t)
        
        self.index += 1


            
        return x, e, t