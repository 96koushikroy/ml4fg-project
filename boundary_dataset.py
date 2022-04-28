import torch
import torch.utils.data
import h5py
import numpy as np

LETTERTOINDEX = {'A': 0, 'C': 1, 'G': 2, 'T':3, 'N':4}
INDEXTOLETTER = {0: 'A', 1: 'C', 2: 'G', 3: 'T', 4: 'N'}

class AnchorDataset(torch.utils.data.Dataset):

    def __init__(self, boundary_file_templ, label_file_templ, rnn_len, length, argmax=False):
        super().__init__()
        self.rnn_len = rnn_len
        self.boundary_file_templ = boundary_file_templ
        self.label_file_templ = label_file_templ
        self.length = length
        self.argmax = argmax
        
    def __len__(self):
        return self.length
    
    def open_hdf5(self):
        # wid = torch.utils.data.get_worker_info().id
        # boundary_file = self.boundary_file_templ.format(wid)
        # label_file = self.label_file_templ.format(wid)
        boundary_file = self.boundary_file_templ
        label_file = self.label_file_templ
        
        self.boundaries = h5py.File(boundary_file, 'r')
        self.labels = h5py.File(label_file, 'r') if self.label_file_templ is not None else None
        self.shard_length = len(self.boundaries)
    
    def __getitem__(self, idx):
        if not hasattr(self, 'boundaries'):
            self.open_hdf5()
                
        key = str(idx % self.shard_length)
        region = self.boundaries[key][:]

        # Convert one-hot to token encoding for transformers
        if self.argmax:
            region = region.argmax(axis=1)
            # region = "".join(np.vectorize(INDEXTOLETTER.get)(region))
        
        trunc = (len(region) - self.rnn_len) // 2
        rnn_region = region[trunc:-trunc]
        
        if self.labels is not None:
            label = self.labels[key][()]
            return region, rnn_region, label
        else:
            return region, rnn_region
    
    def __del__(self):
        if hasattr(self, 'boundaries'):
            self.boundaries.close()
        if hasattr(self, 'labels') and self.labels is not None:
            self.labels.close()

class AnchorCollate:

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def __call__(self, inp):
        region, rnn_region, label = [x[0] for x in inp], [x[1] for x in inp], [x[2] for x in inp]
        tokens = self.tokenizer(region, padding="max_length", truncation=True, return_tensors='pt')
        return tokens, torch.Tensor([0]), torch.Tensor(label)
