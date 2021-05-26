import h5py as h5
import numpy as np
from torch.utils.data.dataset import Dataset
import os
from .attributes import (
    String
)

class AtomicFile:
    '''
        Wrapper file for h5 in case multiprocess writes is needed
    '''
    def __init__(self, path):
        self.fd = os.open(path, os.O_RDONLY)
        self.pos = 0

    def seek(self, pos, whence=0):
        if whence == 0:
            self.pos = pos
        elif whence == 1:
            self.pos += pos
        else:
            self.pos = os.lseek(self.fd, pos, whence)
        return self.pos

    def tell(self):
        return self.pos

    def read(self, size):
        b = os.pread(self.fd, size, self.pos)
        self.pos += len(b)
        return b

class H5Dataset(Dataset):

    def __init__(self, schema, save_filename, data_iter=None,
        data_length=None, chunk_size=300, compression=None, 
        transform=None, append_mode=False):

        '''
        Note: 
            * data length must be known value otherwise chunk size will not be enabled
            * chunk size affects reading speed, usually a size of 100-500 is suitable value
            * compression algorithm affects reading speed, so if storage is not your concern is recommended not to enable
        '''
        self.schema = schema
        self.save_filename = save_filename
        self.data_length = data_length # dataset maximum size
        self.transform = transform # transform function before returned by index access
        self.append_mode = append_mode # force append ?

        self.chunk_size = None if data_length is None else chunk_size
        assert compression in [None, 'lzf', 'gzip', 'szip']
        self.compression = compression
        if not os.path.exists(self.save_filename):
            self.preprocess(data_iter)

        self.reader = h5.File(self.save_filename, 'r')
        first_key = list(self.schema.keys())[0]
        self.num_entries = self.reader[first_key].shape[0]-1


    def preprocess(self, data_iter):
        idx = 0
        for data in data_iter:
            if idx == 0:
                with h5.File(self.save_filename, 'w', libver='latest', swmr=True) as fout:
                    fout.swmr_mode = True 
                    for key, value in data.items():
                        attribute = self.schema[key]
                        value = attribute.transform(value)
                        max_shape = list(attribute.max_shape)
                        max_shape[0] = self.data_length
                        max_shape = tuple(max_shape)
                        shape = (len(value), 1)
                        if not isinstance(attribute, String):
                            shape = value.shape
                        dset = fout.create_dataset(key,data=value, shape=shape, maxshape=max_shape, dtype=attribute.dtype )
            else:
                with h5.File(self.save_filename, 'a', libver='latest', swmr=True) as fout:
                    fout.swmr_mode = True 
                    for key, value in data.items():
                        attribute = self.schema[key]
                        value = attribute.transform(value)
                        attribute.append(fout, value)

            idx += 1

    def __len__(self):
        return self.num_entries


    def __getitem__(self, idx):
        data = {}
        for key in self.schema.keys():
            raw_output = self.reader[key][idx]
            if isinstance(self.schema[key], String):
                data[key] = raw_output[0].decode(self.schema[key].encoding )
            else:
                data[key] = raw_output

        if self.transform is not None:
            return self.transform(data)
        
        return data


