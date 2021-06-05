import h5py as h5
import numpy as np
from torch.utils.data.dataset import Dataset
import os
from .attributes import (
    String, ImageSequence
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
        transform=None, append_mode=False, verbose=0):

        '''
        Note: 
            * data length must be known value otherwise chunk size will not be enabled
            * chunk size affects reading speed, usually a size of 100-500 is suitable value
            * compression algorithm affects reading speed, so if storage is not your concern is recommended not to enable
        '''

        # normalized schema design to dictionary
        if isinstance(schema, list) or isinstance(schema, tuple):
            schema = {  s.name: s  for s in schema }
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
        self.num_entries = self.reader[first_key].shape[0]


    def preprocess(self, data_iter):
        idx = 0
        for data in data_iter:
            if idx == 0:
                with h5.File(self.save_filename, 'w', libver='latest', swmr=True) as fout:
                    fout.swmr_mode = True 
                    for key, value in data.items():
                        attribute = self.schema[key]
                        attribute.init_attributes(fout, value, 
                            self.compression, self.data_length)
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
            attribute = self.schema[key]
            if isinstance(attribute, String):
                data[key] = raw_output[0].decode(attribute.encoding )
            elif isinstance(attribute, ImageSequence):
                # heavy reshaping is needed as variable length dimension (last dimension)
                # is always treated as np.array
                # rendering high dimension shape becomes a np.object matrix
                # 
                data[key] = raw_output[0].reshape(
                    attribute.img_channel, attribute.w, attribute.h, -1  )
            else:
                data[key] = raw_output

        if self.transform is not None:
            return self.transform(data)
        
        return data


