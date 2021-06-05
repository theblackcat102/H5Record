import unittest
import os
import numpy as np
import torch

class TestDataModality(unittest.TestCase):


    # def test_pow(self):
    #     import h5py
    #     import numpy as np
    #     if os.path.exists('temp.h5'):
    #         os.remove('temp.h5')
    #     f = h5py.File('temp.h5','w')
    #     float32_t = h5py.special_dtype(vlen=np.dtype('float32'))
    #     evolutionary_ = f.create_dataset('evolutionary', shape=(1, 3, 3, ), maxshape=(None, 3,3, ), dtype=float32_t)
    #     a = np.random.randn(1, 3, 3, 4)
    #     b = np.random.randn(1, 3, 3, 6)

    #     evolutionary_[0] = a

    #     evolutionary_.resize(3, axis=0)
    #     evolutionary_[1] = b

    #     f = h5py.File('temp.h5','r')
    #     print( np.vstack(f['evolutionary'][0]).shape )

    #     assert np.stack(f['evolutionary'][0], axis=0) == (3, 32, 32, 4)
    #     assert f['evolutionary'][1] == (3, 32, 32, 6)


    def test_gif_based_schema(self):
        from h5record.dataset import H5Dataset
        from h5record.attributes import ImageSequence
        gif_attr = ImageSequence(name='gif', h=32, w=32)
        schema = [
            gif_attr
        ]
        data_size = 1

        gif_paths = ['test/gif/rotate_earth.gif']*data_size

        def pair_iter():
            for (gif_path) in gif_paths:

                yield {
                    'gif': gif_attr.read_gif(gif_path),
                }
        if os.path.exists('gif_dataset.h5'):
            os.remove('gif_dataset.h5')

        dataset = H5Dataset(schema, './gif_dataset.h5', pair_iter())
        for idx in range(data_size):
            # Currently this returns matrix of 3 x 32 x 32 x np.array 
            # which is treated as numpy.object_

            # solution include :
            # 1. costly reshape
            # 2. flatten the 4D matrix as one large 1D matrix which is suitable variable array
            gif = dataset[idx]['gif']
            assert gif.shape == (3, 32, 32, 44)

        assert len(dataset) == data_size

        os.remove('gif_dataset.h5')

