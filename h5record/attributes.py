import h5py as h5
import numpy as np

try:
    from PIL import Image as PImage
    from PIL import ImageSequence as PImageSequence
except ImportError as e:
    pass

class Attribute():

    def append(self, h5, data):
        raise NotImplementedError

    def transform(self, data):
        raise NotImplementedError


    def init_attributes(self, fout, value, compression, data_length):
        value = self.transform(value)
        max_shape = list(self.max_shape)
        max_shape[0] = data_length
        max_shape = tuple(max_shape)

        shape = value.shape
        fout.create_dataset(self.name, data=value, shape=shape, 
            maxshape=max_shape, 
            dtype=self.dtype, 
            compression=compression )


class Integer(Attribute):
    '''
        One dimensional data shape
    '''
    dtype = 'int64'
    def __init__(self, name='label'):
        self.name = name
        self.shape = (None, )
        self.max_shape = (None, )

    def append(self, h5, data):
        h5[self.name].resize( h5[self.name].shape[0]+data.shape[0], axis=0)
        h5[self.name][-data.shape[0]:] = data
        return h5

    def transform(self, data):
        return np.array([data])

class Float(Attribute):
    '''
        One dimensional data shape
    '''
    dtype = 'float32'
    def __init__(self, name='label'):
        self.name = name
        self.shape = (None, )
        self.max_shape = (None, )

    def append(self, h5, data):
        h5[self.name].resize( h5[self.name].shape[0]+data.shape[0], axis=0)
        h5[self.name][-data.shape[0]:] = data
        return h5

    def transform(self, data):
        return np.array([data])

class Float16(Float):
    '''
        One dimensional data shape
    '''
    dtype = 'float16'


class Image(Attribute):

    dtype = 'uint8'
    def __init__(self, h, w, c=3, name='image'):
        self.c = c
        self.h = h
        self.w = w
        self.name = name

        self.shape = (None, self.c, self.h, self.w)
        self.max_shape = (None, self.c, self.h, self.w)

    def read_image(self, filename, resize=True): 
        # read image by file path and return numpy object
        img = PImage.open(filename)
        if resize:
            img  = img.resize((self.w, self.h))
        np_img = np.array(img)
        if len(np_img.shape) == 2:
            np_img = np.expand_dims(np_img, -1)
            np_img = np.repeat(np_img, 3, axis=-1)

        return np.transpose(np_img, (2, 0, 1))

    def append(self, h5, data):
        h5[self.name].resize( h5[self.name].shape[0]+data.shape[0], axis=0)
        h5[self.name][-data.shape[0]:] = data
        return h5

    def transform(self, data):
        data =  np.array(data)
        if len(data.shape) == 3: 
            # ensure data shape is B x C x H x W
            data = np.expand_dims(data, axis=0)
        return data

class ImageSequence(Attribute):
    dtype = h5.special_dtype(vlen=np.dtype('uint8'))
    img_channel = 3

    def __init__(self, h, w, c=3, name='img_seq'):
        self.c = c
        self.h = h
        self.w = w
        self.name = name

        self.shape = (1, 1, )
        self.max_shape = (None, 1, )

    def read_gif(self, filename, resize=True): 
        # read image by file path and return numpy object
        # Channel x H x W x Length
        gif = PImage.open(filename)
        np_images = []
        for frame in PImageSequence.Iterator(gif):
            if resize:
                frame  = frame.resize((self.w, self.h))
            np_img = np.array(frame)
            if len(np_img.shape) == 2:
                np_img = np.expand_dims(np_img, -1)
                np_img = np.repeat(np_img, self.img_channel, axis=-1)
            np_images.append(np_img)
        gif.close()
        np_images = np.stack(np_images)

        return np.transpose(np_images.astype(self.dtype), (3, 1, 2, 0)).flatten()


    def transform(self, data):
        data =  np.array(data)
        if len(data.shape) == 1: 
            # make sure its  1 x C x H x W x sequence length
            data = np.array([data.flatten() ], dtype=self.dtype)
        return data

    def append(self, h5, data):
        h5[self.name].resize( h5[self.name].shape[0]+data.shape[0], axis=0)
        h5[self.name][-1] = data
        return h5

    def init_attributes(self, fout, value, compression, data_length):
        max_shape = self.max_shape
        max_shape = list(self.max_shape)
        max_shape[0] = data_length
        max_shape = tuple(max_shape)

        dset = fout.create_dataset(self.name, 
            shape=self.shape,
            maxshape=max_shape,
            dtype=self.dtype, 
            compression=compression )
        dset[0] = value


class Sequence(Attribute):

    dtype = h5.special_dtype(vlen=np.dtype('int32'))

    def __init__(self, name='sequence', sub_attributes=None):
        self.name = name
        self.shape = (1, 1, )
        self.sub_attributes = sub_attributes
        self.max_shape = (None, 1, )


    def append(self, h5, data):
        if isinstance(data, dict):
            for key in self.sub_attributes:
                np_seq = data[key]
                # np_seq = np.array([np_seq], dtype=self.dtype)
                h5[self.name + '_'+key].resize( h5[self.name + '_'+key].shape[0]+np_seq.shape[0], axis=0)
                h5[self.name + '_'+key][-np_seq.shape[0]:] = np_seq
        elif isinstance(data, np.ndarray):
            h5[self.name].resize( h5[self.name].shape[0]+data.shape[0], axis=0)
            h5[self.name][-data.shape[0]:] = data
        else:
            raise ValueError("invalid data type: {}".format(type(data)))

        return h5


    def transform(self, data):
        if isinstance(data, dict):
            assert self.sub_attributes is not None, "sub attributes not defined"

            for key in self.sub_attributes:
                np_seq = data[key]
                if len(np_seq.shape) == 2:
                    np_seq = np.array([np_seq], dtype=self.dtype)
                    data[key] = np_seq
            return data
        elif isinstance(data, np.ndarray):
            if len(data.shape) == 1:
                # make sure its B x 1 x sequence length
                data = np.array([data], dtype=self.dtype)
            return data
        else:
            raise ValueError("invalid data type: {}".format(type(data)))

    def init_attributes(self, fout, value, compression, data_length):
        max_shape = self.max_shape
        max_shape = list(self.max_shape)
        max_shape[0] = data_length
        max_shape = tuple(max_shape)

        dset = fout.create_dataset(self.name, 
            shape=self.shape,
            maxshape=max_shape,
            dtype=self.dtype, 
            compression=compression )
        dset[0] = value

# hard to define how small float should be
class FloatSequence(Sequence):
    dtype = h5.special_dtype(vlen=np.dtype('float32'))

class Float16Sequence(Sequence):
    dtype = h5.special_dtype(vlen=np.dtype('float16'))


class String(Attribute):

    encoding = 'utf-8'
    dtype = h5.string_dtype(encoding='utf-8')

    def __init__(self, name='string'):
        self.name = name
        self.max_shape = (None, 1)
        self.shape = None

    def append(self, h5, data):
        buf_size = len(data)
        h5[self.name].resize((h5[self.name].shape[0]+buf_size), axis=0)
        h5[self.name][-buf_size:] = data
        return h5

    def transform(self, data):
        assert isinstance(data, str)
        return [data]

    def init_attributes(self, fout, value, compression, data_length):
        value = self.transform(value)
        max_shape = list(self.max_shape)
        max_shape[0] = data_length
        max_shape = tuple(max_shape)

        shape = (len(value), 1)

        fout.create_dataset(self.name, data=value, shape=shape, 
            maxshape=max_shape, 
            dtype=self.dtype, 
            compression=compression )
