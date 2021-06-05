import unittest
import os
from h5record.dataset import H5Dataset

class TestDataModality(unittest.TestCase):


    def test_list_based_schema(self):
        from h5record.attributes import String, Integer, Image

        img_attr = Image(name='image', h=32, w=32)

        schema = [
            img_attr,
            Integer(name='label'),
            String(name='sentence1'),
            String(name='sentence2'),
        ]
        image_paths = [
            'test/images/1.jpeg',
            'test/images/2.jpeg',
            'test/images/3.jpeg',
        ]

        data = [
            ['HDF5 supports chunking and compression.', 'You may want to experiment', 0 ],
            ['Starting to load file into an HDF5 file with chunk size','and compression is gzip', 1 ],
            ['Reading from an HDF5 file which you will probably be','about to overwrite! Override this error only if you know what youre doing ', 1],
        ]

        def pair_iter():
            for (row, image_path) in zip(data, image_paths):
                yield {
                    'sentence1': row[0],
                    'sentence2': row[1],
                    'image': img_attr.read_image(image_path),
                    'label': row[2]
                }
        if os.path.exists('question_image_pair.h5'):
            os.remove('question_image_pair.h5')

        dataset = H5Dataset(schema, './question_image_pair.h5', pair_iter())
        for idx in range(len(data)):
            row = dataset[idx]
            sent1, sent2, label = data[idx]
            assert sent1 == row['sentence1']
            assert sent2 == row['sentence2']
            assert label == row['label']

        os.remove('question_image_pair.h5')

        dataset = H5Dataset(schema, './question_image_pair.h5', pair_iter(),
            data_length=len(data), chunk_size=4)

        for idx in range(len(data)):
            row = dataset[idx]
            sent1, sent2, label = data[idx]
            assert sent1 == row['sentence1']
            assert sent2 == row['sentence2']
            assert label == row['label']

        os.remove('question_image_pair.h5')

    def test_tuple_based_schema(self):
        from h5record.attributes import String, Integer, Image

        img_attr = Image(name='image', h=32, w=32)

        schema = ( 
            String(name='sentence1'),
            String(name='sentence2'),
            img_attr,
            Integer(name='label')
        )
        image_paths = [
            'test/images/1.jpeg',
            'test/images/2.jpeg',
            'test/images/3.jpeg',
        ]

        data = [
            ['HDF5 supports chunking and compression.', 'You may want to experiment', 0 ],
            ['Starting to load file into an HDF5 file with chunk size','and compression is gzip', 1 ],
            ['Reading from an HDF5 file which you will probably be','about to overwrite! Override this error only if you know what youre doing ', 1],
        ]

        def pair_iter():
            for (row, image_path) in zip(data, image_paths):
                yield {
                    'sentence1': row[0],
                    'sentence2': row[1],
                    'image': img_attr.read_image(image_path),
                    'label': row[2]
                }
        if os.path.exists('question_image_pair.h5'):
            os.remove('question_image_pair.h5')

        dataset = H5Dataset(schema, './question_image_pair.h5', pair_iter())
        for idx in range(len(data)):
            row = dataset[idx]
            sent1, sent2, label = data[idx]
            assert sent1 == row['sentence1']
            assert sent2 == row['sentence2']
            assert label == row['label']

        os.remove('question_image_pair.h5')

        dataset = H5Dataset(schema, './question_image_pair.h5', pair_iter(),
            data_length=len(data), chunk_size=4)

        for idx in range(len(data)):
            row = dataset[idx]
            sent1, sent2, label = data[idx]
            assert sent1 == row['sentence1']
            assert sent2 == row['sentence2']
            assert label == row['label']

        os.remove('question_image_pair.h5')


    def test_dict_based_schema(self):
        from h5record.attributes import String, Image

        schema = {
            'image': Image(name='image', h=32, w=32),
            'caption': String(name='caption')
        }

        captions = [
            'Lenna profile',
            'Lenna back patch',
            'Lenna lower patch',
            'meme image', 
            'greyscale image'
        ]
        image_path = [
            'test/images/1.jpeg',
            'test/images/2.jpeg',
            'test/images/3.jpeg',
            'test/images/4.jpeg',
            'test/images/5.jpeg',
        ]
        def pair_iter():
            for (caption, img_path) in zip(captions, image_path):
                yield {
                    'image': schema['image'].read_image(image_path[0]),
                    'caption': caption
                }
        if os.path.exists('img_caption.h5'):
            os.remove('img_caption.h5')

        dataset = H5Dataset(schema, './img_caption.h5', pair_iter(),
            data_length=len(image_path), chunk_size=4)

        for idx in range(len(image_path)):
            data = dataset[idx]
            assert data['caption'] == captions[idx]

        os.remove('img_caption.h5')


        dataset = H5Dataset(schema, './img_caption.h5', pair_iter(),
            compression='lzf')

        for idx in range(len(image_path)):
            data = dataset[idx]
            assert data['caption'] == captions[idx]

        os.remove('img_caption.h5')



    def test_seq_schema(self):
        '''
            Suitable for tokenized sequences
        '''
        from h5record.attributes import Sequence, FloatSequence, Float16Sequence
        import numpy as np

        schema = ( 
            Sequence(name='seq1'),
            Sequence(name='seq2')
        )

        data = [
            [  np.array([ 0, 1, 2, 3 ]), np.array([  1, 2, 3 ])  ],
            [  np.array([ 0, 1, 2, 3, 4, 5 ]), np.array([  1, 2, 3, 3 ])  ],
            [  np.array([ 0, 1, 2 ]), np.array([  1, 2, -1 ])  ],
        ]
        def pair_iter():
            for (seq1, seq2) in data:
                yield {
                    'seq1': seq1,
                    'seq2': seq2
                }
        if os.path.exists('tokens.h5'):
            os.remove('tokens.h5')

        dataset = H5Dataset(schema, './tokens.h5', pair_iter(),
            data_length=len(data), chunk_size=4)

        for idx in range(len(data)):
            row = dataset[idx]
            assert (row['seq1'][0] == data[idx][0]).all()

        os.remove('tokens.h5')

        data = [
            [  np.array([ 0.1, 1, 2, 3 ]), np.array([  1, 2.13, 3 ])  ],
            [  np.array([ 0.11, 1, 2, 3, 4, 5 ]), np.array([  1, 2, 3.33333333, 3 ])  ],
            [  np.array([ 3.14159, 1, 2 ]), np.array([  1.988, 2, -1 ])  ],
        ]
        schema = ( 
            FloatSequence(name='seq1'),
            FloatSequence(name='seq2')
        )
        if os.path.exists('tokens.h5'):
            os.remove('tokens.h5')

        dataset = H5Dataset(schema, './tokens.h5', pair_iter(),
            data_length=len(data), chunk_size=4)

        for idx in range(len(data)):
            row = dataset[idx]
            assert (row['seq1'][0] - data[idx][0]).sum() < 1e-6
            assert (row['seq2'][0] - data[idx][1]).sum() < 1e-6

        os.remove('tokens.h5')

        schema = ( 
            Float16Sequence(name='seq1'),
            Float16Sequence(name='seq2')
        )
        if os.path.exists('tokens.h5'):
            os.remove('tokens.h5')

        dataset = H5Dataset(schema, './tokens.h5', pair_iter(),
            data_length=len(data), chunk_size=4)

        for idx in range(len(data)):
            row = dataset[idx]
            assert (row['seq1'][0] - data[idx][0]).mean() < 1e-3
            assert (row['seq2'][0] - data[idx][1]).mean() < 1e-3

        os.remove('tokens.h5')

