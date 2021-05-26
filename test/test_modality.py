import unittest
import os

class TestDataModality(unittest.TestCase):


    def test_string_pair(self):
        from h5record.dataset import H5Dataset
        from h5record.attributes import String
        schema = {
            'sentence1': String(name='sentence1'),
            'sentence2': String(name='sentence2')
        }
        data = [
            ['HDF5 supports chunking and compression.', 'You may want to experiment'],
            ['Starting to load file into an HDF5 file with chunk size','and compression is gzip' ],
            ['Reading from an HDF5 file which you will probably be','about to overwrite! Override this error only if you know what youre doing '],
        ]

        def pair_iter():
            for row in data:
                yield {
                    'sentence1': row[0],
                    'sentence2': row[1]
                }

        dataset = H5Dataset(schema, './question_pair.h5', pair_iter())
        for idx in range(len(data)):
            row = dataset[idx]
            sent1, sent2 = data[idx]
            assert sent1 == row['sentence1']
            assert sent2 == row['sentence2']
        os.remove('question_pair.h5')


        dataset = H5Dataset(schema, './question_pair.h5', pair_iter(),
            data_length=len(data), chunk_size=4)

        for idx in range(len(data)):
            row = dataset[idx]
            sent1, sent2 = data[idx]
            assert sent1 == row['sentence1']
            assert sent2 == row['sentence2']

        os.remove('question_pair.h5')

    def test_image_string_pair(self):
        from h5record.dataset import H5Dataset
        from h5record.attributes import String, Image

        schema = {
            'image': Image(name='image', h=32, w=32),
            'caption': String(name='caption')
        }

        captions = [
            'Lenna profile',
            'Lenna back patch',
            'Lenna lower patch',
            'meme image'
        ]

        image_path = [
            'test/images/1.jpeg',
            'test/images/2.jpeg',
            'test/images/3.jpeg',
            'test/images/4.jpeg',
        ]

        