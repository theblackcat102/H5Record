import unittest
import os

class TestDataModality(unittest.TestCase):


    def test_string_pair(self):
        from h5record.dataset import H5Dataset
        from h5record.attributes import String, Integer
        schema = {
            'sentence1': String(name='sentence1'),
            'sentence2': String(name='sentence2'),
            'label': Integer(name='label')
        }

        data = [
            ['HDF5 supports chunking and compression.', 'You may want to experiment', 0],
            ['Starting to load file into an HDF5 file with chunk size','and compression is gzip', 1 ],
            ['Reading from an HDF5 file which you will probably be','about to overwrite! Override this error only if you know what youre doing ', 1],
        ]

        def pair_iter():
            for row in data:
                yield {
                    'sentence1': row[0],
                    'sentence2': row[1],
                    'label': row[2]
                }
        if os.path.exists('question_pair.h5'):
            os.remove('question_pair.h5')

        dataset = H5Dataset(schema, './question_pair.h5', pair_iter())

        assert len(dataset) == len(data)

        for idx in range(len(data)):
            row = dataset[idx]
            sent1, sent2, label = data[idx]
            assert sent1 == row['sentence1']
            assert sent2 == row['sentence2']
            assert label == row['label']

        os.remove('question_pair.h5')

        dataset = H5Dataset(schema, './question_pair.h5', pair_iter(),
            data_length=len(data), chunk_size=4)
        assert len(dataset) == len(data)

        for idx in range(len(data)):
            row = dataset[idx]
            sent1, sent2, label = data[idx]
            assert sent1 == row['sentence1']
            assert sent2 == row['sentence2']
            assert label == row['label']

        os.remove('question_pair.h5')


    def test_unicode(self):
        from h5record.dataset import H5Dataset
        from h5record.attributes import String, Integer
        schema = {
            'sentence1': String(name='sentence1'),
            'sentence2': String(name='sentence2'),
            'label': Integer(name='label')
        }

        data = [
            [  '小明去學校', '結果已經下課了', 0]
        ]

        def pair_iter():
            for row in data:
                yield {
                    'sentence1': row[0],
                    'sentence2': row[1],
                    'label': row[2]
                }
        if os.path.exists('question_pair.h5'):
            os.remove('question_pair.h5')

        dataset = H5Dataset(schema, './question_pair.h5', pair_iter())
        assert len(dataset) == len(data)

        for idx in range(len(data)):
            row = dataset[idx]
            sent1, sent2, label = data[idx]

            assert sent1 == row['sentence1']
            assert sent2 == row['sentence2']
            assert label == row['label']

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
            'meme image', 
            'greyscale image'
        ]

        image_paths = [
            'test/images/1.jpeg',
            'test/images/2.jpeg',
            'test/images/3.jpeg',
            'test/images/4.jpeg',
            'test/images/5.jpeg',
        ]
        def pair_iter():
            for (caption, img_path) in zip(captions, image_paths):
                yield {
                    'image': schema['image'].read_image(img_path),
                    'caption': caption
                }
        if os.path.exists('img_caption.h5'):
            os.remove('img_caption.h5')

        dataset = H5Dataset(schema, './img_caption.h5', pair_iter(),
            data_length=len(image_paths), chunk_size=4)
        assert len(dataset) == len(captions)

        for idx in range(len(image_paths)):
            data = dataset[idx]
            assert data['caption'] == captions[idx]

        os.remove('img_caption.h5')


        dataset = H5Dataset(schema, './img_caption.h5', pair_iter(),
            compression='lzf')

        for idx in range(len(image_paths)):
            data = dataset[idx]
            assert data['caption'] == captions[idx]

        os.remove('img_caption.h5')
