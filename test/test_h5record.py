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
            ['Lorem ipsum dolor sit amet, consectetur adipiscing elit','sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.', 1],
            ['Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris','nisi ut aliquip ex ea commodo consequat.', 0],
            ['Duis aute irure dolor in reprehenderit in voluptate velit','esse cillum dolore eu fugiat nulla pariatur.', 0],
            ['Excepteur sint occaecat cupidatat non proident, ','sunt in culpa qui officia deserunt mollit anim id est laborum.', 0],

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

        for idx in range(len(data)):
            row = dataset[idx]
            sent1, sent2, label = data[idx]
            assert sent1 == row['sentence1']
            assert sent2 == row['sentence2']
            assert label == row['label']

        os.remove('question_pair.h5')

