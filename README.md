# H5Record

Large dataset ( > 100G, <= 1T) storage format for Pytorch (wip) 


## Why?

* Writing large dataset is still a wild west in pytorch. Approaches seen in the wild include:

    - large directory with lots of small files : slow IO when complex file is fetched, deserialized frequently 
    - database approach : depend on what kind of database engine used, usually multi-process read is not supported
    - the above method scale non linear in terms of data - storage size

* TFRecord solved the above problems well ( multiprocess fetch, (de)compression ), fast serialization ( protobuf )

* However TFRecord port does not support data size evaluation (used frequently by Dataloader ), no index level access available ( important for data evaluation or verification )

H5Record aim to tackle TFRecord problems by compressing the dataset into [HDF5](https://support.hdfgroup.org/HDF5/doc/TechNotes/BigDataSmMach.html) file with an easy to use interface through predefined interfaces ( String, Image, Sequences, Integer).

Some advantage of using H5Record

* Support multi-process read

* Relatively simple to use and low technical debt

* Support compression/de-compression on the fly

* Quick load to memory if required

### Simple usage

```
pip install h5record
```


1. Sentence Similarity

```python
from h5record import H5Dataset, Float, String

schema = (
    String(name='sentence1'),
    String(name='sentence2'),
    Float(name='label')
)
data = [
    ['Sent 1.', 'Sent 2', 0.1],
    ['Sent 3', 'Sent 4', 0.2],
]

def pair_iter():
    for row in data:
        yield {
            'sentence1': row[0],
            'sentence2': row[1],
            'label': row[2]
        }

dataset = H5Dataset(schema, './question_pair.h5', pair_iter())
for idx in range(len(dataset)):
    print(dataset[idx])

```


## Note

Due to in progress development, this package should be use in care in storage with FAT, FAT-32 format 

## Comparison between different compression algorithm

No chunking is used

| Compression Type  | File size  | Read speed row/second  |
|---|---|---|
| no compression  | 2.0G  | 2084.55 it/s  |
| lzf  | 1.7G  | 1496.14 it/s  |
| gzip | 1.1G  | 843.78 it/s  |

benchmarked in i7-9700, 1TB NVMe SSD



If you are interested to learn more feel free to checkout the [note](NOTES.md) as well!


