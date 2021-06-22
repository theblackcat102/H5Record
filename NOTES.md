# Note

### Materials

Useful materials about HDF5 

[HDF5 tech note](https://support.hdfgroup.org/ftp/HDF5/documentation/doc1.8/TechNotes/TechNote-HDF5-ImprovingIOPerformanceCompressedDatasets.pdf)

[Compression benchmark ](https://www.hdfgroup.org/2018/06/hdf5-or-how-i-learned-to-love-data-compression-and-partial-i-o/)


Other materials and discussion

1. [A discussion industrial large dataset solution in Pytorch](https://github.com/pytorch/pytorch/issues/20822)


2. [H5Record reddit post](https://www.reddit.com/r/MachineLearning/comments/nsq3ai/p_h5records_store_large_datasets_in_one_single/)



## Comparison between LMDB and HDF5

Data obtain from [w86763777 script](https://github.com/w86763777/LMDBvsHDF5)

| Compression Type  | Write  | Read  | Size |
|---|---|---|---|
| HDF5  | 4.32 secs |  1.20 secs | 496K | 
| LMDB | 1.68 secs  | 0.10 secs  | 224M |

* Benchmarked on 103 images, total size of 5.4M, image resized on LMDB file

Overall LMDB provide a 2.6x improvement on write and 12x on read speed (results are averaged on 10 reads/writes session, benchmark on macbook 2017 Intel Core i5  ). 

Maybe H5record should include additional backend choice for LMDB since it supports significant fast load of binary file.


### TODO

- [ ] Test combinations of different data modalities

- [ ] Do more tuning and experiments on different driver settings

- [ ] Performance benchmark:

    - [ ] Performance comparison between zip in multiple workers ( I suspect there's some improvement to be done here )

    - [ ] In memory (dataset[:]) access vs no compression

