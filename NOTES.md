# Note

### Matrials

Useful materials about HDF5 

[HDF5 tech note](https://support.hdfgroup.org/ftp/HDF5/documentation/doc1.8/TechNotes/TechNote-HDF5-ImprovingIOPerformanceCompressedDatasets.pdf)

[Compression benchmark ](https://www.hdfgroup.org/2018/06/hdf5-or-how-i-learned-to-love-data-compression-and-partial-i-o/)


Other materials and discussion

1. [A discussion industrial large dataset solution in Pytorch](https://github.com/pytorch/pytorch/issues/20822)


2. [H5Record reddit post](https://www.reddit.com/r/MachineLearning/comments/nsq3ai/p_h5records_store_large_datasets_in_one_single/)



### TODO

- [ ] Test combinations of different data modalities

- [ ] Do more tuning and experiments on different driver settings

- [ ] Performance benchmark:

    - [ ] Performance comparison between zip in multiple workers ( I suspect there's some improvement to be done here )

    - [ ] In memory (dataset[:]) access vs no compression

