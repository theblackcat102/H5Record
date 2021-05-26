# H5Record

Large dataset ( > 100G, <= 1T) storage format for Pytorch (wip) 

## Why?

* Writing large dataset is still a wild west in pytorch. Approaches seen in the wild include:

    - large directory with lots of small files : slow IO when complex file is fetched, deserialized frequently 
    - database approach : depend on what kind of database engine used, usually multi-process read is not supported
    - the above method scale non linear in terms of data - storage size

* TFRecord solved the above problems well ( multiprocess fetch, (de)compression ), fast serialization ( protobuf )

* However TFRecord port does not support data size evaluation (used frequently by Dataloader ), no index level access available ( important for data evaluation or verification )

H5Record aim to tackle TFRecord problems by compressing the dataset into [HDF5](https://support.hdfgroup.org/HDF5/doc/TechNotes/BigDataSmMach.html) file with an easy to use interface through predefined interfaces ( String, Image, Sequences, Label).


## Note

Due to in progress development, this package should be use in care in storage with FAT, FAT-32 format 

### TODO

- [ ] Test combinations of different data modalities

- [ ] Do more tuning and experiments on different driver settings

- [ ] Performance benchmark


