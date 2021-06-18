import torch
from torchvision.datasets import MNIST
import numpy as np
from tqdm import tqdm # for progress tracking


from h5record import H5Dataset, Integer, Image

def mnist_generator():
    dataset = MNIST(root='mnist', download=True)
    for data in tqdm(dataset):
        image, label = data
        np_img = np.array(image).reshape(1, 28, 28)
        # NOTE: the key must be same as schema name
        yield {'img': np_img, 'label': label }


schema = (
    Integer('label'),
    Image(name='img', h=28, w=28)
)

# it's recommended to provide dataset size
# this would provide faster index access
dataset = H5Dataset(schema, 'mnist.h5', 
    mnist_generator(), 
    data_length=60000, chunk_size=300, 
    multiprocess=True)


print('Data size ', len(dataset))
from torch.utils.data import DataLoader

dataloader = DataLoader(dataset, batch_size=128, 
    shuffle=True, num_workers=4)

for batch in tqdm(dataloader):
    imgs = batch['img']
    labels = batch['label']
print(imgs.shape, labels.shape)