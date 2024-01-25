import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from matplotlib import pyplot as plt


def imshow(inp, title=None):
    inp = inp.cpu().detach().numpy().transpose((1, 2, 0))
    #inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)

    
class CustomAudioDataset(Dataset):

    def __init__(self, images, labels):
        self.images = torch.tensor(images)
        self.labels = torch.tensor(labels)
        
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label =self.labels[idx]
        return image, label


def make_our_dataset(loader_bs1, maximages):
    images = []
    labels = []
    k = 0
    freq = np.ceil(len(loader_bs1.dataset)/maximages)
    for i, batch in enumerate(tqdm(loader_bs1)):
        if k%freq==0:
            img, lab = batch
            images.append(img[0].numpy())
            labels.append(lab.numpy())
            
            k+=1
        else:
            k+=1
    images = np.array(images)
    labels = np.array(labels)
    dataset = CustomAudioDataset(images, labels)
    return dataset

def make_our_dataset_v2(data, idxes):
    images = []
    labels = []
    k = 0
    for i in tqdm(range(len(idxes))):
        image, label = data[idxes[i]]
        images.append(image.numpy())
        
        labels.append(label)

    images = np.array(images)
    labels = np.array(labels)
    dataset = CustomAudioDataset(images, labels)
    return dataset