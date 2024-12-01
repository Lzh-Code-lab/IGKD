import os
import cv2
import numpy as np
import torch
import random
import torch.nn as nn
import torch.utils.data
from torchvision import transforms, datasets


def labels_txt_load(labels_path):
    lines = open(labels_path, 'r').readlines()
    dataset = []
    for line in lines:
        key, label = line.splitlines()[0], line.split('/')[0]
        label = int(label)
        dataset.append([key, label])
    return dataset


def add_gaussian_noise(image_array, mean=0.0, var=30):
    std = var**0.5
    noisy_img = image_array + np.random.normal(mean, std, image_array.shape)
    noisy_img_clipped = np.clip(noisy_img, 0, 255).astype(np.uint8)
    return noisy_img_clipped


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, img_file_path, transform=None, mode='train'):
        self.dataset = dataset
        self.img_file_path = img_file_path
        self.img_label_list = [x for _, x in self.dataset]
        self.transform = transform
        self.mode = mode
        self.img_one_path_list = []
        for img_one in dataset:
            img_one_path = os.path.join(self.img_file_path, img_one[0])
            self.img_one_path_list.append(img_one_path)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        key, label = self.dataset[index]
        images = cv2.imread(self.img_one_path_list[index])
        img = images[:, :, ::-1]

        if self.mode == 'train':
            if random.uniform(0, 1) > 0.50:
                img = add_gaussian_noise(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, label


class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
    """

    def __init__(self, dataset, indices=None, num_samples=None):

        # if indices is not provided,
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) \
            if indices is None else indices

        # if num_samples is not provided,
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples

        # distribution of classes in the dataset
        label_to_count = {}
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1

        # weight for each sample
        weights = [1.0 / label_to_count[self._get_label(dataset, idx)]
                   for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, dataset, idx):
        dataset_type = type(dataset)
        return dataset.img_label_list[idx]

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(
            self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples


class DataLoad(nn.Module):
    def __init__(self, datasets_path, train_batch_size=128, val_batch_size=256, workers=8):
        super(DataLoad, self).__init__()
        self.datasets_path = datasets_path
        assert os.path.exists(self.datasets_path), "{} path does not exist.".format(self.datasets_path)
        self.img_size = 256
        self.img_crop_size = 224
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.workers = workers

    def train_load(self):
        data_transform = {
            "train": transforms.Compose([transforms.ToPILImage(),
                                         transforms.Resize(self.img_size),
                                         transforms.RandomCrop(self.img_crop_size),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                         ])}
        train_labels_path = os.path.join(self.datasets_path, 'Annotated/train_labels.txt')
        train_dataset = labels_txt_load(train_labels_path)
        train_img_path = os.path.join(self.datasets_path, 'train')
        train_set = ImageDataset(train_dataset, img_file_path=train_img_path, transform=data_transform['train'])
        # RAF数据集加载方式
        train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, batch_size=self.train_batch_size, num_workers=self.workers)
        # AffectNet数据集采用不平衡数据集加载方式
        # train_loader = torch.utils.data.DataLoader(train_set, sampler=ImbalancedDatasetSampler(train_set),
        #                                            batch_size=self.train_batch_size, num_workers=self.workers)

        return train_loader, len(train_dataset)

    def val_load(self):
        data_transform = {
            "val": transforms.Compose([transforms.ToPILImage(),
                                       transforms.Resize(self.img_size),
                                       transforms.CenterCrop(self.img_crop_size),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                       ])}
        val_labels_path = os.path.join(self.datasets_path, 'Annotated/val_labels.txt')
        val_dataset = labels_txt_load(val_labels_path)
        val_img_path = os.path.join(self.datasets_path, 'val')
        val_set = ImageDataset(val_dataset, img_file_path=val_img_path, transform=data_transform['val'], mode='val')
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=self.val_batch_size, shuffle=False, num_workers=self.workers)

        return val_loader, len(val_dataset)

    def forward(self, mode='train'):
        assert mode == 'train' or 'val', 'mode is false!'
        if mode == 'train':
            train_loader, train_num = self.train_load()
            val_loader, val_num = self.val_load()
            print("using {} images for training, {} images for validation.".format(train_num, val_num))
            return train_loader, val_loader, train_num, val_num
        else:
            val_loader, val_num = self.val_load()
            print("val number is {}.".format(val_num))
            return val_loader, val_num


class DataErase(nn.Module):
    def __init__(self, p=1.0, scale=(0.10, 0.25), ration=(0.4, 2.5)):
        super(DataErase, self).__init__()
        self.random_erasing = transforms.RandomErasing(p, scale, ration, value=random.random())

    def forward(self, x):
        img_origin = x
        img_erasing = self.random_erasing(x)
        img_result = torch.stack((img_origin, img_erasing), dim=1)
        return img_result






