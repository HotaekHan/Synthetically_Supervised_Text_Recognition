#!/usr/bin/python
# encoding: utf-8

# pytorch
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

# python
import cv2
import numpy as np

# user-defined
from utils import LabelEncoder
from utils import SynthImageMaker

class listDataset(Dataset):
    def __init__(self, root_path, transform, encoder, img_size):
        self.root_path = root_path
        fp = open(self.root_path, 'r')
        self.lines = fp.readlines()
        fp.close()
        self.nSamples = len(self.lines)

        self.img_size = img_size
        self.transform = transform
        if encoder is None:
            raise ValueError("Encoder have to be defined")
        self.encoder = encoder

        # This is Renderer R in the paper
        self.synthData_maker = SynthImageMaker()

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'

        line_splits = self.lines[index].strip().split('\t')
        if len(line_splits) > 2:
            for iter_idx in range(2, len(line_splits), 1):
                line_splits[1] = line_splits[1] + ' ' + line_splits[iter_idx]
        imgpath = line_splits[0]
        img = cv2.imread(imgpath)

        if img is None:
            print('Corrupted image for %d' % index)
            return self[index + 1]

        # if input image is gray scale, then repeat along channel
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=2)
        if img.shape[2] == 1:
            img = np.repeat(img, 3, axis=2)
        elif img.shape[2] != 3:
            print('Check image channel. Only 1 or 3 channels are acceptable.')
            return self[index + 1]

        img = cv2.resize(img, self.img_size, interpolation=cv2.INTER_LINEAR)

        label = line_splits[1]

        # make synth image
        synthText = self.synthData_maker.do_make_synthData(label.upper())
        if synthText is None:
            print('SynthText is None')
            print(label)
            return self[index + 1]
        synthText = cv2.resize(synthText, self.img_size, interpolation=cv2.INTER_LINEAR)

        cv2.imshow('ori', img)
        cv2.imshow('test', synthText)
        cv2.waitKey(0)

        # normalize [-1, 1]
        img = img * (2. / 255.) - 1
        img = img.astype(np.float32)
        synthText = synthText * (2. / 255.) - 1
        synthText = synthText.astype(np.float32)
        synthText = np.expand_dims(synthText, 2)
        synthText = cv2.cvtColor(synthText, cv2.COLOR_GRAY2BGR)

        if self.transform is not None:
            img = self.transform(img)
            synthText = self.transform(synthText)

        return (img, label, synthText, imgpath)

    def collate_fn(self, batch):
        images, labels, synthTexts, imgpaths = zip(*batch)

        labels = list(labels)
        images = torch.cat([img.unsqueeze(0) for img in images], 0)
        synthTexts = torch.cat([synthText.unsqueeze(0) for synthText in synthTexts], 0)
        labels, lengths = self.encoder.encode(labels)

        return images, labels, synthTexts, lengths, imgpaths

def test_listdata():
    labels = "0123456789abcdefghijklmnopqrstuvwxyz"
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    encoder = LabelEncoder(labels=labels, is_ignore_case=True)

    list_dataset = listDataset(root_path='data/VGG90_val.txt', transform=transform,
                               encoder=encoder, img_size=(100, 32))

    data_loader = torch.utils.data.DataLoader(
        list_dataset, batch_size=10,
        shuffle=False, num_workers=1,
        collate_fn=list_dataset.collate_fn)

    with torch.set_grad_enabled(False):
        for batch_idx, (inputs, sequences, lengths, imgpaths) in enumerate(data_loader):
            print(inputs.shape)
            print(sequences)
            print(lengths)
            print(imgpaths)
            print(inputs.requires_grad)
            print(sequences.requires_grad)

    with torch.set_grad_enabled(True):
        for batch_idx, (inputs, sequences, lengths, imgpaths) in enumerate(data_loader):
            print(inputs.shape)
            print(sequences)
            print(lengths)
            print(imgpaths)
            print(inputs.requires_grad)
            print(sequences.requires_grad)

if __name__ == '__main__':
    test_listdata()