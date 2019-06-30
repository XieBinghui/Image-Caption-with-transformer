import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import pickle
import numpy as np
import nltk
from PIL import Image
from build_vocab import Vocabulary
from pycocotools.coco import COCO
import json

class CocoDataset(data.Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""
#     def __init__(self, root, json, vocab, transform=None):
#         """Set the path for images, captions and vocabulary wrapper.
        
#         Args:
#             root: image directory.
#             json: coco annotation file path.
#             vocab: vocabulary wrapper.
#             transform: image transformer.
#         """
#         self.root = root
#         self.coco = COCO(json)
#         self.ids = list(self.coco.anns.keys())
#         self.vocab = vocab
#         self.transform = transform

#     def __getitem__(self, index):
#         """Returns one data pair (image and caption)."""
#         coco = self.coco
#         vocab = self.vocab
#         ann_id = self.ids[index]
#         caption = coco.anns[ann_id]['caption']
#         img_id = coco.anns[ann_id]['image_id']
#         path = coco.loadImgs(img_id)[0]['file_name']

#         image = Image.open(os.path.join(self.root, path)).convert('RGB')
#         if self.transform is not None:
#             image = self.transform(image)

#         # Convert caption (string) to word ids.
#         tokens = nltk.tokenize.word_tokenize(str(caption).lower())
#         caption = []
#         caption.append(vocab('<start>'))
#         caption.extend([vocab(token) for token in tokens])
#         caption.append(vocab('<end>'))
#         target = torch.Tensor(caption)
#         return image, target
    def __init__(self, root, json_file, vocab, transform=None):
        """Set the path for images, captions and vocabulary wrapper.
        
        Args:
            root: image directory.
            json: coco annotation file path.
            vocab: vocabulary wrapper.
            transform: image transformer.
        """
        self.root = root
        self.flickr = json.load(open(json_file, 'r'))
        # self.ids = list(self.coco.anns.keys())
        self.vocab = vocab
        self.transform = transform

    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""
        flickr = self.flickr
        vocab = self.vocab
        caption = flickr[index]['caption']
        caption=' '.join(caption)
        img_id = flickr[index]['img_id']
        path = flickr[index]['file']
        lengths = flickr[index]['length']

        image = Image.open(os.path.join(self.root, path)).convert('RGB')
        image = image.resize([224, 224], Image.LANCZOS)
        if self.transform is not None:
            image = self.transform(image)

        # Convert caption (string) to word ids.
#         print(caption)
#         print(len(caption))
#         print("leng12={}".format(lengths))
        tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        target = torch.Tensor(caption)
#         print("shape of target={}".format(np.shape(target)))
        return img_id, image, target, lengths+2

    def __len__(self):
        return len(self.flickr)


def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).
    
    We should build custom collate_fn rather than using default collate_fn, 
    because merging caption (including padding) is not supported in default.

    Args:
        data: list of tuple (image, caption). 
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length (descending order).
    # data.sort(key=lambda x: len(x[2]), reverse=True)
    img_id, images, captions, lengths = zip(*data)
    # print(captions)
    lengths = lengths[0]
    # print("length={}".format(lengths))
#     print("len caption{}".format(len(captions)))
    

    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    # lengths = [len(cap) for cap in captions]
    targets = torch.zeros(1, lengths).long()
#     print("target shape{}".format(np.shape(captions)))
    # for i in range(lengths):
    # print(type(captions))
    # print(np.shape(captions))
    targets[0] = captions[0]
    lens = torch.zeros(1, 1)
    lens[0][0] = lengths
    return img_id, images, targets, lens

def get_loader(root, json, vocab, transform, batch_size, shuffle, num_workers):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    # COCO caption dataset
    coco = CocoDataset(root=root,
                       json_file=json,
                       vocab=vocab,
                       transform=transform)
    
    # Data loader for COCO dataset
    # This will return (images, captions, lengths) for each iteration.
    # images: a tensor of shape (batch_size, 3, 224, 224).
    # captions: a tensor of shape (batch_size, padded_length).
    # lengths: a list indicating valid length for each caption. length is (batch_size).
    data_loader = torch.utils.data.DataLoader(dataset=coco, 
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)
    return data_loader