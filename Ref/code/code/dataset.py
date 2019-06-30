import json
import numpy as np
import torch
import random
import os

from torch.utils.data import Dataset, DataLoader

from transformer.functional import subsequent_mask

DEBUG = False
CAPTION_NUM = 5
IX_TO_WORD = "ix_to_word"
FILE = "feat"
IMG_INFO = "images"
IMG_INFO_PATH = u'file_path'
IMG_INFO_IDX = u'id'
IMG_INFO_SPLIT = u'split'
TRAIN = u'train'
VALID = u'val'
TEST = u'test'
INPUT_SUFFIX = u'npz'
CAPTION = u'caption'
BOS = u'<s>'
EOS = u'<\s>.'
PAD = u'<pad>'

class FlickrDataset(Dataset):
    def __init__(self, args, data_type):
        self.data_path = args.dataset
        self.feat_path = os.path.join(args.dataset, args.feature)
        self.cap_json = json.load(open(os.path.join(args.dataset, args.cap_json)))
        self.dic_json = json.load(open(os.path.join(args.dataset, args.dic_json)))
        self.batch_size = args.batch_size
        self.max_len = args.max_len
        self.data_type = data_type

        self.idx2word, self.word2idx, self.padding = self._word_preprocessing(self.dic_json)
        self.data_idx = self._data_split(self.dic_json)
        self.vocab_size = len(self.word2idx)

    def _word_preprocessing(self, dic_json):
        idx2word = dic_json[IX_TO_WORD]
        idx2word = {int(k): idx2word[k] for k in idx2word}
        # assume the idx is ranging from 1 to len(idx2word)
        idx2word[len(idx2word)+1] = BOS
        idx2word[len(idx2word)+1] = EOS
        idx2word[0] = PAD
        # convert the word into idx
        word2idx = {idx2word[k]: k for k in idx2word}
        print(" ==> Vocab Size: %d" % len(idx2word))
        return idx2word, word2idx, word2idx[PAD]

    def _data_split(self, dic_json):
        img_info = dic_json[IMG_INFO]
        training_idx = [i for i in range(len(img_info)) if img_info[i][IMG_INFO_SPLIT]==TRAIN]
        valid_idx = [i for i in range(len(img_info)) if img_info[i][IMG_INFO_SPLIT]==VALID]
        test_idx = [i for i in range(len(img_info)) if img_info[i][IMG_INFO_SPLIT]==TEST]
        print(" ==> Dataset Size: %d training, %d validation, %d testing" % 
                            (len(training_idx), len(valid_idx), len(test_idx)))
        if self.data_type == TRAIN:
            data_idx = training_idx
        elif self.data_type == VALID:
            data_idx = valid_idx
        elif self.data_type == TEST:
            data_idx = test_idx
        else:
            raise ValueError
        return self._sort_by_length(data_idx)

    def _sort_by_length(self, data_idx):
        length_dict = {}
        if self.data_type != VALID:
            for i in data_idx:
                caption_num = len(self.cap_json[i])
                for j in range(caption_num):
                    length_dict[(i,j)] = len(self.cap_json[i][j][CAPTION])
        else:
            for i in data_idx:
                length_dict[(i, 0)] = len(self.cap_json[i][0][CAPTION])
        # sorted by descending length
        sorted_idx = [i[0] for i in sorted(length_dict.items(), key=lambda x: -x[1])]
        if DEBUG:
            print(length_dict[sorted_idx[0]])
            print(max(length_dict.values()))
            print(length_dict[sorted_idx[-1]])
            print(min(length_dict.values()))
        return sorted_idx

    def make_target_mask_ntoken(self, caption_tensor):
        target = caption_tensor[:, :-1]
        target_y = caption_tensor[:, 1:]
        target_mask = (target!=self.padding).unsqueeze(-2)
        if DEBUG:
            print(target_mask)
        target_mask = target_mask & subsequent_mask(target_mask.size(-1))
        return target, target_mask, target_y, (target_y!=self.padding).sum().item()

    def __getitem__(self, idx):
        idx, caption_idx = self.data_idx[idx]
        img_info = self.dic_json[IMG_INFO][idx]
        if DEBUG:
            print(os.path.join(self.feat_path,
                               str(img_info[IMG_INFO_IDX]) + "." + INPUT_SUFFIX))
        feat = np.load(os.path.join(self.feat_path, str(img_info[IMG_INFO_IDX]) + "." + INPUT_SUFFIX))[FILE]
        caption_words = self.cap_json[idx][caption_idx][CAPTION]
        caption = [self.word2idx[BOS]] + [self.word2idx[w] for w in caption_words] + [self.word2idx[EOS]]
        caption = np.array(caption + [self.padding] * (self.max_len - len(caption)), dtype=np.int64)

        if self.data_type != TRAIN:
            return str(img_info[IMG_INFO_IDX]), feat, caption

        return feat, caption

    def __len__(self):
        return len(self.data_idx)

class Cfg(object):
    def __init__(self):
        self.dataset = "../DATASET_Flickr30k"
        self.feature = "resnet101_fea/fea_att"
        self.cap_json = "cap_flickr30k.json"
        self.dic_json = "dic_flickr30k.json"
        self.batch_size = 1
        self.max_len = 90
        self.padding = 0
        self.data_type = TRAIN

if __name__ == "__main__":
    args = Cfg()
    data = FlickrDataset(args)
    train_loader = DataLoader(dataset=data, batch_size=args.batch_size, shuffle=False)
    for batch in train_loader:
        print(batch)
        print(data.make_mask_target_ntoken(batch[1]))
        break