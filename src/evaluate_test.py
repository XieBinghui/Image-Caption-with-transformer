import torch
import matplotlib.pyplot as plt
import numpy as np 
import argparse
import pickle 
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from torchvision import transforms 
from build_vocab import Vocabulary
# from model import EncoderCNN, DecoderRNN
from Advanced_Model.Transformer_Model import EncoderCNN, Transformer
from PIL import Image
from loader import get_loader
import logging
import json
from torch.nn.utils.rnn import pack_padded_sequence


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def ToWord(sentence, vocab):
    index = torch.argmax(sentence, dim=1)
    print(index)
    sentences = []
    for i in range(len(sentence)):
        sentences.append(vocab.idx2word[index[i].item()])
    print(sentences)
        
    return sentences


class Sentences(object):
    def __init__(self, savedir):
        self.sentences = []
        if os.path.exists(savedir) == False:
            os.makedirs(savedir)
        self.filepath = os.path.join(savedir, "flickr_transformer.json")

    def add_sentence(self, image_id, sentence):
        caption = ' '.join(sentence[1:-1])
        s = {'image_id':image_id, 'caption':caption}
        if (image_id % 200) == 0:
            print(s)
        self.sentences.append(s)

    def save_sentences(self):
        with open(self.filepath, 'w') as f:
            json.dump(self.sentences, f)

def main(args):
    # Image preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])
    
    # Load vocabulary wrapper
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    data_loader = get_loader(args.image_dir, args.caption_path, vocab, 
                                transform, 1,
                                shuffle=True, num_workers=args.num_workers) 

    # Build models
    encoder = EncoderCNN(args.embed_size).eval()  # eval mode (batchnorm uses moving mean/variance)
    # decoder = DecoderRNN(args.embed_size, args.hidden_size, len(vocab), args.num_layers)
    decoder = Transformer(dict_size=len(vocab), image_feature_dim=2048, vocab=vocab, tf_ratio=0).to(device)
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    decoder.eval()
    
    
    # Load the trained model parameters
    encoder.load_state_dict(torch.load(args.encoder_path))
    decoder.load_state_dict(torch.load(args.decoder_path))

    # Prepare an image
    # image = load_image(args.image, transform)
    # image_tensor = image.to(device)

    total_step = len(data_loader)
    s = Sentences('./data')
    for i, (img_id, images, captions, features, lengths) in enumerate(data_loader):
        
        if i==0:
            continue
        
        # Set mini-batch dataset
        images = images.to(device)
        captions = captions.to(device)
        
        
        # Generate an caption from the image
        len_max = max(lengths)
        features = features.cuda()
        sentence = decoder(features, len_max, None, None)

        # Convert word_ids to words
#         sampled_caption = []
#         for word_id in sampled_ids:
#             word = vocab.idx2word[word_id]
#             sampled_caption.append(word)
#             if word == '<end>':
#                 break
#         sentence = ' '.join(sampled_caption)
        s.add_sentence(img_id[0], ToWord(sentence[0],vocab))
        if (i % 500) == 0:
                logging.info("  {:4d}".format(i))
    logging.info("Saving sentences...")
    s.save_sentences()
    logging.info("Done.")

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, default='./data/DATASET_Flickr30k/images', help='directory for resized images')
    # parser.add_argument('--image', type=str, required=True, help='input image for generating caption')
    parser.add_argument('--encoder_path', type=str, default='flickr_model_transformer/encoder-4-1000.ckpt', help='path for trained encoder')
    parser.add_argument('--decoder_path', type=str, default='flickr_model_transformer/decoder-4-1000.ckpt', help='path for trained decoder')
    parser.add_argument('--vocab_path', type=str, default='data/vocab.pkl', help='path for vocabulary wrapper')
    parser.add_argument('--caption_path', type=str, default='data/annotations/val_flickr.json', help='path for train annotation json file')
    
    # Model parameters (should be same as paramters in train.py)
    parser.add_argument('--embed_size', type=int , default=256, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int , default=512, help='dimension of lstm hidden states')
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--num_layers', type=int , default=1, help='number of layers in lstm')
    parser.add_argument('--beam_width', type=int, default=5,
            help='Beam width (used in evaluation)')
    args = parser.parse_args()
    main(args)
