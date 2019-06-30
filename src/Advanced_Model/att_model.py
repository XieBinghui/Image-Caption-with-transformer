'''
Source code for an attention based image caption generation system described
in:
Show, Attend and Tell: Neural Image Caption Generation with Visual Attention
International Conference for Machine Learning (2015)
http://arxiv.org/abs/1502.03044
'''

import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
import numpy as np

MAX_LENGTH = 20

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def cuda_variable(tensor):
    if torch.cuda.is_available():
        return Variable(tensor.cuda())
    else:
        return Variable(tensor)

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(EncoderCNN, self).__init__()
        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-1]      # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        
    def forward(self, images):
        """Extract feature vectors from input images."""
        with torch.no_grad():
            features = self.resnet(images)
        features = features.reshape(features.size(0), -1)
        features = self.bn(self.linear(features))
        return features

class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_out, decoder_hidden):
        att1 = self.encoder_att(encoder_out)
        att2 = self.decoder_att(decoder_hidden)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)
        alpha = self.softmax(att)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)
        return attention_weighted_encoding, alpha

class DecoderRNN(nn.Module):

    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size, encoder_dim=256, dropout=0.5):
        super(DecoderRNN, self).__init__()
        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout

        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(p=self.dropout)
        self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)
        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)   # linear layer to create a sigmoid-activated gate
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(decoder_dim, vocab_size)
        self.init_weights()

    def init_weights(self):
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def load_pretrained_embeddings(self, embeddings):
        self.embedding.weight = nn.Parameter(embeddings)

    def fine_tune_embeddings(self, fine_tune=True):
        for p in self.embedding.parameters():
            p.requires_grad = fine_tune

    def init_hidden_state(self, encoder_out):
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)
        c = self.init_c(mean_encoder_out)
        return h, c

    def forward(self, encoder_out, encoded_captions, caption_lengths):
        """
        :return: scores for vocabulary, sorted encoded captions, decode lengths, weights
        """
        # print("caption_length={}".format(caption_lengths))
        # print(caption_lengths)
        batch_size = encoder_out.size(0)
        # print("batch_size={}".format(batch_size))
        encoder_dim = encoder_out.size(-1)
        # print("encoder_dim={}".format(encoder_dim))
        vocab_size = self.vocab_size
        
        encoded_captions=encoded_captions.to(device)

        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)
        # print("shape of encoder_out = {}".format(np.shape(encoder_out)))
        num_pixels = encoder_out.size(1)

        embeddings = self.embedding(encoded_captions)

        h, c = self.init_hidden_state(encoder_out)

        decode_lengths = [c-1 for c in caption_lengths]
        # len = int(decode_lengths[0][0])

        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(device)   # for train
        # predictions = torch.zeros(batch_size, len, vocab_size).to(device)       #for test    
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(device)
        # alphas = torch.zeros(batch_size, len, num_pixels).to(device)

        for t in range(max(decode_lengths)):
        # for t in range(len):
            batch_size_t = sum([l > t for l in decode_lengths ])
            # batch_size_t = 1
            attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t], h[:batch_size_t])
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))
            attention_weighted_encoding = gate * attention_weighted_encoding
            h, c = self.decode_step(
                torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1),
                (h[:batch_size_t], c[:batch_size_t]))
            preds = self.fc(self.dropout(h))
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha
        # print("shape of prediction = {}".format(np.shape(predictions)))
        return predictions, encoded_captions, decode_lengths, alphas
    
#     def sample(self, features, states=None):
#         """Generate captions for given image features using greedy search."""
#         sampled_ids = []
#         inputs = features.unsqueeze(1)
#         for i in range(self.max_seg_length):
#             hiddens, states = self.lstm(inputs, states)          # hiddens: (batch_size, 1, hidden_size)
#             outputs = self.linear(hiddens.squeeze(1))            # outputs:  (batch_size, vocab_size)
#             _, predicted = outputs.max(1)                        # predicted: (batch_size)
#             sampled_ids.append(predicted)
#             inputs = self.embed(predicted)                       # inputs: (batch_size, embed_size)
#             inputs = inputs.unsqueeze(1)                         # inputs: (batch_size, 1, embed_size)
#         sampled_ids = torch.stack(sampled_ids, 1)                # sampled_ids: (batch_size, max_seq_length)
#         return sampled_ids