import sys,os
import numpy as np
import tensorflow as tf

def load_data(filename):
    vocab2idx = {}
    idx2vocab = {}
    label2idx = {}
    idx2label = {}
    idx = 0
    vocab2idx["UNK"] = idx
    idx += 1
    label_idx = 0
    label2idx["O"] = label_idx
    label_idx += 1
    with open(filename) as fin:
        for line in fin:
            lin = line.strip().split("\t")
            query = lin[0].decode("utf8").lower()
            label = lin[1]
            if label not in label2idx:
                label2idx[label] = label_idx
                idx2label[label_idx] = label
                label_idx += 1
            for word in query:
                if word not in vocab2idx and word not in vocab2idx:
                    vocab2idx[word] = idx
                    idx2vocab[idx] = word
                    idx = idx + 1
    sys.stderr.write('load vocab from file:[%s], length(%s)' % (filename, len(vocab2idx)))
    return vocab2idx, idx2vocab, label2idx, idx2label

class Config():
    def __init__(self):
        embedding_dim = 64
        seq_length = 600
        self.train_file_name = "data/train.data"
        self.test_file_name = "data/test.data"
        self.dev_file_name = "data/dev.data"
        self.vocab2idx, self.idx2vocab, self.label2idx, self.idx2label\
                = load_data(self.train_file_name)
        self.vocab_size = len(self.vocab2idx)
        self.num_classes = len(self.idx2label)
        hidden_dim = 128
        rnn = 'gru'
        self.dropout_keep_prob = 0.8
        self.learning_rate = 0.001
        self.batch_size = 10
        self.num_epochs = 100
        self.attention_size = 50
        self.embedding_dim = 100
        self.label_list_file = "data/label.list"
        self.word_list_file = "data/word.list"
        self.print_dict()
        self.train = load_data(self.train_file_name)
        self.dev = load_data(self.dev_file_name)
        self.test = load_data(self.test_file_name)
    def print_dict(self):
        fin = open(self.label_list_file, "w")
        output_list = sorted(self.label2idx.items(), key = lambda d:d[1], reverse = False)
        for item in output_list:
            fin.write(item[0] + "\t" + str(item[1]) + "\n")
        fin.close()
        fin = open(self.word_list_file, "w")
        output_list = sorted(self.vocab2idx.items(), key = lambda d:d[1], reverse = False)
        for item in output_list:
            fin.write(item[0].encode("utf8") + "\t" + str(item[1]) + "\n")
        fin.close()
    def load_data(self, filename):
        fin = open(filename, "r")
        words = []
        result = []
        for line in fin:
            lin = line.strip().split("\t")
            if len(lin) != 2:
                continue
            sen = lin[0].decode("utf8")
            intent = self.label2idx[lin[1]]
            for idx, item in enumerate(sen):
                item = item.lower()
                if item in vocab2idx:
                    words.append(vocab2idx[item])
                else:
                    words.append(vocab2idx["UNK"])
            result.append([words, intent])
        fin.close()
        return result
    def minibatches(data, minibatch_size):
        x_batch, y_batch = [], []
        for (x,y) in data:
            if len(x_batch) == minibatch_size:
                yield x_batch, y_batch
                x_batch, y_batch = [], []
            x_batch += [x]
            y_batch += [y]
        if len(x_batch) != 0:
            yield x_batch, y_batch



if __name__ == "__main__":
    config = Config()



