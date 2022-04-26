import torch
import torch.nn as nn
import pandas as pd
import pickle
from torch.utils.data import Dataset, DataLoader


# 定义”我的数据集“类
class MyDataset(Dataset):
    def __init__(self, a, b, c, d):
        self.en_corpus_list = a
        self.ch_corpus_list = b
        self.en_word_2_index = c
        self.ch_word_2_index = d

    def __getitem__(self, item):
        en = self.en_corpus_list[item]  # 获取每一条语句
        ch = self.ch_corpus_list[item]  # 获取每一条语句
        # 进行索引式转化
        en_index = [self.en_word_2_index[i] for i in en]
        ch_index = [self.ch_word_2_index[i] for i in ch]
        return en_index, ch_index

    def batch_data_process(self, batch_data):
        global device
        en_list, ch_list = [], []
        en_len, ch_len = [], []

        for x, y in batch_data:
            en_list.append(x)
            ch_list.append(y)
            en_len.append(len(x))
            ch_len.append(len(y))

        max_en_len = max(en_len)
        max_ch_len = max(ch_len)

        new_en_list = [s + [self.en_word_2_index['<PAD>']] * (max_en_len - len(s)) for s in en_list]
        new_ch_list = [[self.ch_word_2_index['<BOS>']] + s + [self.ch_word_2_index['<EOS>']] + [self.ch_word_2_index['<PAD>']] * (max_ch_len - len(s)) for s in ch_list]

        new_en_list = torch.tensor(new_en_list, device=device)
        new_ch_list = torch.tensor(new_ch_list, device=device)

        return new_en_list, new_ch_list

    def __len__(self):
        assert len(self.en_corpus_list) == len(self.ch_corpus_list)
        return len(self.en_corpus_list)


class Encoder(nn.Module):
    def __init__(self):
        global len_en
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(50, 100, batch_first=True)
        self.embedding = nn.Embedding(len_en, 50)

    def forward(self, batch_en_index):
        embedding_en = self.embedding(batch_en_index)  # batch_size x word_nums x embedding_dimension
        _, encoder_hidden = self.lstm(embedding_en)  # 这里的encode_hidden是一个元组，包括hn（隐藏状态向量）和cn（lstm机制里面的一个变量）

        return encoder_hidden


class Decoder(nn.Module):
    def __init__(self):
        global len_ch
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(len_ch, 50)
        self.lstm = nn.LSTM(50, 100, batch_first=True)
        pass

    def forward(self, batch_ch_index, encode_hidden):
        embedding_ch = self.embedding(batch_ch_index)  # batch_size(sentence_nums) x word_nums x embedding_dimension
        decoder_output, decoder_hidden = self.lstm(embedding_ch, encode_hidden)

        return decoder_output, decoder_hidden
    pass


class Seq2Seq(nn.Module):
    def __init__(self):
        global len_ch
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

        self.classifier = nn.Linear(100, len_ch)
        self.loss_function = nn.CrossEntropyLoss()

    def forward(self, all_data):
        batch_en_index = all_data[0]
        batch_ch_index = all_data[1]
        encoder_hidden = self.encoder(batch_en_index)
        decoder_inputs = batch_ch_index[:, :-1]
        labels = batch_ch_index[:, 1:]  # batch_size x word_nums
        decoder_outputs, decoder_hidden = self.decoder(decoder_inputs, encoder_hidden)
        results = self.classifier(decoder_outputs)  # batch_size x word_nums x dimension
        loss = self.loss_function(results.reshape(-1, results.shape[-1]), labels.reshape(-1))

        return loss
    pass


def translate(sentence):
    global en_word_2_index, model, device
    sentence_index = torch.tensor([[en_word_2_index[word] for word in sentence]], device=device)  # 1 x word_nums
    encoder_hidden = model.encoder(sentence_index)  # 1 x hidden_dimension
    decoder_inputs = torch.tensor([[ch_word_2_index['<BOS>']]], device=device)  # 1 x 1

    results = []

    while True:
        decoder_outputs, decoder_hidden = model.decoder(decoder_inputs, encoder_hidden)  # 1 x hidden_dimension
        predict = model.classifier(decoder_outputs)  # 1 x len_ch
        predict_index = int(torch.argmax(predict, dim=-1))
        word = ch_index_2_word[predict_index]

        if word == '<EOS>' or len(results) > 50:
            break

        results.append(word)

        decoder_inputs = torch.tensor([[predict_index]], device=device)

    print('译文为', ''.join(results))


if __name__ == '__main__':
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # 获取基于语料库产生的一些数据
    with open("datas/ch.vec", "rb") as f1:
        _, ch_word_2_index, ch_index_2_word = pickle.load(f1)
    with open("datas/en.vec", "rb") as f2:
        _, en_word_2_index, en_index_2_word = pickle.load(f2)

    len_ch = len(ch_word_2_index)
    len_en = len(en_word_2_index)

    # 在word2index字典中添加<PAD>、<BOS>、<EOS>字符
    ch_word_2_index.update({'<PAD>': len_ch, '<BOS>': len_ch+1, '<EOS>': len_ch+2})
    en_word_2_index.update({'<PAD>': len_en})
    # 在index2word列表中添加<PAD>、<BOS>、<EOS>字符
    ch_index_2_word += ['<PAD>', '<BOS>', '<EOS>']
    en_index_2_word += ['<PAD>']
    '''由此收获了完整的4个变量ch_word_2_index、en_word_2_index、ch_index_2_word、en_index_2_word'''

    # 更新词库长度
    len_ch = len(ch_word_2_index)
    len_en = len(en_word_2_index)

    # 获取原始语料
    df = pd.read_csv('datas/translate.csv')
    en_corpus_list = list(df['english'][:200])  # 要转换成list
    ch_corpus_list = list(df['chinese'][:200])
    '''至此获得了原始语料变量en_corpus_list、ch_corpus_list'''

    # 构建数据集
    dataset = MyDataset(en_corpus_list, ch_corpus_list, en_word_2_index, ch_word_2_index)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=dataset.batch_data_process)

    model = Seq2Seq()
    model = model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=0.001)

    for times in range(50):
        for i in dataloader:
            loss = model(i)
            loss.backward()
            opt.step()
            opt.zero_grad()

        print(loss)

    while True:
        s = input('请输入英文：')
        translate(s)