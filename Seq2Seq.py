import math
import time

import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchtext import data
from torchtext.data import Iterator
import torch.nn.functional as F
SEED = 1234

random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True  # 确保实验的重复性


class Dataset(data.Dataset):
    name = "lic2019"

    def __init__(self, path, src_field, trg_field, **kwargs):
        fields = [("src", src_field), ("trg", trg_field)]
        examples = []
        print('loading dataset from {}'.format(path))
        with open(path, encoding="utf-8") as f:
            # special
            for line in f.readlines():
                src = line.split("\t")[0]
                trg = line.split("\t")[1].replace("\n", "")
                examples.append(data.Example.fromlist([src, trg], fields=fields))

        print('size of dataset in {} is : {}'.format(path, len(examples)))
        super(Dataset, self).__init__(examples, fields, **kwargs)


# 增加初始token和结尾token
BOS_WORD = "<sos>"
EOS_WORD = "<eos>"
BLANK_WORD = "<blank>"
MAX_LEN = 30

SRC = data.Field(init_token=BOS_WORD, eos_token=EOS_WORD, lower=True)
TRG = data.Field(init_token=BOS_WORD, eos_token=EOS_WORD, lower=True)

train_path = "train1.txt"
valid_path = "valid1.txt"
test_path = "test1.txt"
train_data = Dataset(train_path, src_field=SRC, trg_field=TRG)
valid_data = Dataset(valid_path, src_field=SRC, trg_field=TRG)
test_data = Dataset(test_path, src_field=SRC, trg_field=TRG)

SRC.build_vocab(train_data, min_freq=2)  # 建立词汇表的时候需要把valid，test的单词都加进去吗？ yes
TRG.build_vocab(train_data, min_freq=2)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 128

train_iterator, valid_iterator, test_iterator = Iterator.splits(
    (train_data, valid_data, test_data),
    batch_size=BATCH_SIZE,
    device=device, sort_key=lambda x: (len(x.src), len(x.trg)))


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.input_dim = input_dim  # 输入encoder的one-hot向量的维度
        self.emb_dim = emb_dim  # embedding层的维度，该层将one-hot向量转换成dense向量
        self.hid_dim = hid_dim  # 隐状态的维度
        self.n_layers = n_layers  # RNN的层数
        self.dropout = dropout  # 正则化参数，防止过拟合

        self.embedding = nn.Embedding(input_dim, emb_dim)  # 为什么要加这个embedding层呢？直接用emb_dim不行吗

        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src = [src sent len, batch size]

        embedded = self.dropout(self.embedding(src))

        # embedded = [src sentence length, batch size, emb dim]

        outputs, (hidden, cell) = self.rnn(embedded)

        # outputs = [src sent len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        # outputs are always from the top hidden layer

        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.dropout = dropout

        self.embedding = nn.Embedding(output_dim, emb_dim)

        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)

        self.out = nn.Linear(hid_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):
        print(hidden.shape)
        # input = [batch size]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        # n directions in the decoder will both always be 1, therefore:
        # hidden = [n layers, batch size, hid dim]
        # context = [n layers, batch size, hid dim]

        input = input.unsqueeze(0)  # 在第一维增加一个维度，将input的维度变为（1，batch size）
        # input = [1, batch size]


        embedded = self.dropout(self.embedding(input))  # dropout不是说output跟input的形状一样吗？
        # embedded = [1, batch size, emb dim]

        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))

        # output = [sent len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        # sent len and n directions will always be 1 in the decoder, therefore:
        # output = [1, batch size, hid dim]
        # hidden = [n layers, batch size, hid dim]
        # cell = [n layers, batch size, hid dim]
        prediction = self.out(output.squeeze(0))
        # prediction = [batch size, output dim]
        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        # 为了方便处理，所以将encoder和decoder的隐状态维度和层数设置成一样的值
        assert encoder.hid_dim == decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"


    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src = [src sent len, batch size]
        # trg = [trg sent len, batch size]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time

        batch_size = trg.shape[1]
        max_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        # tensor to store decoder outputs
        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)

        # last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden, cell = self.encoder(src)

        # first input to the decoder is the <sos> tokens
        input = trg[0, :]

        for t in range(1, max_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.max(1)[1]
            input = (trg[t] if teacher_force else top1)

        return outputs


INPUT_DIM = len(SRC.vocab)
OUTPUT_DIM = len(TRG.vocab)
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
HID_DIM = 512
N_LAYERS = 2
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5

enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)

model = Seq2Seq(enc, dec, device).to(device)


# 初始化权重
def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)


model.apply(init_weights)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print(f'The model has {count_parameters(model):,} trainable parameters')

optimizer = optim.Adam(model.parameters())

# 计算loss，注意要忽略padding token的loss
PAD_IDX = TRG.vocab.stoi['<pad>']

criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)


def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0

    for i, batch in enumerate(iterator):
        src = batch.src
        trg = batch.trg
        optimizer.zero_grad()
        output = model(src, trg)

        # trg = [trg sent len, batch size]
        # output = [trg sent len, batch size, output dim]
        # .view函数将多维展开成一维，方便loss函数计算。并且output[1:]和trg[1:]删去了<sos>token，因为无需计算<sos>token的loss
        output = output[1:].view(-1, output.shape[-1])
        trg = trg[1:].view(-1)

        # trg = [(trg sent len - 1) * batch size]
        # output = [(trg sent len - 1) * batch size, output dim]

        loss = criterion(output, trg)

        loss.backward()

        # 修剪梯度防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg

            output = model(src, trg, 0)  # turn off teacher forcing

            # trg = [trg sent len, batch size]
            # output = [trg sent len, batch size, output dim]

            output = output[1:].view(-1, output.shape[-1])
            trg = trg[1:].view(-1)

            # trg = [(trg sent len - 1) * batch size]
            # output = [(trg sent len - 1) * batch size, output dim]

            loss = criterion(output, trg)

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


# 计算每一个epoch所花费的时间
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def data_preprocess(sentence):
    return [SRC.vocab.stoi[word] for word in sentence.split(' ')] + [SRC.vocab.stoi[EOS_WORD]]


def greedy_search(src, model, TRG):
    eos_tok = TRG.vocab.stoi['<eos>']
    all_tokens = torch.zeros([0], device=device, dtype=torch.long)
    #batch_size = TRG.shape[1]
    max_len = MAX_LEN
    #trg_vocab_size = model.decoder.output_dim

    # tensor to store decoder outputs
    #outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(model.device)

    # last hidden state of the encoder is used as the initial hidden state of the decoder
    hidden, cell = model.encoder(src)

    # first input to the decoder is the <sos> tokens
    #input = torch.ones(1, 1 , 1) * TRG.vocab.stoi['<sos>']
    #lengths=torch.tensor([len(TRG.vocab.stoi['<sos>'])])
    #a=TRG.vocab.stoi['<sos>']
    input=torch.LongTensor([TRG.vocab.stoi['<sos>']])
    print("输入")
    print(input.shape)
    for t in range(1, max_len):
        prediction, hidden, cell = model.decoder(input, hidden, cell)
        print("prediction")
        print(prediction)       #得到所有单词的概率分布
        #prediction = [batch size, output dim]
        #output = F.softmax(prediction, dim=-2)  #有些dim等于-1，有些等于1？
        output = prediction.max(1)[1]        #从所有单词的概率分布中选择最大的概率，作为输出单词
        #outputs[t] = output
        print("output")
        output = output.long()       # 将floatTensor转成longTensor，方便下面的cat操作
        print(output)
        all_tokens = torch.cat((all_tokens, output), dim=0)
        print(all_tokens)
        # teacher_force = random.random() < teacher_forcing_ratio
        # top1 = output.max(1)[1]    #这个是最大的输出概率吗？
        input = output   # 如何把torch.Size([1,5618])变成torch.Size([1])


        """topv,topi=output.data.topk(1)
        topi=topi.view(-1)
        decoded_batch[:,t]=topi"""
    print("all_tokens")
    print(all_tokens)
    length = (all_tokens[0] == eos_tok).nonzero()[0]

    return ' '.join([TRG.vocab.itos[tok] for tok in all_tokens[0][1:length]])


# 输入一句话，测试模型输出
def translate(sentence, model, TRG):

    model.eval()
    indexed = data_preprocess(sentence)

    # 需要加view函数转置sentence矩阵，因为得到的是(batch_size,sen_len),而encoder的输入需要(sen_len,batch_size)
    sentence = Variable(torch.LongTensor([indexed])).view(-1,1)

    sentence = greedy_search(sentence, model, TRG)
    return sentence


N_EPOCHS = 0
CLIP = 1

best_valid_loss = float('inf')
print("best_valid_loss: ")
print(best_valid_loss)
for epoch in range(N_EPOCHS):

    start_time = time.time()

    train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
    print("train_loss: ")
    print(train_loss)

    valid_loss = evaluate(model, valid_iterator, criterion)

    print("valid_loss: ")
    print(valid_loss)
    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'tut1-model.pt')

    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

# load the parameters (state_dict) that gave our model the best validation loss and run it the model on the test set.
model.load_state_dict(torch.load('tut1-model.pt'))
sentence = 'hello.'
translate(sentence, model, TRG)
# test_loss = evaluate(model, test_iterator, criterion)

# print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')

# Initialize search module
