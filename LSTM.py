import numpy as np
import csv
import torch
from torch import nn
from torch.autograd import Variable

# 制作训练集, 获得训练的值和真实值
def load_dataset():
    # 训练数据
    dataset = []
    # 真值
    tru_y = []
    # 电影id
    movies = []
    # 获得训练值
    csv_file = open(r'../label.csv', 'r')
    csv_reader_lines = csv.reader(csv_file)
    num = 0
    for temp in csv_reader_lines:
        data = []
        i = 0
        for t in temp:
            if i == 0:
                movies.append(t)
                num += 1
            else:
                print(t)
                data.append(float(t))
            i += 1
        while len(data)<100:
            data.append(0.5)
        dataset.append(data)


    # 获得真实值
    file = open(r'../movieScore.txt', 'r', encoding='utf-8')
    lables = file.readlines()
    relation = {}
    for la in lables:
        moid, name, score = la.split(' ')
        relation[moid] = score[:len(score)-1]
        print(relation)
        # print(moid, mname, score)
    index = []
    i = 0
    for m in movies:
        if relation[m] == '暂无评分':
            score = 5.0
            index.append(i)
        else:
            score = float(relation[m])
        tru_y.append(score)
        i += 1
    # print(len(tru_y))
    index = index[::-1]
    for ind in index:
        del dataset[ind]
        del tru_y[ind]
    return dataset, tru_y


def create_dataset(dataset, tru_y):
    dataX = dataset
    dataY = tru_y
    data_T = []
    for d in dataset:
        data_T.append(sum(d)/len(d))
    return np.array(dataX), np.array(dataY), np.array(data_T)

# 定义模型
class lstm_reg(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1, num_layers=4):
        super(lstm_reg, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers)
        self.reg = nn.Linear(hidden_size, output_size)

    def forward(self, x, y):
        x, _ = self.rnn(x)
        s, b, h = x.shape  # (seq, batch, hidden)
        x = x.view(s * b, h)  # 转化为线性层的输入方式
        x = self.reg(x)
        x = x.view(s, b, -1)
        x = x + y * 10
        return x

def train(epoch, train_x, train_y, train_t, optimizer):
    for e in range(epoch):
        var_x = Variable(train_x).float()
        var_y = Variable(train_y).float()
        var_t = Variable(train_t).float()
        # 前向传播
        out = net(var_x, var_t)
        # pdb.set_trace()
        loss = criterion(out, var_y)
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 10次记录一次loss
        if (e + 1) % 10 == 0:
            print('Epoch:{}, Loss:{:.5f}'.format(e + 1, loss.item()))
            # print('===> Saving models...')
            with open("../logs/loss.csv", 'a', encoding="UTF8", newline='') as f:
                writer = csv.writer(f)
                data = [e+1, loss.item()]
                writer.writerow(data)
            f.close()
    torch.save(net, '../logs/model.pkl')


if __name__ == '__main__':
    dataset, tru_y = load_dataset()
    train_X, train_Y, train_T = create_dataset(dataset, tru_y)
    train_X = train_X.reshape(-1, 1, 100)
    train_Y = train_Y.reshape(-1, 1, 1)
    train_T = train_T.reshape(-1, 1, 1)
    train_x = torch.from_numpy(train_X)
    train_y = torch.from_numpy(train_Y)
    train_t = torch.from_numpy(train_T)
    # 定义好网络结构，输入的维度是 100，因为我们使用100个影评的评分作为输入，隐藏层的维度可以任意指定，这里我们选的 120
    net = lstm_reg(100, 50)
    # net = torch.load('../logs/model.pkl')
    # net.eval()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)

    # 开始训练
    train(1000, train_x, train_y, train_t, optimizer)