import numpy as np
import csv
import torch
from torch import nn
from torch.autograd import Variable
from pyecharts.charts import Line
import pyecharts.options as opts
from pyecharts.globals import ThemeType
# 制作训练集, 获得训练的值和真实值
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

    moviesname = []
    rela = {}
    # 获得真实值
    file = open(r'../movieScore.txt', 'r', encoding='utf-8')
    lables = file.readlines()
    relation = {}
    for la in lables:
        moid, name, score = la.split(' ')
        relation[moid] = score[:len(score) - 1]
        rela[moid] = name
        # print(relation)
        # print(moid, mname, score)
    index = []
    i = 0
    for m in movies:
        moviesname.append(rela[m])
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
        del moviesname[ind]
    return dataset, tru_y, moviesname


def create_dataset(dataset, tru_y):
    dataX = dataset
    dataY = tru_y
    data_T = []
    for d in dataset:
        data_T.append(sum(d) / len(d))
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

if __name__ == '__main__':
    model = lstm_reg(100, 150)
    # 模型类必须在别的地方定义
    model = torch.load('../logs/model.pkl')
    model.eval()
    dataset, tru_y, movies = load_dataset()
    data_X, data_Y, data_T= create_dataset(dataset, tru_y)
    print(data_X.shape, data_Y.shape)
    data_X = data_X.reshape(-1, 1, 100)
    data_T = data_T.reshape(-1, 1, 1)
    data_X = torch.from_numpy(data_X)
    data_T = torch.from_numpy(data_T)
    var_data = Variable(data_X).float()
    var_T = Variable(data_T).float()
    pred_test = model(var_data, var_T)  # 测试集的预测结果
    # 改变输出的格式
    pred_test = pred_test.view(-1).data.numpy()
    p = pred_test.tolist()
    t = data_Y.tolist()

    Line = (
        Line(init_opts=opts.InitOpts(bg_color='rgb(255,255,255)', theme=ThemeType.INFOGRAPHIC))
            .add_xaxis(movies)
            .add_yaxis("真实评分", t)
            .add_yaxis("预测评分", p)
            .set_global_opts(title_opts=opts.TitleOpts(title="训练集结果", subtitle=""),
                             xaxis_opts=opts.AxisOpts(
                                 axislabel_opts=opts.LabelOpts(rotate=30)),
                             toolbox_opts=opts.ToolboxOpts(is_show=True))
                  # datazoom_opts=opts.DataZoomOpts(is_show=True),
            .set_series_opts(label_opts=opts.LabelOpts(is_show=False))
    )

    Line.render(r'../logs/result.html')
