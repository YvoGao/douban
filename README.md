# douban
对豆瓣中51个电影的100条影评通过Snownlp进行情感分析，通过这100条数据
利用LSTM进行预测电影的评分
挑选影评再次的电影进行训练预测，避免因为100条数据太少不足以预测的情况。
＃环境
python== 3.6
snownlp 
torch = 0.4+
pyecharts = 1.0+
