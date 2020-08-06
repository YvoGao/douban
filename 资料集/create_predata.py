from snownlp import SnowNLP
import csv
import os
import re

# 获得需要可以用作训练的44个电影,存到movieid里
sql_dir_war = r'../review2'
movieid = os.listdir(sql_dir_war)
for movie in movieid:
    if os.path.isfile(os.path.join(sql_dir_war, movie)):
        movieid.remove(movie)

# 获得每一个电影的影评内容，打分记录到文件里
for movie in movieid:
    txtfile = os.listdir(sql_dir_war+'/'+movie)
    scores = []

    # 打开文件夹里的所有文件，进行打分
    for txt in txtfile:
        file = open(sql_dir_war+'/'+movie+'/'+txt, "r", encoding='utf-8')
        contents = file.readlines()
        con =""
        for content in contents:
            if content != '':
                con += content[:len(content)-1]
        score = 0
        count = 0
        # 按照标点符号分隔
        pattern = r';|\?|!|。|！|；|·|…'
        contents = re.split(pattern, con)
        # 对该影评打分
        for content in contents:
            try:
                # 去掉空字符串
                if content.isspace():
                    continue
                s = SnowNLP(content)
                # 进行对没条评论情感分析打分累加
                score = score + s.sentiments
                # 对评论行数进行累加
                count = count + 1
            except:
                print(content,"error")
        tempscore = round(score / count, 3)
        print(tempscore)
        scores.append(tempscore)
    label = []
    label.append(movie)
    label += scores
    with open('../prelabel.csv', 'a+',  encoding='utf-8', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(label)
    csv_file.close()
    print(movie)


