import pandas as pd
import numpy as np
import csv
from tqdm import tqdm
import math
import seaborn
import random
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy import stats
import time
from datetime import datetime, date
from tools import *
from pylab import mpl



def mape(actual, pred):
    actual, pred = np.array(actual), np.array(pred)
    return np.mean(np.abs((actual - pred) / actual)) * 100

def write_to_csv(result, list):
    with open(result, 'w', newline='', encoding='UTF-8-sig') as ff:
        for i in tqdm(range(len(list))):
            writer = csv.writer(ff)
            writer.writerow(list[i])

def sat():
    all_lines1 = pd.read_csv('./bili_data5.csv')
    all_lines2 = pd.read_csv('./pd_data2.csv')
    all_lines1 = np.array(all_lines1)
    all_lines2 = np.array(all_lines2)
    sum1 = []
    sum2 = []

    for data in all_lines1:
        sum1.append(data[-1])
    for data in all_lines2:
        sum2.append(data[-1])

    mean1 = np.mean(sum1)
    std1 = np.std(sum1)
    mean2 = np.mean(sum2)
    std2 = np.std(sum2)

    print(mean1)
    print(mean2)
    print(std1)
    print(std2)


def addtag_bili():
    path1 = './utils/data_new3.csv'
    result = './utils/data_new4.csv'
    all_lines = pd.read_csv(path1)
    all_lines = np.array(all_lines)
    new_lines = []
    new_lines1 = []
    tagcount = []
    # i = 0
    for data in all_lines:
        # i += 1
        data[2] = str(data[2])
        # print(type(data[2]), i)
        if (data[2] is np.nan) or data[2] == 'nan':
            data[2] = data[1]
        else:
            data[2] = data[2] + ',' + str(data[1])
        data = np.append(data, len(data[3]))
        data = np.append(data, data[2].count(',') + 1)
        # tagcount.append(data[9])
        new_lines.append(data)

    # t_m = np.mean(tagcount)
    # t_s = np.std(tagcount)
    #
    # for data in new_lines:
    #     data[9] = (data[9] - t_m) / t_s
    #     new_lines1.append(data)

    with open(result, 'w', newline='', encoding='UTF-8') as ff:
        for i in tqdm(range(len(new_lines))):
            writer = csv.writer(ff)
            writer.writerow(new_lines[i])


def addtag_pd():
    all_lines = pd.read_csv('./pd_data3.csv')
    all_lines = np.array(all_lines)
    new_lines = []
    new_lines1 = []
    tagcount = []

    # aaa = all_lines[823][1] is np.nan

    for data in all_lines:
        if data[1] is np.nan:
            data[1] = '0'
            tag = '0'
        else:
            tag = data[1]
        data = np.insert(data, 2, tag, 0)
        if tag == '0':
            data = np.insert(data, 8, 0, 0)
        else:
            data = np.insert(data, 8, tag.count(',') + 1, 0)
        tagcount.append(data[8])
        new_lines.append(data)

    t_m = np.mean(tagcount)
    t_s = np.std(tagcount)

    for data in new_lines:
        data[8] = (data[8] - t_m) / t_s
        new_lines1.append(data)

    with open('./pd_data4.csv', 'w', newline='', encoding='UTF-8') as ff:
        for i in tqdm(range(len(new_lines1))):
            writer = csv.writer(ff)
            writer.writerow(new_lines1[i])


def exchange():
    df1 = pd.read_csv('./pd_data4.csv')
    df1 = np.array(df1)
    df1[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]] = df1[:, [0, 1, 2, 3, 4, 5, 8, 6, 7, 9]]
    with open('./pd_data5.csv', 'w', newline='', encoding='UTF-8') as ff:
        for i in range(len(df1)):
            writer = csv.writer(ff)
            writer.writerow(df1[i])
            print('write ', i)


def viewcount():
    # all_lines = pd.read_csv('./utils/dataset_7day.csv', delimiter='\t')
    # all_lines = pd.read_csv('./utils/dataset_7day.csv')
    # all_lines = np.array(all_lines)
    with open('./utils/dataset_14day.csv', 'r', newline='', encoding='utf-8-sig') as f:
        all_lines = f.readlines()

    views = []
    pop_sum = []
    for i in all_lines:
        id = int(i.split(',')[0].strip('"'))
        v = i.split(',')[-3]
        if v == '""':
            v = 0
        else:
            v = int(v.strip('"'))

        # if v == '\\N\r\n':
        #     v = 0
        # else:
        #     v = int(v.rstrip('\r\n').strip('"'))

        views.append([id, v])
        # pop = math.log(v/7, 2) + 1
    #     pop_sum.append(v)
    # v_m = np.mean(pop_sum)
    # v_s = np.std(pop_sum)
    # # views1 = []
    # for i in views:
    #     i[1] = (i[1]-v_m)/v_s
    #     # views1.append(i)

    with open('./utils/bili_14day.csv', 'w', newline='', encoding='UTF-8') as ff:
        for i in range(len(views)):
            writer = csv.writer(ff)
            writer.writerow(views[i])
            print('write ', i)


def ori_normalize():
    path1 = './utils/data_new5_ori.csv'
    result = './utils/data_new5_nor.csv'
    all_lines = pd.read_csv(path1, header=None)
    all_lines = np.array(all_lines)
    new_lines = []
    time_num = 4
    time_mean = 1645009821.68777
    time_std = 5297100.38230103
    followers_num = 5
    followers_mean = 23124.8296071084
    followers_std = 286979.069789721
    nima_num = 6
    nima_mean = 4.51660428731417
    nima_std = 0.405650235563969
    iipa_num = 7
    iipa_mean = 1.86071626090554
    iipa_std = 1.41480026930039
    lenth_num = 8
    lenth_mean = 21.4780214155559
    lenth_std = 13.6513958106551
    tag_num = 9
    tag_mean = 2.94711627049935
    tag_std = 9.57323859665051

    for data in all_lines:
        data[time_num] = (data[time_num]-time_mean)/time_std
        data[followers_num] = (data[followers_num]-followers_mean)/followers_std
        data[nima_num] = (data[nima_num]-nima_mean)/nima_std
        data[iipa_num] = (data[iipa_num]-iipa_mean)/iipa_std
        data[lenth_num] = (data[lenth_num]-lenth_mean)/lenth_std
        data[tag_num] = (data[tag_num]-tag_mean)/tag_std
        new_lines.append(data)

    with open(result, 'w', newline='', encoding='UTF-8-sig') as ff:
        for i in range(len(new_lines)):
            writer = csv.writer(ff)
            writer.writerow(new_lines[i])
            print('write ', i)


def errorcount():
    path1 = './utils/bili_data_1day7day_v.csv'
    all_lines = pd.read_csv(path1)
    all_lines = np.array(all_lines)
    count = 0
    new_lines = []
    num1 = 10
    num2 = 11
    for data in all_lines:
        # data[10] = float(data[10])
        data[num1], data[num2] = float(data[num1]), float(data[num2])
        # if (data[1] > data[2] and data[2] == 0):
        if (data[num1] > data[num2]):
            continue
        new_lines.append(data)

    with open(path1, 'w', newline='', encoding='UTF-8') as ff:
        for i in range(len(new_lines)):
            writer = csv.writer(ff)
            writer.writerow(new_lines[i])
            print('write ', i)


def delete0():
    path1 = './utils/bili_1_7.csv'
    all_lines = pd.read_csv(path1)
    all_lines = np.array(all_lines)
    zero_sum = []
    new_lines = []
    for data in all_lines:
        datac = data[11]
        datac = float(datac)
        if datac == 0:
            zero_sum.append(data)
        else:
            new_lines.append(data)
    print(len(zero_sum))
    print(len(new_lines))
    frac = 0.98  # how much of a/b do you want to exclude
    inds = set(random.sample(list(range(len(zero_sum))), int(frac * len(zero_sum))))
    zero_sum = [n for i, n in enumerate(zero_sum) if i not in inds]
    print(len(zero_sum))

    with open(path1, 'w', newline='', encoding='UTF-8') as ff:
        for i in range(len(zero_sum)):
            writer = csv.writer(ff)
            writer.writerow(zero_sum[i])

        for i in range(len(new_lines)):
            writer = csv.writer(ff)
            writer.writerow(new_lines[i])


def delete():
    path1 = './utils/bili_1day7day_max.csv'
    result = './utils/bili_1day7day_max.csv'
    # path1 = '../results/result1day7day_max_v.csv'
    # result = '../results/result1day7day_max_v.csv'
    all_lines = pd.read_csv(path1)
    all_lines = np.array(all_lines)
    zero_sum = []
    new_lines = []
    for data in all_lines:
        datac = data[0]
        datac = float(datac)
        if datac == 683063526:
            zero_sum.append(data)
        else:
            new_lines.append(data)

    with open(result, 'w', newline='', encoding='UTF-8') as ff:
        for i in range(len(new_lines)):
            writer = csv.writer(ff)
            writer.writerow(new_lines[i])


def merge0():
    path1 = './utils/bili_1day_v.csv'
    path2 = './utils/bili_1m_v.csv'
    result_path = './utils/bili_1day1m.csv'
    ###参数合并
    df10 = pd.DataFrame(pd.read_csv(path1, header=None, names=['key', '1']))
    df11 = pd.DataFrame(pd.read_csv(path2, header=None, names=['key', '1']))

    merge2 = pd.merge(left=df10, right=df11, on=['key'],
                      how="inner")  # 实现满足点数与区域都匹配两个条件时候合并表格，只匹配一个条件采用left_on="plot_no",right_on="plot_no"
    # merge2 = pd.concat([df10, df11], axis=1)

    merge2.to_csv(result_path, index=False, header=None)

def count_columns(path):
    all_lines = pd.read_csv(path, header=None)
    all_lines = np.array(all_lines)
    column_num = []
    for i in range(all_lines.shape[1]):
        column_num.append(str(i))
    return column_num


def merge():
    # path1 = './bili_data6.csv'
    path1 = '1230_ori_l.csv'
    path2 = 'views0204_cl.csv'
    result_path = '1230_30seq.csv'
    ###参数合并
    df10 = pd.DataFrame(pd.read_csv(path1, header=None, names=count_columns(path1)))
    # df10 = pd.DataFrame(pd.read_csv(path1, header=None, names=['key', '1', '2', '3', '4', '5','6']))
    # df10 = pd.DataFrame(pd.read_csv(path1, header=None, names=['key', '1', '2']))
    # df10 = df10[['key', '1', '2', '4', '6', '7']]
    # df10 = df10[['key']]
    df11 = pd.DataFrame(pd.read_csv(path2, header=None, names=count_columns(path2)))
    # df11 = df11[['0', '1']]
    # df11[['key']] = df11[['key']].astype(int)
    # df10.to_csv(result_path, index=False, header=None)
    merge2 = pd.merge(left=df10, right=df11, on=['0'],
                      how="inner")  # 实现满足点数与区域都匹配两个条件时候合并表格，只匹配一个条件采用left_on="plot_no",right_on="plot_no"
    # merge2 = pd.concat([df10, df11], axis=1)
    # cols = merge2.columns[[0,1,2,3,4,5,12,7,8,9,10,11]]
    # merge2 = merge2[cols]
    merge2.to_csv(result_path, index=False, header=None, encoding='utf-8-sig')

def adjust():
    path1 = './utils/data_new2.csv'
    result_path = './utils/data_new3.csv'
    df10 = pd.DataFrame(pd.read_csv(path1, header=None, names=['key', '1', '2', '3', '4', '5','6']))
    df10 = df10[['key', '1', '2', '4', '5', '6']]
    df10.to_csv(result_path, index=False, header=None)

def get_label():
    path1 = './utils/bili_1day7day_max_v.csv'
    result = './utils/bili_1day7day_max.csv'
    # result = path1
    all_lines = pd.read_csv(path1)
    all_lines = np.array(all_lines)
    v_sum = []
    new_lines = []
    num1, day1 = -2, 1
    num2, day2 = -1, 7

    for data in all_lines:
        data[num1], data[num2] = float(data[num1]), float(data[num2])
        data[num1] = math.log(float(data[num1] / day1) + 1, 2)
        data[num2] = math.log(float(data[num2] / day2) + 1, 2)
        # v_sum.append(data[10])

    # v_m = np.mean(v_sum)
    # v_s = np.std(v_sum)
    # # for data in all_lines:
    # #     data[10] = (data[10]-v_m)/v_s

    with open(result, 'w', newline='', encoding='UTF-8') as ff:
        for i in range(len(all_lines)):
            writer = csv.writer(ff)
            writer.writerow(all_lines[i])
            print('write ', i)


def get_label_c():
    path1 = './utils/bili_data_7day.csv'
    all_lines = pd.read_csv(path1)
    all_lines = np.array(all_lines)
    v_sum = []
    new_lines = []
    for data in all_lines:
        # data[10], data[11] = int(data[10]), int(data[11])
        data[10] = float(data[10])
        data[10] = math.pow(2, data[10] - 1)
        data[10] = math.log(data[10] + 1, 2)

    with open('./bili_data_total.csv', 'w', newline='', encoding='UTF-8') as ff:
        for i in range(len(all_lines)):
            writer = csv.writer(ff)
            writer.writerow(all_lines[i])
            print('write ', i)


def look():
    path1 = './data_s/0918_1d7d_pn.csv'
    result = './data_s/0918_1d7d_pn0.csv'
    num = 11
    all_lines = pd.read_csv(path1, header=None)
    all_lines = np.array(all_lines)
    new_lines = []
    for data in all_lines:
        data[num] = float(data[num])
        if (data[num] >= 0 and data[num] < math.log(100/7+1,2)):
            new_lines.append(data)

    write_to_csv(result, new_lines)

def change_col20():
    path1 = 'test0111.csv'
    result = 'utils/0111_pn_.csv'

    all_lines = pd.read_csv(path1, header=None)

    start_col = 0
    end_col = 3
    for i in range(start_col, end_col + 1):
        new_lines = []
        all_lines1 = np.array(all_lines)
        for data in all_lines1:
            data[i] = '0'
            new_lines.append(data)
        write_to_csv(add_fname(result, str(i)), new_lines)

    start_col = 4
    end_col = 14
    for i in range(start_col, end_col + 1):
        new_lines = []
        all_lines1 = np.array(all_lines)
        for data in all_lines1:
            data[i] = 0
            new_lines.append(data)
        write_to_csv(add_fname(result, str(i)), new_lines)


def change2c():
    path1 = './utils/data_new5_7day.csv'
    result = './utils/data_new5_2c.csv'
    value = 0
    all_lines = pd.read_csv(path1, header=None)
    all_lines = np.array(all_lines)
    sum1 = 0
    sum2 = 0
    for data in all_lines:
        data[10] = float(data[10])
        if data[10] <= value:
            data[10] = 0
        else:
            data[10] = 1

    with open(result, 'w', newline='', encoding='UTF-8-sig') as ff:
        for i in range(len(all_lines)):
            writer = csv.writer(ff)
            writer.writerow(all_lines[i])
            print('write ', i)

def change3c():
    # path1 = './train/train1d7d_4c.csv'
    path1 = './utils/0809_0825normal.csv'
    result = './utils/0809_0825normal_3c.csv'
    value1 = 50
    value2 = 500
    all_lines = pd.read_csv(path1,header=None)
    all_lines = np.array(all_lines)
    sum0 = 0
    sum1 = 0
    sum2 = 0
    num = 11
    for data in all_lines:
        data[num] = float(data[num])
        if data[num] <= value1:
            data[num] = 0
            sum0 += 1
        if (data[num] > value1 and data[num] <= value2):
            data[num] = 1
            sum1 += 1
        if data[num] > value2:
            data[num] = 2
            sum2 += 1


    print(sum0,sum1,sum2)

    with open(result, 'w', newline='', encoding='UTF-8-sig') as ff:
        for i in range(len(all_lines)):
            writer = csv.writer(ff)
            writer.writerow(all_lines[i])

def change4c():
    path1 = '0111_p.csv'

    all_lines = pd.read_csv(path1,header=None)
    all_lines = np.array(all_lines)
    sum = np.zeros(4)
    value = np.array([0, 100, 1000, 10000, 1e+10])
    value = np.log2(value/7 + 1)

    num = -1
    for i in range(len(value) - 1):
        new_lines = []
        for data in all_lines:
            data[num] = float(data[num])
            if value[i] <= data[num] <= value[i+1]:
                sum[i] += 1
                new_lines.append(data)

        write_to_csv('utils/0111_p'+str(i)+'.csv', new_lines)

    print(sum)



def select():
    path1 = '0111_pn.csv'
    result = '0-8.csv'
    num = 8
    value = 8/24
    # value = np.log(value/7+1)

    all_lines = pd.read_csv(path1)
    all_lines = np.array(all_lines)
    new_lines = []

    for data in all_lines:
        data[num] = float(data[num])
        if (data[num] < value):
            new_lines.append(data)

    write_to_csv(result,new_lines)

def step_select():
    path1 = 'data_s/test/test0918_pn.csv'
    result = 'data_s/utils/0918_17_t.csv'
    num = 11
    value = [0, 100, 1000, 10000, 1e+8]
    for i in range(len(value)):
        value[i] = np.log(value[i]/7+1)

    all_lines = pd.read_csv(path1)
    all_lines = np.array(all_lines)

    for a in range(len(value)-1):
        new_lines = []
        for data in all_lines:
            data[num] = float(data[num])
            if (data[num] >= value[a] and data[num] < value[a+1]):
                new_lines.append(data)

        write_to_csv(add_fname(result, str(a)), new_lines)


def step_count():
    path1 = 'b1225.csv'
    num = 4
    value = [0,4,8,12,16,20,24]

    all_lines = pd.read_csv(path1)
    all_lines = np.array(all_lines)

    for a in range(len(value)-1):
        new_lines = []
        for data in all_lines:
            data[num] = float(data[num])
            if (data[num] >= value[a] and data[num] < value[a+1]):
                new_lines.append(data)
        print(len(new_lines))



def distribution():
    mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体（解决中文无法显示的问题）
    mpl.rcParams['axes.unicode_minus'] = False

    path1 = 'b1225.csv'
    all_lines = pd.read_csv(path1)
    all_lines = np.array(all_lines)
    hours = []

    for data in all_lines:
        # time1 = data[-3]
        # time2 = data[-1]
        # hours.append(time2hours(time1, time2))
        hours.append(data[4])

    m = np.mean(hours)
    s = np.std(hours)
    mid = np.median(hours)

    print(m)
    print(s)
    print(mid)

    plt.figure(dpi=120)
    # seaborn.set(style='dark')
    # seaborn.set_style("dark", {"axes.facecolor": "#e9f3ea"})  # 修改背景色
    seaborn.distplot(hours, norm_hist=True, hist=False, kde=True)
    plt.grid(color='grey')
    plt.xlabel('时间/小时', size=18)
    plt.ylabel('占比/%', size=18)
    plt.xticks(size=14)
    plt.yticks(size=14)
    plt.show()




def compete():

    preds = []
    labels = []
    path1 = './data_s/0918_1d7d_p.csv'
    # path1 = './train/train0825_e.csv'
    all_lines = pd.read_csv(path1)
    all_lines = np.array(all_lines)
    num1 = -2
    num2 = -1
    for data in all_lines:
        if data[num2] >= 1:
        # if data[num2] >= 100000 and data[num2] < 100000000:
            preds.append(data[num1])
            labels.append(data[num2])

    print_output1(labels, preds)
    # mape1 = mape(labels, preds)
    # print(mape1)

    # preds = []
    # labels = []
    # for data in all_lines:
    #     if data[num2] > 10000:
    #         preds.append(data[num1])
    #         labels.append(data[num2])
    #
    # mape1 = mape(labels, preds)
    # print(mape1)
    # mae = mean_absolute_error(labels, preds)
    # mse = mean_squared_error(labels, preds)
    # spearmanr_corr = stats.spearmanr(labels, preds)[0]
    # print(mae)
    # print(mse)
    # print(spearmanr_corr)

def step_compete(path):

    preds = []
    labels = []
    path1 = path
    # path1 = './train/train0825_e.csv'
    all_lines = pd.read_csv(path1)
    all_lines = np.array(all_lines)
    num1 = 1
    num2 = 2
    for data in all_lines:
        if data[num2] >= 0 and data[num2] < 10:
            preds.append(data[num1])
            labels.append(data[num2])
    print('views 0-10:')
    print_output(labels, preds)
    
    
    start_num = 10
    for i in range(5): # 1e+5
        preds = []
        labels = []
        for data in all_lines:
            if data[num2] >= start_num and data[num2] < start_num*10:
                preds.append(data[num1])
                labels.append(data[num2])

        print('views {}-{}:'.format(start_num,start_num*10))
        print_output1(labels, preds)
        start_num = start_num*10
        
        

def pre_v(path):
    path1 = path
    result = './data_s/' + path1
    all_lines = pd.read_csv(path1)
    all_lines = np.array(all_lines)
    num1 = 1
    num2 = 2
    # day1 = 14
    day2 = 7
    # day3 = 30
    day7pop = []
    new_lines = []
    for data in all_lines:
        # day1_v = data[num1]

        # data[num1] = math.log((float(data[num1]) / day1) + 1, 2)

        day2_v = (math.pow(2, data[num1]) - 1) * day2
        true_v = (math.pow(2, data[num2]) - 1) * day2
        # # day3_v = (math.pow(2, data[num1]) - 1) * day3
        #
        data = np.append(data, [day2_v,true_v])
        new_lines.append(data)
    with open(result, 'w', newline='', encoding='UTF-8') as ff:
        for i in tqdm(range(len(new_lines))):
            writer = csv.writer(ff)
            writer.writerow(new_lines[i])

def pre_ev():
    path1 = 'result_gls_l.csv'
    result = './data_s/' + path1
    all_lines = pd.read_csv(path1)
    all_lines = np.array(all_lines)
    num1 = 1
    num2 = 2

    new_lines = []
    for data in all_lines:

        day2_v = np.exp(data[num1]) - 1
        true_v = np.exp(data[num2]) - 1
        # # day3_v = (math.pow(2, data[num1]) - 1) * day3
        #
        data = np.append(data, [day2_v, true_v])
        new_lines.append(data)
    with open(result, 'w', newline='', encoding='UTF-8') as ff:
        for i in tqdm(range(len(new_lines))):
            writer = csv.writer(ff)
            writer.writerow(new_lines[i])

def split_3c():
    path1 = './utils/0809_cla.csv'
    result0 = './utils/0809_0.csv'
    result1 = './utils/0809_1.csv'
    result2 = './utils/0809_2.csv'

    all_lines = pd.read_csv(path1, header=None)
    all_lines = np.array(all_lines)
    num = 12
    c0, c1, c2 = [], [], []
    for data in all_lines:
        if data[num] == 0:
            c0.append(data)
        if data[num] == 1:
            c1.append(data)
        if data[num] == 2:
            c2.append(data)

    write_to_csv(result0, c0)
    write_to_csv(result1, c1)
    write_to_csv(result2, c2)



def time_strp():
    path1 = './utils/data_new4.csv'
    result = './utils/data_new5.csv'
    all_lines = pd.read_csv(path1, header=None)
    all_lines = np.array(all_lines)

    new_lines1 = []

    for data in all_lines:
        try:
            struct_time = time.strptime(data[4], '%Y/%m/%d %H:%M:%S')
            data[4] = time.mktime(struct_time)
        except:
            data[4] = 1510408320.0

        new_lines1.append(data)

    with open(result, 'w', newline='', encoding='UTF-8-sig') as ff:
        for i in tqdm(range(len(new_lines1))):
            writer = csv.writer(ff)
            writer.writerow(new_lines1[i])

def get_log():
    all_lines = pd.read_csv('data_s/0918_1d7d.csv')
    all_lines = np.array(all_lines)
    new_lines = []
    result = 'data_s/0918_1d7d_l.csv'

    for data in all_lines:
        for num in range(10, 12):
            if (data[num] == '\\N' or '') or (data[num] is np.nan):
                data[num] = 0
            else:
                data[num] = np.log(data[num] + 1)

    with open(result, 'w', newline='', encoding='UTF-8-sig') as ff:
        for i in tqdm(range(len(all_lines))):
            writer = csv.writer(ff)
            writer.writerow(all_lines[i])

def look_relation():
    all_lines = pd.read_csv('data_s/0918_1d_ori.csv')
    all_lines = np.array(all_lines)
    name = ["粉丝", "一天评论", "一天点赞", "一天收藏"]
    p = [3, 5, 6, 7]
    print("与一天播放量的相关性:")
    for i in range(len(p)):

        x = all_lines[:, p[i]]
        y = all_lines[:, -1]

        PLCC = stats.pearsonr(x, y)[0]
        SROCC = stats.spearmanr(x, y)[0]
        KROCC = stats.kendalltau(x, y)[0]

        print(" ", name[i])
        print("     PLCC:", PLCC)
        print("     SROCC:", SROCC)
        print("     KROCC:", KROCC)

    print("其中，\n"
          "PLCC皮尔逊相关系数，体现线性相关性\n"
          "SROCC斯皮尔曼等级相关，体现单调关系相关性\n"
          "KROCC肯德尔秩相关系数，体现排序相关性")


def src_about_30seq():
    path = 'views0204_cl.csv'
    all_lines = pd.read_csv(path, header=None)
    all_lines = np.array(all_lines)
    d1 = all_lines[:, 1]
    seq = all_lines[:, 1:]
    _, cols = seq.shape
    src_list = []
    for i in range(cols):
        s = seq[:, i]
        spearmanr_corr = stats.spearmanr(d1, s)[0]
        src_list.append(spearmanr_corr)
    print(src_list)

    # plt.ylim(0.8, 1.1)
    plt.plot(src_list)
    plt.show()


def jiaoji():
    # 读取第一个CSV文件
    file1 = 'SMTPD/basic_content.csv'  # 请替换成第一个CSV文件的路径
    df1 = pd.read_csv(file1)

    # 读取第二个CSV文件
    file2 = 'SMTPD/popularity.csv'  # 请替换成第二个CSV文件的路径
    df2 = pd.read_csv(file2, header=None)

    # 提取两个DataFrame的ID列
    id_column1 = df1['video_id']
    id_column2 = df2[0]

    # 计算交集
    intersection = pd.Series(list(set(id_column1) & set(id_column2)))

    # 使用isin()函数过滤数据
    df1_filtered = df1[df1['video_id'].isin(intersection)].drop_duplicates('video_id')
    df2_filtered = df2[df2[0].isin(intersection)].drop_duplicates(0)

    print(df2_filtered.shape[1])
    day_list = ["Day " + str(i+1) for i in range(df2_filtered.shape[1]-1)]
    header_d = ["video_id"]
    header_d.extend(day_list)
    print(header_d)
    df1_filtered.to_csv(file1.rstrip(".csv")+"_1"+".csv", index=False, header=df1.columns)
    df2_filtered.to_csv(file2.rstrip(".csv")+"_1"+".csv", index=False, header=header_d)

    # 打印交集的数量
    print("交集的数量:", len(intersection))


if __name__ == '__main__':
    # addtag_bili()
    # exchange()
    # viewcount()
    # errorcount()
    # delete0()
    # delete()
    # merge0()
    # time_strp()
    # ori_normalize()

    # look()
    # change_col20()
    # change2c()
    # change3c()
    # change4c()
    # select()
    # step_select()
    # step_count()

    # merge()
    # adjust()
    # errorcount()
    # delete0()
    # get_label()
    # test()
    # delete0()
    # split_3c()
    # get_label_c()
    # get_log()
    # distribution()
    # compete()
    # src_about_30seq()
    # look_relation()
    # pre_v('result_gls_p.csv')
    # step_compete('result_gls_v.csv')
    jiaoji()


