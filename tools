import numpy as np
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch
import math
import os
import time
import csv
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA


def TFIDF(text):
    vectorizer = TfidfVectorizer()
    weight = vectorizer.fit_transform(text).toarray()
    # word = vectorizer.get_feature_names()
    return weight


def PCA_down(weight, dimension):
    pca = PCA(n_components=dimension)  # 初始化PCA
    vec = pca.fit_transform(weight)  # 返回降维后的数据
    return vec


# tools
def write_to_csv(result, list):
    with open(result, 'w', newline='', encoding='UTF-8-sig') as ff:
        for i in tqdm(range(len(list))):
            writer = csv.writer(ff)
            writer.writerow(list[i])


def time2hours(time1, time2):
    time1 = time.mktime(time.strptime(time1, "%Y/%m/%d %H:%M:%S"))
    time2 = time.mktime(time.strptime(time2, "%Y/%m/%d %H:%M:%S"))
    seconds = time2 - time1
    hours = seconds / 3600
    return hours


def get_time():
    time_tuple = time.localtime(time.time())
    for i in range(len(time_tuple)):
        if time_tuple[i] < 10:
            time_tuple[i] = "0" + str(i)
        else:
            time_tuple[i] = str(i)

    time_str = time_tuple[1] + time_tuple[2] + "_"+time_tuple[3] + ":" + time_tuple[4]
    return time_str


def add_fname(file_name, add_str):
    name = file_name.split('.')[0] + add_str + '.' \
           + file_name.split('.')[1]
    return name


def mape(actual, pred):
    actual, pred = np.array(actual), np.array(pred)
    return np.mean(np.abs((actual - pred) / actual)) * 100


def nmse(predictions, targets):
    differences = (predictions - targets) / targets
    differences_squared = differences ** 2
    mean_of_differences_squared = differences_squared.mean()
    rmse_val = np.sqrt(mean_of_differences_squared)

    return rmse_val


def print_output(labels, preds, verbose=True):
    mae = mean_absolute_error(labels, preds)
    mse = mean_squared_error(labels, preds)
    spearmanr_corr = stats.spearmanr(labels, preds)[0]
    # pearsonr_corr = stats.pearsonr(labels, preds)[0]
    #
    if verbose:
        print(mae)
        print(mse)
        print(spearmanr_corr)
    # print(pearsonr_corr)
    return mae, mse, spearmanr_corr


def print_output_seq(labels, preds):
    preds = np.array(preds)
    labels = np.array(labels)
    length = labels.shape[1]
    mae_list, mse_list, src_list = [], [], []
    for i in range(length):
        mae, mse, src = print_output(labels[:, i], preds[:, i], verbose=False)
        mae_list.append(mae)
        mse_list.append(mse)
        src_list.append(src)

    aver_mae = np.mean(np.array(mae_list))
    aver_mse = np.mean(np.array(mse_list))
    aver_src = np.mean(np.array(src_list))
    print(mae_list, aver_mae)
    print(mse_list, aver_mse)
    print(src_list, aver_src)
    return mae_list, mse_list, src_list, aver_mae, aver_mse, aver_src


def print_output_mape(labels, preds, verbose=True):
    mae = mean_absolute_error(labels, preds)
    mse = mean_squared_error(labels, preds)
    spearmanr_corr = stats.spearmanr(labels, preds)[0]
    mape1 = mape(labels, preds)

    if verbose:
        print(mae)
        print(mse)
        print(spearmanr_corr)
        print(mape1)

    return mae, mse, spearmanr_corr, mape1


def print_output_part(labels, preds):
    mae = mean_absolute_error(labels, preds)
    mse = mean_squared_error(labels, preds)
    spearmanr_corr = stats.spearmanr(labels, preds)[0]

    print(mae)
    print(mse)
    print(spearmanr_corr)

    l1 = []
    p1 = []
    for i in range(len(labels)):
        if labels[i] >= 1:
            l1.append(labels[i])
            p1.append(preds[i])
    mape1 = mape(l1, p1)
    print(mape1)
    return mae, mse, spearmanr_corr, mape1


def correct(x):
    if type(x) == int or type(x) == float:
        if x < 0:
            return 0
        else:
            return x
    if type(x) == list or type(x) == np.ndarray:
        c = []
        for i in x:
            if i < 0:
                c.append(0)
            else:
                c.append(i)
        if type(x) == list:
            return c
        if type(x) == np.ndarray:
            return np.array(c)


def v2p(array):
    num1, day1 = -2, 1
    num2, day2 = -1, 7

    for data in array:
        data[num1], data[num2] = float(data[num1]), float(data[num2])
        data[num1] = math.log(float(data[num1] / day1) + 1, 2)
        data[num2] = math.log(float(data[num2] / day2) + 1, 2)

    return array


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_optimizer(model, args):
    parameters = []
    for name, param in model.named_parameters():
        if 'fc' in name or 'class' in name or 'last_linear' in name or 'ca' in name or 'sa' in name:
            parameters.append({'params': param, 'lr': args.lr * args.lr_fc_times})
        else:
            parameters.append({'params': param, 'lr': args.lr})
    # parameters = model.parameters()
    if args.optimizer == 'sgd':
        return torch.optim.SGD(parameters,
                               # model.parameters(),
                               args.lr,
                               momentum=args.momentum, nesterov=args.nesterov,
                               weight_decay=args.weight_decay)
    elif args.optimizer == 'rmsprop':
        return torch.optim.RMSprop(parameters,
                                   # model.parameters(),
                                   args.lr,
                                   alpha=args.alpha,
                                   weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        return torch.optim.Adam(parameters,
                                # model.parameters(),
                                args.lr,
                                betas=(args.beta1, args.beta2),
                                weight_decay=args.weight_decay)
    # elif args.optimizer == 'radam':
    #     return RAdam(parameters, lr=args.lr, betas=(args.beta1, args.beta2),
    #                       weight_decay=args.weight_decay)

    else:
        raise NotImplementedError


def save_checkpoint(state, is_best, single=True, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    if single:
        fold = ''
    else:
        fold = str(state['fold']) + '_'
    cur_name = 'checkpoint.pth.tar'
    filepath = os.path.join(checkpoint, fold + cur_name)
    curpath = os.path.join(checkpoint, fold + 'model_cur.pth')

    torch.save(state, filepath)
    torch.save(state['state_dict'], curpath)

    if is_best:
        model_name = 'model_' + str(state['epoch']) + '_' + str(int(round(state['train_acc'] * 100, 0))) + '_' + str(
            int(round(state['acc'] * 100, 0))) + '.pth'
        model_path = os.path.join(checkpoint, fold + model_name)
        torch.save(state['state_dict'], model_path)


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    batch_size = target.size(0)
    # top1 accuracy
    maxk = max(topk)
    _, pred = output.topk(maxk, 1, True, True)  # 返回最大的k个结果（按最大到小排序）

    pred = pred.t()  # 转置

    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res = correct_k.mul_(100.0 / batch_size)
    return res
