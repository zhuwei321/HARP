import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from smp_model import *
from smp_data import *
from tqdm import tqdm
import csv
from tools import *
from transformers import logging
import warnings
import random
#from cb import *
import logging
import builtins
import torch.nn.functional as F
from expand_csv import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='Model trainer')
parser.add_argument('--warm_start_epoch', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--images_dir', type=str, default='')
parser.add_argument('--gt_path', type=str, default="0")
parser.add_argument('--data_files', type=str, default="")
parser.add_argument('--new_files', type=str, default="")
parser.add_argument('--seq_len', type=int, default=29)
parser.add_argument('--ckpt_path', type=str, default="ckpt_with_lai")
parser.add_argument('--result_file', type=str, default='all_result.csv')
parser.add_argument('--write', type=bool, default=False)
parser.add_argument('--train', type=bool, default=True)
parser.add_argument('--test', type=bool, default=False)
parser.add_argument('--K_fold', type=int, default=0)
parser.add_argument('--use_mlp', type=bool, default=False)

class CustomLoss(nn.Module):
    def __init__(self, initial_lambda1=1.0, initial_lambda2=1.0, initial_weight=1, epsilon=1e-6,
                 base_loss=nn.SmoothL1Loss(0.1), min_factor=0.5):
        """
        min_factor: 正则项最小保持的比例，防止权重降为0
        """
        super(CustomLoss, self).__init__()
        self.base_loss = base_loss
        self.initial_lambda1 = initial_lambda1
        self.initial_lambda2 = initial_lambda2
        self.lambda1 = initial_lambda1
        self.lambda2 = initial_lambda2
        self.epsilon = epsilon
        self.initial_weight = initial_weight
        self.peak_weight = initial_weight
        self.min_factor = min_factor

    def update_weights(self, current_step, total_steps):
        # 使用余弦退火策略更新权重，但设置一个下界，防止权重过低
        cosine_value = 0.5 * (1 + torch.cos(torch.tensor(current_step / total_steps * 3.141592653589793)))
        # 限制最小值，确保不会低于 min_factor * initial_value
        factor = torch.clamp(cosine_value, min=self.min_factor)
        self.lambda1 = self.initial_lambda1 * factor
        self.lambda2 = self.initial_lambda2 * factor
        self.peak_weight = self.initial_weight * factor

    def forward(self, outputs, targets, current_step, total_steps):
        self.update_weights(current_step, total_steps)

        # 基础损失
        base_loss = self.base_loss(outputs, targets)

        # 检测峰值并创建 one-hot 向量
        peak_indices_outputs = torch.argmax(outputs, dim=1)
        peak_indices_targets = torch.argmax(targets, dim=1)
        one_hot_outputs = F.one_hot(peak_indices_outputs, num_classes=outputs.size(1)).float()
        one_hot_targets = F.one_hot(peak_indices_targets, num_classes=targets.size(1)).float()
        peak_loss = F.l1_loss(one_hot_outputs, one_hot_targets)

        # 一阶导数损失
        first_derivative = outputs[:, 1:] - outputs[:, :-1]
        target_first_derivative = targets[:, 1:] - targets[:, :-1]
        first_derivative_loss = self.base_loss(first_derivative, target_first_derivative)

        # 二阶导数损失
        second_derivative = first_derivative[:, 1:] - first_derivative[:, :-1]
        target_second_derivative = target_first_derivative[:, 1:] - target_first_derivative[:, :-1]
        second_derivative_loss = self.base_loss(second_derivative, target_second_derivative)

        # 拉普拉斯平滑项
        laplacian_smoothing = self.epsilon * (
            torch.sum(torch.abs(first_derivative)) + torch.sum(torch.abs(second_derivative))
        )

        # L2 正则化
        l2_reg = torch.tensor(0.0).to(outputs.device)
        for param in model.parameters():
            l2_reg += torch.norm(param)

        # 总损失
        total_loss = base_loss + self.peak_weight * peak_loss + \
                     self.lambda1 * first_derivative_loss + \
                     self.lambda2 * second_derivative_loss + \
                     laplacian_smoothing
        return total_loss

    def get_lambda_values(self):
        return self.lambda1, self.lambda2


def load_data(args, K, n):
    random_seed = 23
    random_generator = random.Random(random_seed)
    data_files = args.data_files
    print(data_files)
    new_files = args.new_files 
    # expand_dataset_with_hypergraph(data_files,new_files, k=10, chunk_size=2000)
    # 加载数据集
    data_set = youtube_data_lstm(new_files, args.images_dir, args.gt_path)
    batch_size = args.batch_size

    # 计算每个折的大小
    fold_size = len(data_set) // K
    indices = list(range(len(data_set)))
    random_generator.shuffle(indices)

    val_start = n * fold_size
    train_start = (n + 1) * fold_size

    # 验证集与测试集采用同一索引
    val_indices = indices[val_start:train_start]
    test_indices = val_indices

    train_indices = [i for i in indices if i not in val_indices]

    train_set = Subset(data_set, train_indices)
    val_set = Subset(data_set, val_indices)
    test_set = Subset(data_set, test_indices)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=args.num_workers, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=args.num_workers, drop_last=True)

    return train_loader, val_loader, test_loader

def train(args, model, train_loader, val_loader):
    log_file = os.path.join(args.ckpt_path, f'train_{args.K_fold}.log')
    logging.basicConfig(filename=log_file, level=logging.INFO)
    
    # 使用优化后的 CustomLoss，设置 min_factor 保证权重不会降得过低
    loss_fn = CustomLoss(min_factor=0.5)
    lr = args.lr
    weight_decay = 0.001
    optimizer = optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=weight_decay)
    # 调整 ReduceLROnPlateau 的 patience 参数，防止学习率过早下降
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.2, patience=3, verbose=True, min_lr=1e-6)
    min_mae = 10
    # early stopping 计数器提前定义，跨 epoch 累计
    early_stop_count = 0
    total_steps = len(train_loader) * args.epochs

    for epoch in range(args.epochs):
        batch_train_losses = []
        model.train()
        preds = []
        labels = []
        
        for num, data in enumerate(tqdm(train_loader)):
            img = data['img'].to(device)
            text = data['text'].to(device) if isinstance(data['text'], torch.Tensor) else data['text']
            meta = data['meta'].to(device)
            cat = data['cat'].to(device) if isinstance(data['cat'], torch.Tensor) else data['cat']
            label = data['label'].to(device)
            retrieved_video_ids = data['retrieved_video_ids']
            retrieved_similarities = data['retrieved_similarities']
            retrieved_textual_features = data['retrieved_textual_features']
            retrieved_ep = data['retrieved_ep']

            optimizer.zero_grad()
            out = model(img, text, meta, cat, retrieved_video_ids, retrieved_similarities, retrieved_textual_features, retrieved_ep)
            current_step = epoch * len(train_loader) + num
            train_loss = loss_fn(out, label, current_step, total_steps)
            batch_train_losses.append(train_loss.item())

            train_loss.backward()
            # 可根据需要打开梯度裁剪
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            for i in range(out.shape[0]):
                preds.append(out[i].cpu().detach().numpy().tolist())
                labels.append(label[i].cpu().detach().numpy().tolist())

        avg_train_loss = round(sum(batch_train_losses) / len(batch_train_losses), 5)
        print(f'=====Epoch {epoch + 1} averaged training loss: {avg_train_loss:.6f}=====')
        out_print = print_output_seq(labels, preds)
        logging.info(f'=====Epoch {epoch + 1} averaged training loss: {avg_train_loss:.6f}=====')
        logging.info(out_print)

        # 验证阶段
        model.eval()
        batch_val_losses = []
        preds_val = []
        val_labels = []
        for num, data in enumerate(tqdm(val_loader)):
            current_step = epoch * len(train_loader) + num
            img = data['img'].to(device)
            text = data['text']  # 如果 text 是 tensor，则也需要 .to(device)
            meta = data['meta'].to(device)
            cat = data['cat']
            label = data['label'].to(device)
            retrieved_video_ids = data['retrieved_video_ids']
            retrieved_similarities = data['retrieved_similarities']
            retrieved_textual_features = data['retrieved_textual_features']
            retrieved_ep = data['retrieved_ep']

            out = model(img, text, meta, cat, retrieved_video_ids, retrieved_similarities, retrieved_textual_features, retrieved_ep)
            val_loss = loss_fn(out, label, current_step, total_steps)
            batch_val_losses.append(val_loss.item())

            for i in range(out.shape[0]):
                preds_val.append(out[i].cpu().detach().numpy().tolist())
                val_labels.append(label[i].cpu().detach().numpy().tolist())

        avg_val_loss = round(sum(batch_val_losses) / len(batch_val_losses), 5)
        scheduler.step(avg_val_loss)
        # 计算 MAE
        mae = mean_absolute_error(val_labels, preds_val)
        print(f'=====Epoch {epoch + 1} averaged val loss: {avg_val_loss:.6f}=====')
        out_print_val = print_output_seq(val_labels, preds_val)
        logging.info(f'=====Epoch {epoch + 1} averaged val loss: {avg_val_loss:.6f}=====')
        logging.info(out_print_val)

        torch.cuda.empty_cache()

        # 模型保存和 Early Stopping 逻辑
        if mae < min_mae:
            min_mae = mae
            early_stop_count = 0
            ckpt_name = f'{args.K_fold}-{epoch + 1}-{mae:.4f}.pth'
            torch.save(model.state_dict(), os.path.join(args.ckpt_path, ckpt_name))
            print('Saved model. Testing...')
        else:
            early_stop_count += 1
            # 连续 early_stop_count 个 epoch 无改进则提前停止训练
            if early_stop_count >= 20:
                print("Early stopping triggered. No significant improvement in MAE.")
                break
# test
def test(args, model, test_loader):
    model.eval()
    output_path = args.ckpt_path + '/' + args.result_file

    # save result
    if args.write:
        with open(output_path, 'w') as f:
            pass

    preds = []
    labels = []
    count = 0

    for num, data in enumerate(tqdm(test_loader)):

        img = data['img'].to(device)
        text = data['text']
        meta = data['meta'].to(device)
        cat = data['cat']

        label = data['label'].to(device)

        with torch.no_grad():
             out = model(img, text, meta, cat)

        count += 1
        # print(out)
        for i in range(out.shape[0]):
            preds.append(out[i].cpu().detach().numpy().tolist())
            labels.append(label[i].cpu().detach().numpy().tolist())
        # write result
        if args.write:
            with open(output_path, 'a+', newline='', encoding='UTF-8-sig') as f:
                for i in range(out.shape[0]):
                    new_lines = [data['id'][i], out[i].cpu().detach().numpy().tolist(), label[i].cpu().detach().numpy().tolist()]
                    writer = csv.writer(f)
                    writer.writerow(new_lines)
    print_output_seq(labels, preds)
    # return print_output_seq(labels, preds)


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    args = parser.parse_args()

    train_loader, val_loader, test_loader = load_data(args, 5, args.K_fold)

    if args.use_mlp is True:
        model = youtube_MLP(args.seq_len, args.batch_size)
    else:
        model = youtube_mLSTM(args.seq_len, args.batch_size)

    if args.test:
        import glob
        # model_files = glob.glob(os.path.join(args.ckpt_path, str(args.K_fold) + "*.pth"))[0]
        # model_dict = torch.load(model_files)
        model_files = os.path.join(args.ckpt_path, "")
        model_dict = torch.load(model_files)
        model.load_state_dict(model_dict)
        print('Loaded model ' + model_files)

    model = model.to(device)

    if args.train:
        train(args, model, train_loader, val_loader)
    elif args.test:
        test(args, model, test_loader)
    else:
        print(r"please choose 'train' or 'test'")
