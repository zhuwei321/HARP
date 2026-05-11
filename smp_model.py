import numpy as np
import torch
# from pytorch_pretrained_bert import BertTokenizer, BertModel
from transformers import BertTokenizer, BertModel, logging
from torchvision import models
from torch import nn
import torch.nn.functional as F
import fasttext as ft
from collections import deque, defaultdict
import os
import pdb
import langid
from typing import Tuple, Optional, List
from PIL import Image, ImageFile
from torchvision import transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
from einops.layers.torch import Rearrange

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.set_verbosity_warning()
logging.set_verbosity_error()



# 特征整合
class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, tokens, input_ids, input_mask, input_type_ids):
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = input_type_ids


# BERT的分词处理
def convert_examples_to_features(examples, seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example)

        if len(tokens_a) > seq_length - 2:
            tokens_a = tokens_a[0:(seq_length - 2)]

        tokens = []
        input_type_ids = []
        tokens.append("[CLS]")
        input_type_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            input_type_ids.append(0)
        tokens.append("[SEP]")
        input_type_ids.append(0)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < seq_length:
            input_ids.append(0)
            input_mask.append(0)
            input_type_ids.append(0)

        assert len(input_ids) == seq_length
        assert len(input_mask) == seq_length
        assert len(input_type_ids) == seq_length

        features.append(
            InputFeatures(
                tokens=tokens,
                input_ids=input_ids,
                input_mask=input_mask,
                input_type_ids=input_type_ids))
    return features


# BERT的词向量提取过程
def bert_feature(examples, model, tokenizer, seq_length=64):
    features = convert_examples_to_features(
        examples=examples, seq_length=seq_length, tokenizer=tokenizer)

    input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long).to(device)
    input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long).to(device)
    outputs = model(input_ids, token_type_ids=None, attention_mask=input_mask)
    pooled_output = outputs[1]

    return pooled_output


def ft_feature(examples, model):
    ft_list = []
    for t in examples:
        vec = model.get_word_vector(t)
        ft_list.append(vec)
    out = torch.tensor(ft_list).to(device)
    return out

def fourier_transform_constraint(x):
    """频域正则化模块"""
    fft = torch.fft.rfft(x, dim=-1)
    amp = torch.abs(fft)
    phase = torch.angle(fft)
    # 抑制高频噪声
    amp[:, 4:] *= 0.5  # 保留前4个主频
    return torch.fft.irfft(amp * torch.exp(1j*phase), n=x.size(-1), dim=-1)
        
class ResidualEnhancedLSTMCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, time_window: int = 3, bias: bool = True) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.time_window = time_window
        self.stage_thresholds = (3, 15)  # 阶段分界点: (stage1_end, stage2_end)

        # 原始门控参数
        self.W_i = nn.Parameter(nn.init.xavier_normal_(torch.empty(input_size, hidden_size)))
        self.W_f = nn.Parameter(nn.init.xavier_normal_(torch.empty(input_size, hidden_size)))
        self.W_o = nn.Parameter(nn.init.xavier_normal_(torch.empty(input_size, hidden_size)))
        self.W_q = nn.Parameter(nn.init.xavier_normal_(torch.empty(input_size, hidden_size)))
        self.W_k = nn.Parameter(nn.init.xavier_normal_(torch.empty(input_size, hidden_size)))
        self.W_v = nn.Parameter(nn.init.xavier_normal_(torch.empty(input_size, hidden_size)))

        # 双门控参数（增加残差投影）
        self.W_i2 = nn.Parameter(nn.init.xavier_normal_(torch.empty(hidden_size, hidden_size)))
        self.W_o2 = nn.Parameter(nn.init.xavier_normal_(torch.empty(hidden_size, hidden_size)))
        self.W_res = nn.Parameter(nn.init.xavier_normal_(torch.empty(hidden_size, hidden_size)))  # 新增残差投影
        
        # 新增时间维度残差参数
        self.W_t_res = nn.Parameter(nn.init.orthogonal_(torch.empty(hidden_size, hidden_size)))  # 正交初始化
        self.temporal_gate = nn.Sequential(
            nn.Linear(2*hidden_size, hidden_size),
            nn.Sigmoid()
        )

        # 状态感知参数
        self.alpha_layer = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

        # 残差门控参数
        self.res_gate = nn.Linear(2*hidden_size, hidden_size)  # 新增残差门控

        if bias:
            self.B_i = nn.Parameter(torch.zeros(hidden_size))
            self.B_f = nn.Parameter(torch.zeros(hidden_size))
            self.B_o = nn.Parameter(torch.zeros(hidden_size))
            self.B_q = nn.Parameter(torch.zeros(hidden_size))
            self.B_k = nn.Parameter(torch.zeros(hidden_size))
            self.B_v = nn.Parameter(torch.zeros(hidden_size))
            self.B_i2 = nn.Parameter(torch.zeros(hidden_size))
            self.B_o2 = nn.Parameter(torch.zeros(hidden_size))
            self.B_res = nn.Parameter(torch.zeros(hidden_size))  # 新增残差偏置

    def forward(self, x, states, t):
        """新增参数t表示当前时间步（从0开始）"""
        C, n, m, h_prev, h_queue = states
        h_prev_prev = h_queue[-1] if len(h_queue) > 0 else torch.zeros_like(h_prev)
        
        # ====== 阶段判断 ======
        if t < self.stage_thresholds[0]:
            stage = 1  # 前3天
        elif t < self.stage_thresholds[1]:
            stage = 2  # 第4-15天
        else:
            stage = 3  # 第15天后

        # ====== 阶段2启用时间窗口残差计算 ======
        if stage == 2 and len(h_queue) >= self.time_window:
            # 滑动窗口残差计算
            h_temporal = torch.stack([h_queue[-i] @ self.W_t_res for i in range(1, self.time_window+1)])
            h_res = torch.mean(h_temporal, dim=0)
            
            # 动态门控融合
            gate_t = self.temporal_gate(torch.cat([h_prev, h_res], dim=1))
            h_prev = gate_t * h_prev + (1 - gate_t) * h_res  # 公式(1)

        # 残差投影（将输入映射到隐空间）
        x_res = torch.matmul(x, self.W_res) + self.B_res  # [batch, hidden]

        # 状态变化感知（结合残差信息）
        delta = h_prev - x_res  # 比较隐状态与输入残差投影
        alpha = self.alpha_layer(delta)*0.5 + 0.5

        # 双输入门（引入残差连接）
        i_tilda1 = torch.matmul(x, self.W_i) + self.B_i
        i_tilda2 = torch.matmul(h_prev_prev, self.W_i2) + self.B_i2
        i_tilda = alpha * (i_tilda1 + x_res) + (1-alpha) * i_tilda2  # 输入残差增强

        # 其他门计算
        f_tilda = torch.matmul(x, self.W_f) + self.B_f
        o_tilda = torch.matmul(x, self.W_o) + self.B_o
        q_t = torch.matmul(x, self.W_q) + self.B_q
        
        epsilon = 1e-8
        k_t = torch.matmul(x, self.W_k) / torch.sqrt(torch.tensor(self.hidden_size) + epsilon)
        if self.bias: k_t += self.B_k
            
        v_t = torch.matmul(x, self.W_v) + self.B_v

        # 激活函数
        i_t = torch.exp(i_tilda)
        f_t = torch.sigmoid(f_tilda)
        o_t = torch.sigmoid(o_tilda)

        # 状态更新（保留原始LSTM核心结构）
        m_t = torch.max(torch.log(f_t) + m, torch.log(i_t))
        i_prime = torch.exp(i_tilda - m_t)

        C_t = f_t.unsqueeze(-1)*C + i_prime.unsqueeze(-1)*torch.einsum("bi,bk->bik", v_t, k_t)
        n_t = f_t*n + i_prime*k_t

        # 双输出门（残差增强）
        normalize_inner = torch.diagonal(torch.matmul(n_t, q_t.T))
        divisor = torch.max(torch.abs(normalize_inner), torch.ones_like(normalize_inner))
        h_tilda = torch.einsum("bkj,bj->bk", C_t, q_t) / divisor.view(-1,1)
        
        # 主输出门 + 残差
        h_main = o_t * h_tilda
        
        # 残差门控机制
        combined = torch.cat([h_main, h_prev], dim=1)
        res_weight = torch.sigmoid(self.res_gate(combined))  # 自适应残差权重
        h_skip = res_weight * h_main + (1-res_weight) * h_prev  # 门控残差连接

        # ====== 阶段1和2更新队列，阶段3不更新 ======
        if stage in [1, 2]:
            detach_factor = 0.3  # 可学习参数
            h_queue.append(detach_factor * h_skip.detach() + (1-detach_factor)*h_skip)
            if len(h_queue) > 17:  # 控制队列长度
                h_queue.pop(0)

        # 更新状态（h_prev存储残差输出）
        new_states = (C_t, n_t, m_t, h_skip, h_queue)

        return h_skip, new_states

    def init_hidden(self, batch_size, **kwargs):
        base_state = [
            torch.zeros(batch_size, self.hidden_size, self.hidden_size, **kwargs),
            torch.zeros(batch_size, self.hidden_size, **kwargs),
            torch.zeros(batch_size, self.hidden_size, **kwargs),
            torch.zeros(batch_size, self.hidden_size, **kwargs),  # h_prev初始为0
            deque([torch.zeros(batch_size, self.hidden_size, **kwargs) 
                  for _ in range(3*self.time_window)], maxlen=17)
        ]
        return tuple(base_state)

class FANLayer(nn.Module):
    """
    FANLayer: The layer used in FAN (https://arxiv.org/abs/2410.02675).

    Args:
        input_dim (int): The number of input features.
        output_dim (int): The number of output features.
        p_ratio (float): The ratio of output dimensions used for cosine and sine parts (default: 0.25).
        activation (str or callable): The activation function to apply to the g component. If a string is passed,
            the corresponding activation from torch.nn.functional is used (default: 'gelu').
        use_p_bias (bool): If True, include bias in the linear transformations of p component (default: True). 
            There is almost no difference between bias and non-bias in our experiments.
    """

    def __init__(self, input_dim, output_dim, p_ratio=0.25, activation='gelu', use_p_bias=True, dropout=0.1):
        super(FANLayer, self).__init__()

        # Ensure the p_ratio is within a valid range
        assert 0 < p_ratio < 0.5, "p_ratio must be between 0 and 0.5"

        self.p_ratio = p_ratio
        p_output_dim = int(output_dim * self.p_ratio)
        g_output_dim = output_dim - p_output_dim * 2  # Account for cosine and sine terms

        # Linear transformation for the p component (for cosine and sine parts)
        self.input_linear_p = nn.Linear(input_dim, p_output_dim, bias=use_p_bias)

        # Linear transformation for the g component
        self.input_linear_g = nn.Linear(input_dim, g_output_dim)

        # Set the activation function
        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation

        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        """
        Args:
            src (Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            Tensor: Output tensor of shape (batch_size, output_dim), after applying the FAN layer.
        """

        # Apply the linear transformation followed by the activation for the g component
        g = self.activation(self.input_linear_g(src))

        # Apply the linear transformation for the p component
        p = self.input_linear_p(src)

        # Concatenate cos(p), sin(p), and activated g along the last dimension
        output = torch.cat((torch.cos(p), torch.sin(p), self.dropout(g)), dim=-1)

        return output

class DP_FANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, p_ratio=0.3, 
                 num_heads=4, tensor_rank=32, dropout=0.1):
        super().__init__()
        
        # 允许输入输出维度不同
        self.needs_projection = input_dim != output_dim
        
        # 基础FAN组件
        self.base_fan = FANLayer(input_dim, output_dim, p_ratio=p_ratio)
        
        # 动态相位调制（维度适配）
        self.phase_learner = nn.Sequential(
            nn.Linear(output_dim, output_dim//2),
            nn.GELU(),
            nn.Linear(output_dim//2, 2)
        )
        
        # 交叉注意力门控（统一维度）
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True)
        
        # 张量融合组件（维度修正）
        self.tensor_fusion = nn.Parameter(
            torch.randn(tensor_rank, output_dim, output_dim))  # [R, D, D]
        self.tensor_norm = nn.LayerNorm(output_dim)
        
        # 智能残差路径
        self.res_path = nn.Sequential()
        if self.needs_projection:
            self.res_path.append(nn.Linear(input_dim, output_dim))
        self.res_path.append(nn.GELU())
        
        # 初始化参数
        nn.init.orthogonal_(self.tensor_fusion)

    def forward(self, x, cross_x=None):
        # 基础FAN变换
        base_out = self.base_fan(x)  # [B, D_out]
        
        # 动态相位调制
        phase_shift = self.phase_learner(base_out)  # 使用base_out作为输入
        phase_cos = torch.cos(phase_shift[:, 0]).unsqueeze(-1)
        phase_sin = torch.sin(phase_shift[:, 1]).unsqueeze(-1)
        modulated_out = base_out * phase_cos + base_out.roll(1, dims=-1) * phase_sin
        
        # 交叉注意力门控（维度统一）
        if cross_x is not None:
            # 确保cross_x维度匹配
            if cross_x.size(-1) != modulated_out.size(-1):
                cross_x = F.layer_norm(cross_x, (cross_x.size(-1),))
                cross_x = F.linear(cross_x, 
                                 torch.eye(modulated_out.size(-1)), 
                                 bias=None)
            attn_out, _ = self.cross_attn(
                query=modulated_out.unsqueeze(1),
                key=cross_x,
                value=cross_x)
            gated_out = modulated_out + attn_out.squeeze(1)
        else:
            gated_out = modulated_out
        
        # 张量融合（维度安全操作）
        tensor_out = torch.einsum('bd,rdh->brh', gated_out, self.tensor_fusion)
        tensor_out = torch.einsum('brh,rdh->bd', tensor_out, self.tensor_fusion)
        tensor_out = self.tensor_norm(tensor_out + gated_out)
        
        # 残差连接（智能维度投影）
        res_x = self.res_path(x)  # [B, input_dim] -> [B, output_dim]
        final_out = tensor_out + res_x
        
        return fourier_transform_constraint(final_out)

class FrequencyFeatureBank(nn.Module):
    def __init__(self, topk=10, feature_dim=64):
        super().__init__()
        self.bank = defaultdict(lambda: deque(maxlen=50))  # 存储topk视频的时序特征队列
        self.topk = topk
        self.feature_dim = feature_dim
        
    def update(self, retrieved_video_ids, pred_sequence):
        """更新逻辑：对每个topk位置的视频分别维护特征队列"""
        with torch.no_grad():
            # 傅里叶变换提取主频特征
            fft_features = torch.fft.rfft(pred_sequence, dim=1)
            amp = torch.abs(fft_features)
            
            # 对每个样本的topk视频更新特征（参考网页3[3](@ref)的频域分析）
            for k in range(self.topk):  # 遍历topk位置
                for b in range(len(retrieved_video_ids[k])):  # 遍历batch
                    vid = retrieved_video_ids[k][b]
                    # 提取主频能量占比
                    current_k = min(3, amp[b].numel())  # 确保k不超过元素总数
                    top3_amp = torch.topk(amp[b], k=current_k).values
                    energy_ratio = top3_amp / amp[b].sum()
                    self.bank[(vid, k)].append(energy_ratio.mean().item())  # 按位置存储
                    
    def retrieve(self, retrieved_video_ids):
        """检索逻辑：对每个topk位置独立检索特征"""
        batch_feats = []
        for k in range(self.topk):  # 遍历topk位置
            pos_feats = []
            for b in range(len(retrieved_video_ids[k])):  # 遍历batch
                vid = retrieved_video_ids[k][b]
                # 位置敏感检索
                if (vid, k) in self.bank and len(self.bank[(vid, k)]) > 0:
                    feat = torch.tensor(self.bank[(vid, k)]).mean().item()
                else:
                    # 空值处理：基于位置的高斯初始化（参考网页3[3](@ref)的噪声抑制）
                    feat = torch.randn(1).item() * 0.01 + k/self.topk
                pos_feats.append(feat)
            batch_feats.append(torch.tensor(pos_feats))
        return torch.stack(batch_feats).to(device)  # [topk, batch]

        
class youtube_mLSTM(nn.Module):
    def __init__(self, seq_length, batch_size):
        super(youtube_mLSTM, self).__init__()
        self.text_num = 5
        self.meta_num = 6

        self.batch_size = batch_size
        self.img_feature = nn.Sequential(*list(models.resnet101(pretrained=True).children())[:-1])


        # embedding vocabulary
        self.cate_vocab = {"People  Blogs": 1,
                           "Gaming": 2,
                           "News  Politics": 3,
                           "Entertainment": 4,
                           "Music": 5,
                           "Education": 6,
                           "Sports": 7,
                           "Howto  Style": 8,
                           "Film  Animation": 9,
                           "Nonprofits  Activism": 10,
                           "Travel": 11,
                           "Comedy": 12,
                           "Science  Technology": 13,
                           "Autos  Vehicles": 14,
                           "Pets  Animals": 15,
                           "OOA": 0,
                           }
        self.lang_vocab = {"en": 1,
                           "zh": 2,
                           "ko": 3,
                           "ja": 4,
                           "hi": 5,
                           "ru": 6,
                           "OOA": 0,
                           }

        # BERT
        self.tokenizer = BertTokenizer.from_pretrained('/opt/data/private/lstm/bert_multilingual')
        self.bert_model = BertModel.from_pretrained('/opt/data/private/lstm/bert_multilingual').to(device)

        self.conv = nn.Conv2d(self.text_num, 1, 1)
        self.conv.weight.data.normal_(1 / self.text_num, 0.01)

        # visual_MLP
        self.img_FAN = FANLayer(input_dim=2048, output_dim=128, p_ratio=0.25, activation='gelu', use_p_bias=True)

        # embeddings & MLP
        self.cate_embedding = nn.Sequential(
            nn.Embedding(16, 128),
        )
        self.lang_embedding = nn.Sequential(
            nn.Embedding(7, 128),
        )
        self.emb_FAN = FANLayer(input_dim=256, output_dim=128, p_ratio=0.4, activation='gelu', use_p_bias=True)
        
        self.lang_FAN = FANLayer(input_dim=128, output_dim=128, p_ratio=0.4, activation='gelu', use_p_bias=True)

        self.cate_FAN = FANLayer(input_dim=128, output_dim=128, p_ratio=0.4, activation='gelu', use_p_bias=True)

        # textual MLP
        self.text_FAN = FANLayer(input_dim=768, output_dim=128, p_ratio=0.4, activation='gelu', use_p_bias=True)
        
        self.meta_FAN = FANLayer(input_dim=6, output_dim=128, p_ratio=0.4, activation='gelu', use_p_bias=True)

        self.agg_meta_FAN = FANLayer(input_dim=1, output_dim=128, p_ratio=0.4, activation='gelu', use_p_bias=True)


        # all_feature_length and feature_vector_length
        all_f_len = 8*128
        self.vector_len = 128

        self.seq_length = seq_length
        self.batch_size = batch_size
        self.vector_len = 128  # mLSTM 隐藏维度
        self.input_dim = 512   # 输入维度
        self.hidden_dim = 128

        self.input_net = nn.Sequential(
            FANLayer(input_dim=256, output_dim=256, p_ratio=0.4),
            nn.LayerNorm(256),
            FANLayer(256, 128, p_ratio=0.3),
            nn.LayerNorm(128),
            FANLayer(128, self.vector_len, p_ratio=0.2)
        )

        self.feature_fusion = nn.Sequential(
            FANLayer(all_f_len, 512, p_ratio=0.3),
            nn.LayerNorm(512),
            nn.Dropout(0.05),  # 降低dropout比例
            FANLayer(512, 256, p_ratio=0.25),
            nn.LayerNorm(256)
        )
        # 使用单个mLSTMCell（时间步共享）
        self.mLSTMCell = ResidualEnhancedLSTMCell(input_size=self.vector_len, hidden_size=self.vector_len)
        self.cover_transform = transforms.Compose([transforms.Resize([224, 224]), transforms.ToTensor()])
        # 共享的输出网络
        self.output_net = nn.Sequential(
            FANLayer(self.vector_len, 64, p_ratio=0.3),
            nn.LayerNorm(64),
            nn.Dropout(0.1),
            FANLayer(64, 32, p_ratio=0.2),
            nn.Linear(32, 1)
        )

        # 改进的隐藏状态初始化
        self.h_init = nn.Sequential(
            FANLayer(256, 256, p_ratio=0.3),
            nn.LayerNorm(256),
            FANLayer(256, 128, p_ratio=0.2)
        )

        # 新增频域特征处理模块
        self.freq_bank = FrequencyFeatureBank()
        self.freq_proj = nn.Sequential(
            FANLayer(1, 128, p_ratio=0.3),  
            nn.LayerNorm(128)
        )


    def retrieval_aggregation(self, retrieved_video_ids, retrieved_similarities,
                            retrieved_textual_features, retrieved_ep, current_day1):
       """
       修改后的检索特征聚合方法（全PyTorch实现）
       Args:
           retrieved_video_ids: list[list[str]]  # [batch_size][k]
           retrieved_similarities: list[list[float]]  # [batch_size][k]
           retrieved_textual_features: list[list[str]]  # [batch_size][k]
           retrieved_ep: list[list[float]]  # [batch_size][k]
       Returns:
           agg_visual: torch.Tensor [batch_size, 2048]
           agg_text: torch.Tensor [batch_size, 768]
           agg_meta: torch.Tensor [batch_size, 1]
       """

       batch_size = len(retrieved_video_ids)
       device = next(self.img_FAN.parameters()).device
    
       # 检查是否有填充的样本（所有视频ID都是'0'）
       is_padded = torch.zeros(batch_size, dtype=torch.bool, device=device)
       for i in range(batch_size):
           if all(vid == '0' for vid in retrieved_video_ids[i]):
               is_padded[i] = True
    
       # 如果有填充样本，创建零向量
       if is_padded.any():
           zero_vector = torch.zeros(batch_size, 128, device=device)
        
           # 如果全部都是填充样本，直接返回零向量
           if is_padded.all():
               return zero_vector, zero_vector, zero_vector, zero_vector
       # 1. 相似度权重计算（使用PyTorch实现）
       similarities = torch.tensor(retrieved_similarities, dtype=torch.float32, device=device)
       weights = torch.softmax(similarities / 0.3, dim=1)  # [B, k][1,5](@ref)

       # 2. 视觉特征聚合（与文本处理对齐）
       visual_features = []
       # 外层循环遍历每个索引位置（共64个）
       for i in range(len(retrieved_video_ids[0])):  # 假设每个子列表长度相同
           batch_visual = []
           # 内层循环遍历每个子列表（共5个）
           for j in range(len(retrieved_video_ids)):  # 遍历5个子列表
               if is_padded[j]:
                   batch_visual.append(torch.zeros(2048, device=device))
                   continue
               vid = retrieved_video_ids[j][i]  # 取第j个子列表的第i个视频ID
               try:
                   img_path = f"/opt/private/lstm/data_source/img_yt/{vid}.jpg"
                   img = Image.open(img_path).convert("RGB")
                   img_tensor = self.cover_transform(img).unsqueeze(0).to(device)  # [1, C, H, W]
                   with torch.no_grad():
                        feat = self.img_feature(img_tensor).squeeze()  # 输出形状应为[2048]
                   #print('feat.size:',feat.size())
                   batch_visual.append(feat)
               except Exception as e:
                   #print(f"Error loading {vid}: {str(e)}")
                   batch_visual.append(torch.zeros(2048, device=device))
    
           # 将5个视频特征堆叠为[5, 2048]
           visual_features.append(torch.stack(batch_visual, dim=0)) 

       # 最终形状为[64, 5, 2048]（与文本的[64,5,768]结构对齐）
       visual_features = torch.stack(visual_features, dim=0)  
       original_shape = visual_features.shape  
       visual_features_flat = visual_features.reshape(-1, original_shape[-1])
       #print('visual_features_flat.size:', visual_features_flat.size())  # 应输出torch.Size([64,5,2048])


       with torch.no_grad():
          agg_visual_flat = self.img_FAN(visual_features_flat)  # [64,5,128]
           
       
       agg_visual = agg_visual_flat.reshape(original_shape[0], original_shape[1], -1)
       #print('agg_visual.size before:',agg_visual.size())

       # 加权聚合（保持张量在GPU）
       agg_visual = torch.einsum('bk,bkd->bd', weights, agg_visual)  # [B, 128][3](@ref)
       #print('agg_visual.size after:',agg_visual.size())

       # 3. 文本特征聚合
       text_features_list = []
       for i in range(len(retrieved_textual_features[0])):
           batch = []
           for j in range(len(retrieved_textual_features)):
               # 如果是填充样本，使用空字符串
               if is_padded[j]:
                   batch.append("")
               else:
                   batch.append(retrieved_textual_features[j][i])
           with torch.no_grad():
              bert_features = bert_feature(batch, self.bert_model, self.tokenizer)  # 输入形状为 (5, 768)
           text_features_list.append(bert_features)
       text_features = torch.stack(text_features_list, dim=0)           
       #print('text_features.size:', text_features.size())  # 输出应为 torch.Size([64, 5, 768])
       
       text_original_shape = text_features.shape
       text_features_flat = text_features.reshape(-1, text_original_shape[-1])
       with torch.no_grad():
           agg_text_flat = self.text_FAN(text_features_flat)  # [320,128]
       agg_text = agg_text_flat.reshape(text_original_shape[0], text_original_shape[1], -1)

           
       #print('agg_text.size:',agg_text.size())
       
       # 加权聚合（保持张量在GPU）
       agg_text = torch.einsum('bk,bkd->bd', weights, agg_text)  # [B, 128][3](@ref)
       #print('agg_text.size:',agg_text.size())

       # 4. 元数据聚合
       ep_tensor = torch.tensor(retrieved_ep, dtype=torch.float32, device=device).unsqueeze(1)
       agg_meta = torch.einsum('bk,bk->b', weights, ep_tensor).unsqueeze(1)  # [B, 1]
       with torch.no_grad():
           agg_meta = self.agg_meta_FAN(agg_meta)
       #print('agg_meta.size:',agg_meta.size())

       # 5. 新增频域特征聚合
       # 对于填充样本，使用零向量
       if is_padded.any():
           # 只处理非填充样本
           non_padded_indices = torch.where(~is_padded)[0].tolist()
           non_padded_vids = [retrieved_video_ids[i] for i in non_padded_indices]
           freq_feats = self.freq_bank.retrieve(non_padded_vids)  # [topk, non_padded_batch]

           # 位置加权聚合
           position_weights = torch.linspace(1, 0.5, 10).to(device)  # top1权重最高
           agg_freq_non_padded = torch.einsum('k,kb->b', position_weights, freq_feats)
           agg_freq_non_padded = self.freq_proj(agg_freq_non_padded.unsqueeze(1))
        
           # 创建全零向量
           agg_freq = torch.zeros(batch_size, 128, device=device)
           agg_freq[non_padded_indices] = agg_freq_non_padded
       else:
           freq_feats = self.freq_bank.retrieve(retrieved_video_ids)  # [topk, batch]
           position_weights = torch.linspace(1, 0.5, 10).to(device)  # top1权重最高
           agg_freq = torch.einsum('k,kb->b', position_weights, freq_feats)
           agg_freq = self.freq_proj(agg_freq.unsqueeze(1))

       return agg_visual, agg_text, agg_meta, agg_freq

    def forward(self, img, texts, meta, cat, retrieved_video_ids, retrieved_similarities, 
                             retrieved_textual_features, retrieved_ep):
        # '''======================= visual features =========================='''
        with torch.no_grad():
            img_features = self.img_feature(img).squeeze()
        img_features = self.img_FAN(img_features)


        # '''======================= embedding features =========================='''

        cate_voc = []
        lang_voc = []
        for c in cat[0]:
            if c in self.cate_vocab.keys():
                cate_voc.append(self.cate_vocab[c])
            else:
                cate_voc.append(self.cate_vocab["OOA"])

        for c in cat[1]:
            lang = langid.classify(c)[0]
            if lang in self.lang_vocab.keys():
                lang_voc.append(self.lang_vocab[lang])
            else:
                lang_voc.append(self.lang_vocab["OOA"])

        cate_voc = torch.LongTensor(cate_voc).to(device)
        lang_voc = torch.LongTensor(lang_voc).to(device)
        cate_feature = self.cate_embedding(cate_voc)
        lang_feature = self.lang_embedding(lang_voc)
        cate_feature = self.cate_FAN(cate_feature)
        lang_feature = self.lang_FAN(lang_feature)

        #embedding_features = cate_feature * lang_feature
        embedding_features = torch.cat([cate_feature,lang_feature],dim=1)


        embedding_features = self.emb_FAN(embedding_features)
        #embedding_features = self.proj_embedding(embedding_features)

        # '''======================= textual features =========================='''

        text_features_list = []
        for text in texts:
            with torch.no_grad():
                # bert feature
                bert_features = bert_feature(text, self.bert_model, self.tokenizer)
            text_features = bert_features

            text_features_list.append(text_features)
        text_features = torch.stack(text_features_list, 1).unsqueeze(3)
        text_features = self.conv(text_features).permute(0, 2, 1, 3).squeeze()
        text_features = self.text_FAN(text_features)


        # '''======================= numerical features =========================='''
        meta_features = self.meta_FAN(meta)

        # '''======================= agg features =========================='''
        current_day1 = meta[:, 5]
        agg_visual, agg_text, agg_meta, agg_freq  = self.retrieval_aggregation(retrieved_video_ids, retrieved_similarities, retrieved_textual_features, retrieved_ep, current_day1)


        # '''======================= feature fusion=========================='''

        # 将所有特征在时间维度上拼接
        feature_vector = torch.cat([
            img_features, 
            embedding_features,
            text_features, 
            meta_features,
            agg_visual, 
            agg_text, 
            agg_meta,
            agg_freq
        ], dim=1)

        # 改进的特征融合
        fused_features = self.feature_fusion(feature_vector)

        # 初始化 mLSTM 的内部状态 C, n, m
        init_state = self.h_init(fused_features)
        C = torch.einsum('bi,bj->bij', init_state, init_state) 
        C += 1e-4 * torch.eye(self.vector_len, device=C.device).unsqueeze(0)
        n = init_state
        m = init_state

        # 使用deque维护时间窗口队列
        h_queue = deque(maxlen=17)
        h_prev = torch.zeros_like(init_state)  # 初始化为零，将在第一个时间步被覆盖

        # 预测值和隐藏状态的输出存储
        outputs = []
        for t in range(self.seq_length):
            # 共享的输入处理
            x_t  = self.input_net(fused_features)
            # 动态初始化策略
            if t == 0:  # 第一个时间步特殊处理
                # 用x_t生成初始隐藏状态
                h_prev = x_t.detach()  # 初始h_prev来自输入
                # 预填充队列
                for _ in range(3*self.mLSTMCell.time_window):
                    h_queue.append(x_t.detach().clone())
            
            # 执行LSTM计算（修改状态结构）
            states = (C, n, m, h_prev, h_queue)
            h_t, new_states = self.mLSTMCell(x_t, states, t)
            
            # 更新状态组件（带梯度控制）
            C, n, m, h_prev, h_queue = new_states
            C = C.detach()  # 阻断协方差矩阵的梯度
            n = n.detach()  # 阻断归一化状态梯度
            m = m.detach()  # 阻断稳定化状态梯度
            
            # 共享的输出网络
            outputs.append(self.output_net(h_t))
         
        outputs = torch.stack(outputs, dim=1).squeeze(-1)

        # 动态更新频域特征库
        self.freq_bank.update(retrieved_video_ids, outputs.detach())

        return outputs
