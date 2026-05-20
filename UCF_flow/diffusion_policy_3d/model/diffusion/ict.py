import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ==========================================
# 1. 基础组件 (PosEmb & Timestep & Final)
# ==========================================
# 添加自定义的 RMSNorm 以兼容低版本 PyTorch
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, theta=10000.0):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_dim=256):
        super().__init__()
        self.freq_embed = SinusoidalPosEmb(frequency_embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size)
        )

    def forward(self, t):
        return self.mlp(self.freq_embed(t))
    
class FinalLayer(nn.Module):
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.norm = RMSNorm(hidden_size, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_size)

    def forward(self, x):
        x = self.norm(x)
        return self.linear(x)

# ==========================================
# 2. 增强型前缀编码器 (支持多 Token 调整)
# ==========================================
class PrefixConditionEncoder(nn.Module):
    def __init__(self, pc_dim, state_dim, n_emb, 
                 cond_seq_len=2, 
                 num_t_tokens=1, 
                 num_h_tokens=1):
        super().__init__()
        self.n_emb = n_emb
        self.cond_seq_len = cond_seq_len
        self.num_t_tokens = num_t_tokens
        self.num_h_tokens = num_h_tokens
        
        # 1. 观测特征处理 (2帧 -> 2个 Token)
        self.obs_proj = nn.Sequential(
            nn.Linear(pc_dim + state_dim, n_emb),
            nn.GELU(),
            nn.Linear(n_emb, n_emb)
        )
        self.obs_pos_emb = nn.Parameter(torch.randn(1, cond_seq_len, n_emb) * 0.02)
        
        # 2. 时间特征提取器
        self.time_emb_net = TimestepEmbedder(hidden_size=n_emb)
        self.h_emb_net = TimestepEmbedder(hidden_size=n_emb)
        
        # 3. 多 Token 身份偏置 (形状为 [Num_Tokens, n_emb])
        self.t_token_bias = nn.Parameter(torch.randn(num_t_tokens, n_emb) * 0.02)
        self.h_token_bias = nn.Parameter(torch.randn(num_h_tokens, n_emb) * 0.02)

    def forward(self, cond, timestep, r):
        B = cond['pc_feat'].shape[0]
        
        # --- A. 处理观测 Token (B, 2, n_emb) ---
        obs_concat = torch.cat([cond['pc_feat'], cond['state_feat']], dim=-1)
        obs_tokens = self.obs_proj(obs_concat) + self.obs_pos_emb
        
        # --- B. 处理时间 Token (B, N_t/N_h, n_emb) ---
        timestep = timestep.expand(B) if not torch.is_tensor(timestep) else timestep
        r = r.expand(B) if not torch.is_tensor(r) else r
        
        # 1. 计算基础 embedding (B, n_emb)
        t_base = self.time_emb_net(timestep)
        h_base = self.h_emb_net(timestep - r)
        
        # 2. 利用广播机制：(B, 1, n_emb) + (N_tokens, n_emb) -> (B, N_tokens, n_emb)
        t_tokens = t_base.unsqueeze(1) + self.t_token_bias
        h_tokens = h_base.unsqueeze(1) + self.h_token_bias
        
        # --- C. 拼接 ---
        prefix_sequence = torch.cat([t_tokens, h_tokens, obs_tokens], dim=1)
        return prefix_sequence

# ==========================================
# 3. 纯粹的自注意力块 (Pure Transformer Block)
# ==========================================
class PureTransformerBlock(nn.Module):
    """剥离了 AdaLN 和 Cross-Attention 的极简 Block"""
    def __init__(self, n_emb, n_head, p_drop_attn=0.0, p_drop=0.0):
        super().__init__()
        self.n_emb = n_emb
        self.n_head = n_head
        self.head_dim = n_emb // n_head
        
        # 1. Self-Attention 
        self.norm1 = nn.LayerNorm(n_emb, eps=1e-6)
        self.qkv = nn.Linear(n_emb, 3 * n_emb, bias=True)
        self.proj_msa = nn.Linear(n_emb, n_emb, bias=True)
        self.dropout_attn = p_drop_attn
        
        # 2. Feedforward Network (FFN)
        self.norm2 = nn.LayerNorm(n_emb, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(n_emb, 4 * n_emb),
            nn.GELU(),
            nn.Linear(4 * n_emb, n_emb),
            nn.Dropout(p_drop),
        )

    def forward(self, x, attn_mask=None): 
        B, T, D = x.shape
        
        residual = x
        x_norm = self.norm1(x)
        
        qkv = self.qkv(x_norm).reshape(B, T, 3, self.n_head, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        
        # 将 Mask 传入高效注意力函数
        attn_output = F.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=attn_mask,   # <--- 核心修改点
            dropout_p=self.dropout_attn if self.training else 0.0
        )
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(B, T, D)
        
        x = residual + self.proj_msa(attn_output)
        
        residual = x
        ffn_output = self.mlp(self.norm2(x))
        x = residual + ffn_output
        
        return x
# ==========================================
# 4. 主干模型 (Prefix DiT Model)
# ==========================================
class DiTModel(nn.Module):
    def __init__(self,
            input_dim: int,
            horizon: int,
            pc_dim: int,
            state_dim: int,
            n_layer: int = 8,
            n_emb: int = 256,
            num_t_tokens: int = 1,  # 可调参数
            num_h_tokens: int = 1,  # 可调参数
            cond_seq_len: int = 2,
            **kwargs 
        ):
        super().__init__()
        self.n_emb = n_emb
        self.horizon = horizon
        
        # 动态计算前缀总长度
        self.prefix_len = num_t_tokens + num_h_tokens + cond_seq_len
        self.cond_seq_len = cond_seq_len
        self.input_emb = nn.Linear(input_dim, n_emb)
        self.pos_emb = nn.Parameter(torch.zeros(1, horizon, n_emb))

        self.cond_encoder = PrefixConditionEncoder(
            pc_dim=pc_dim, 
            state_dim=state_dim, 
            n_emb=n_emb, 
            cond_seq_len=cond_seq_len,
            num_t_tokens=num_t_tokens,
            num_h_tokens=num_h_tokens
        )

        self.blocks = nn.ModuleList([
            PureTransformerBlock(n_emb, kwargs.get('n_head', 8)) for _ in range(n_layer)
        ])
        self.final_layer = FinalLayer(n_emb, input_dim)
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        self.apply(_basic_init)

        # 1. 基础 Embedding 初始化
        nn.init.normal_(self.input_emb.weight, std=0.02)
        if self.input_emb.bias is not None:
            nn.init.zeros_(self.input_emb.bias)
        nn.init.normal_(self.pos_emb, std=0.02)

        # 2. 预测输出层绝对零度初始化
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

        # 3. 关键：网络主干残差分支的零初始化 (Zero-Init for Identity Mapping)
        def _zero_out_residual_projs(module):
            if isinstance(module, PureTransformerBlock):
                # 自注意力的输出投影归零
                nn.init.constant_(module.proj_msa.weight, 0)
                nn.init.constant_(module.proj_msa.bias, 0)
                
                # MLP 最后一层 (索引为 2 的 Linear) 归零
                nn.init.constant_(module.mlp[2].weight, 0)
                nn.init.constant_(module.mlp[2].bias, 0)
                
        self.apply(_zero_out_residual_projs)

    # # 专门用于生成这个自定义逻辑的 Mask
    # def _create_custom_mask(self, seq_len, device):
    #     # 初始化全 0 (False) 矩阵
    #     mask = torch.zeros((seq_len, seq_len), dtype=torch.bool, device=device)
        
    #     # 索引计算
    #     idx_time_end = 2                # t, h 的结束位置
    #     idx_cond_end = 2 + self.cond_seq_len  # cond 的结束位置
        
    #     # 规则 1: Time tokens (0:idx_time_end) 保持纯净，内部也不互相影响
    #     # 只能看到自己（对角线填充 True）
    #     idx_time = torch.arange(0, idx_time_end, device=device)
    #     mask[idx_time, idx_time] = True
        
    #     # 规则 2: Cond tokens (idx_time_end:idx_cond_end) 保持纯净，内部也不互相影响
    #     # 同样只能看到自己（对角线填充 True）
    #     idx_cond = torch.arange(idx_time_end, idx_cond_end, device=device)
    #     mask[idx_cond, idx_cond] = True
        
    #     # 规则 3: Action tokens (idx_cond_end:) 看全部
    #     mask[idx_cond_end:, :] = True
        
    #     # 为了适配多头注意力，增加维度 (1, 1, T, T)
    #     # 这样 PyTorch 可以自动广播到 (Batch, Head, T, T)
    #     return mask.unsqueeze(0).unsqueeze(0)
    
    # 专门用于生成这个自定义逻辑的 Mask
    # def _create_custom_mask(self, seq_len, device):
    #     # 初始化全 0 (False) 矩阵
    #     mask = torch.zeros((seq_len, seq_len), dtype=torch.bool, device=device)
        
    #     # 索引计算
    #     idx_time_end = 2                    # t 和 r 的结束位置 (假设前两个 token 是 t 和 r)
    #     idx_cond_end = 2 + self.cond_seq_len  # cond 的结束位置
        
    #     # ==================== 修改后的规则 ====================
        
    #     # 规则 1: t 和 r 可以互相 attend（双向）
    #     idx_tr = torch.arange(0, idx_time_end, device=device)  # [0, 1]
    #     mask[idx_tr[:, None], idx_tr] = True   # t 和 r 区域全 True（互相可见）
        
    #     # 规则 2: Cond tokens 内部可以互相 attend（全可见）
    #     idx_cond = torch.arange(idx_time_end, idx_cond_end, device=device)
    #     # cond 区域内部全允许 attend
    #     mask[idx_cond[:, None], idx_cond] = True
        
    #     # 规则 3: Action tokens (idx_cond_end:) 看全部（不变）
    #     mask[idx_cond_end:, :] = True
        
    #     # 可选：如果希望 t/r 也能看到 cond（根据你的实际需求决定）
    #     # mask[idx_tr[:, None], idx_cond] = True
        
    #     # 可选：如果希望 cond 也能看到 t/r
    #     # mask[idx_cond[:, None], idx_tr] = True
        
    #     # 为了适配多头注意力，增加维度 (1, 1, T, T)
    #     return mask.unsqueeze(0).unsqueeze(0)
    
    def forward(self, sample, timestep, r=None, cond=None, global_cond=None, **kwargs):
        # 兼容外部接口
        if cond is None and global_cond is not None:
            cond = global_cond
        if r is None:
            r = torch.zeros_like(timestep)

        # 1. 获取特征并拼接
        action_tokens = self.input_emb(sample) + self.pos_emb
        prefix_tokens = self.cond_encoder(cond, timestep, r)
        seq = torch.cat([prefix_tokens, action_tokens], dim=1)

        # 2. 动态生成 Mask
        # 因为序列长度固定为 8 (2+2+4)，所以每次前向传播生成开销极小
        # attn_mask = self._create_custom_mask(seq.shape[1], device=seq.device)

        # 3. 带着 Mask 进 Block 前向传播
        for block in self.blocks:
            seq = block(seq, attn_mask=None)  # 注意：这里传入 attn_mask，表示使用自定义的 Mask

        # 4. 截断保留 Action 部分并输出
        action_output = seq[:, self.prefix_len:, :]  
        x = self.final_layer(action_output) 
        return x