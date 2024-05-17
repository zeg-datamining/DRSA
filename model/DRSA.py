import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from layers.Embed import DataEmbedding
from layers.Conv_Blocks import Inception_Block_V1
from layers.Dynamic_Routing_Filter import Dynamic_Routing_Filter
from timm.models.layers import trunc_normal_

def FFT_for_Period(x, k=2):  # Preserve the periodicity of time series(TimesNET)
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list

    return period, abs(xf).mean(-1)[:, top_list]

class DRF(nn.Module):
    def __init__(self, configs, dim, att_topk, n_win, num_heads=8, qk_dim=None, kv_per_win=4,
                side_dwconv=5, before_attn_dwconv=3, auto_pad=True):
        super().__init__()
        qk_dim = qk_dim or dim
        self.att_topk = configs.att_topk
        self.n_win = configs.n_win
        self.drop_path = nn.Identity()
        self.pos_embed = nn.Conv2d(dim, dim, kernel_size=before_attn_dwconv, padding=1, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Dynamic_Routing_Filter(dim=dim, num_heads=num_heads, n_win=n_win, qk_dim=qk_dim,
                                            kv_per_win=kv_per_win,
                                            att_topk=att_topk,
                                            side_dwconv=side_dwconv,
                                            )
        self.mlp = nn.Sequential(nn.Linear(dim, int(4 * dim)),
                                 nn.Identity(),
                                 nn.GELU(),
                                 nn.Linear(int(4 * dim), dim)
                                 )

    def forward(self, x):
        x = x + self.pos_embed(x)
        x = x.permute(0, 2, 3, 1)
        x = x + self.drop_path(self.attn(self.norm(x)))  # (B, H, W, C)
        x = x + self.drop_path(self.mlp(self.norm(x)))  # (B, H, W, C)
        x = x.permute(0, 3, 1, 2)
        return x


class DRSA(nn.Module):
    def __init__(self, configs, num_classes=1000, qk_dims=64,
                 kv_per_wins=4, side_dwconv=5,
                 before_attn_dwconv=3, auto_pad=True,
                 ):
        super(DRSA, self).__init__()
        self.d_model = configs.d_model
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.k = configs.top_k
        self.att_topk = configs.att_topk
        self.n_win = configs.n_win
        self.head = nn.Linear(self.d_model, num_classes) if num_classes > 0 else nn.Identity()

        nheads = 8
        self.conv = nn.Sequential(
            Inception_Block_V1(configs.d_model, configs.d_ff,
                               num_kernels=configs.num_kernels),
            nn.GELU(),
            Inception_Block_V1(configs.d_ff, configs.d_model,
                               num_kernels=configs.num_kernels)
        )

        self.feature_ext = nn.Sequential(
            *[DRF(configs=configs, dim=self.d_model, num_heads=nheads, n_win=self.n_win,
                  qk_dim=qk_dims,
                  kv_per_win=kv_per_wins,
                  att_topk=self.att_topk,
                  side_dwconv=side_dwconv,
                  before_attn_dwconv=before_attn_dwconv,
                  auto_pad=auto_pad)],
        )

        self.norm = nn.BatchNorm2d(self.d_model)
        self.pre_logits = nn.Identity()
        self.head = nn.Linear(self.d_model, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def feature_att(self, x):
        x = self.feature_ext(x)
        x = self.norm(x)
        x = self.pre_logits(x)

        return x

    def forward(self, x):
        B, T, N = x.size()
        period_list, period_weight = FFT_for_Period(x, self.k)
        res = []
        for i in range(self.k):
            period = period_list[i]
            # period = period_list
            # padding
            if (self.seq_len + self.pred_len) % period != 0:
                length = (
                                 ((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = (self.seq_len + self.pred_len)
                out = x
            # 1D Time Series->2D Time Series
            out = out.reshape(B, length // period, period,
                              N).permute(0, 3, 1, 2).contiguous()  # B,N,H,W
            out = self.feature_att(out)

            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            if(out.shape[1] < (self.seq_len + self.pred_len)):
                length1 = self.seq_len + self.pred_len
                pad = torch.zeros([out.shape[0], length1, out.shape[2]]).to(x.device)
                out = torch.cat([out, pad], dim=1)
            res.append(out[:, :(self.seq_len + self.pred_len), :])
        res = torch.stack(res, dim=-1)
        # adaptive aggregation
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(
            1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        res = res + x
        return res


class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.model = nn.ModuleList([DRSA(configs)
                                    for _ in range(configs.e_layers)])
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_ff, configs.embed, configs.freq,
                                           configs.dropout)
        self.layer = configs.e_layers
        self.layer_norm = nn.LayerNorm(configs.d_model)

        self.predict_linear = nn.Linear(self.seq_len, self.pred_len + self.seq_len)
        self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out = self.predict_linear(enc_out.permute(0, 2, 1)).permute(0, 2, 1)
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        # porject back
        dec_out = self.projection(enc_out)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1))

        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]