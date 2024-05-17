import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class Dynamic_Routing_Filter(nn.Module):
    def __init__(self, dim, num_heads=8, n_win=8, qk_dim=None,
                 kv_per_win=4, att_topk=4,  side_dwconv=3, pad=True):
        super().__init__()
        self.pad = pad
        self.n_win = n_win
        self.dim = dim
        self.qk_dim = qk_dim
        self.num_heads = num_heads
        assert self.qk_dim % num_heads == 0 and self.dim % num_heads == 0, 'qk_dim and dim must be divisible by num_heads!'
        self.qkv = nn.Linear(dim, qk_dim + qk_dim + dim, bias=True)
        self.wo = nn.Linear(dim, dim)
        self.emb = nn.Identity()
        self.att_topk = att_topk
        self.scale = self.qk_dim ** -0.5
        self.kv_per_win = kv_per_win
        self.kv_down = nn.AdaptiveAvgPool2d(self.kv_per_win)
        self.num_heads = num_heads
        self.Global = nn.Conv2d(dim, dim, kernel_size=side_dwconv, stride=1, padding=side_dwconv // 2,
                              groups=dim) if side_dwconv > 0 else \
            lambda x: torch.zeros_like(x)



    def forward(self, x):
        if self.pad:
            N, H_in, W_in, C = x.size()
            # pad_l = pad_t = 0
            pad_t = 0
            pad_b = (self.n_win - H_in % self.n_win) % self.n_win
            x = F.pad(x, (0, 0,  # dim=-1
                          0, 0,  # dim=-2
                          pad_t, pad_b))  # dim=-3
            _, H, W, _ = x.size()  # padded size
        else:
            N, H, W, C = x.size()

        assert H % self.n_win == 0

        # Blocking
        x = rearrange(x, "n (j h) w c -> n j h w c", j=self.n_win)

        q, kv = self.qkv(x).split([self.qk_dim, self.qk_dim+self.dim], dim=-1)
        # Dynamic Routing
        q_win, k_win = q.mean([2, 3]), kv[..., 0:self.qk_dim].mean([2, 3])


        #S_L
        query, key = q_win.detach(), q_win.detach()
        query_hat, key_hat = self.emb(query), self.emb(key)
        attn_logit = (query_hat * self.scale) @ key_hat.transpose(-2, -1)
        topk_attn_logit, topk_index = torch.topk(attn_logit, k=self.att_topk, dim=-1)
        # R_L
        q_pix = rearrange(q, 'n p h w c -> n p (h w) c')
        kv_pix = self.kv_down(rearrange(kv, 'n p h w c -> (n p) c h w'))
        kv_pix = rearrange(kv_pix, '(n j) c h w -> n j (h w) c', j=self.n_win)

        # K_Filter, V_Filter
        n, p, w2, c_kv = kv_pix.size()
        att_topk = topk_index.size(-1)
        filter_kv = torch.gather(kv_pix.view(n, 1, p, w2, c_kv).expand(-1, p, -1, -1, -1),
                                  dim=2,
                                  index=topk_index.view(n, p, att_topk, 1, 1).expand(-1, -1, -1, w2, c_kv))
        k_filter, v_filter = filter_kv.split([self.qk_dim, self.dim], dim=-1)

        # Routing Attention
        k_filter = rearrange(k_filter, 'n p k w2 (m c) -> (n p) m c (k w2)',
                              m=self.num_heads)
        v_filter = rearrange(v_filter, 'n p k w2 (m c) -> (n p) m (k w2) c',
                              m=self.num_heads)
        q_pix = rearrange(q_pix, 'n p w2 (m c) -> (n p) m w2 c',
                          m=self.num_heads)

        Global = self.Global(rearrange(kv[..., self.qk_dim:], 'n j h w c -> n c (j h) w', j=self.n_win).contiguous())
        Global = rearrange(Global, 'n c (j h) w -> n (j h) w c', j=self.n_win)

        attn_weight = (q_pix * self.scale) @ k_filter

        out = attn_weight @ v_filter
        out = rearrange(out, '(n j) m (h w) c -> n (j h) w (m c)', j=self.n_win,
                        h=H // self.n_win, w=W)
        out = out + Global
        # output linear
        out = self.wo(out)
        out = out[:, :H_in, :W_in, :].contiguous()
        return out