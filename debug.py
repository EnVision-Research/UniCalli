import torch
from einops import rearrange

h, w, ph, pw = 4, 3, 2, 2               # 小例子
H, W = h*ph, w*pw

# 构造按行优先的 patch 序列编号：i = 0..(h*w-1)
idx = torch.arange(h*w)
x = idx.view(1, h*w, 1).float()         # 形状 (b=1, h*w, c*ph*pw=1) 只为演示

img = rearrange(x, "b (hh ww) c -> b c hh ww", hh=h, ww=w)  # 还原到 (h,w) 网格
# 如果再做你那步 unpatchify（带 ph/pw），把 c=ph*pw 的展开加上即可

print(img[0,0])  # 若每行是连续递增块，说明使用的是行优先
# 你也可以把真实 cond_pred 走一遍，观察每行/每列编号或图像是否按期望对齐