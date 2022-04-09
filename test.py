import torchvision.models as models
from torchsummary import summary
import torch.nn as nn
import torch
import torch.nn.functional as F

dummy_in = torch.randn((1,3,32 * 8,32 * 8))

proj = nn.Conv2d(
            3,
            768,
            kernel_size=32,
            stride=32,
            bias=False
        )

tokens = proj(dummy_in).view(-1,768,8*8)
print("tokens shape ",tokens.shape)
print("after projection shape ",tokens.shape)
per = 0.8
random_indx = torch.randperm(64)
print("random indx shape ", random_indx.shape)
# print(random_indx[:10])

nonvisible_idx = random_indx[:int(64 * per)]
print("mask idx ",nonvisible_idx.shape)
labels = tokens[:, :, nonvisible_idx]

tokens[:, :, nonvisible_idx] = torch.zeros(1,768,51)

print(tokens.shape)
# print(dummy_in.shape)
x= tokens.view(-1,768,8,8)

#
# #----
x_mask = (dummy_in.sum(dim=1) != 0).float()[:, None, :, :]
# print("ss ",x_mask)
x_mask = F.interpolate(x_mask, size=(x.shape[2], x.shape[3])).long()

print("x_mask shape ",x_mask.shape)
print("x_mas k ", x_mask[:,:,:,:])
#
# #-----
x_h = x_mask[:, 0].sum(dim=1)[:, 0]
x_w = x_mask[:, 0].sum(dim=2)[:, 0]
print("dims ",(x_w,x_h))
pos_embed = torch.zeros(1, 64 +1, 768)
B, C, H, W = x.shape
spatial_pos = (
    pos_embed[:, 1:, :]
        .transpose(1, 2)
        .view(1, 768, 8, 8)
)
print(spatial_pos.shape)
pos_embed = torch.cat(
    [
        F.pad(
            F.interpolate(
                spatial_pos, size=(h, w), mode="bilinear", align_corners=True,
            ),
            (0, W - w, 0, H - h),
        )
        for h, w in zip(x_h, x_w)
    ],
    dim=0,
)

print(pos_embed.shape)
print(pos_embed[:,0,:,:])



pos_embed = pos_embed.flatten(2).transpose(1, 2)
x = x.flatten(2).transpose(1, 2)
patch_index = (
    torch.stack(
        torch.meshgrid(
            torch.arange(x_mask.shape[-2]), torch.arange(x_mask.shape[-1])
        ),
        dim=-1,
    )[None, None, :, :, :]
    .expand(x_mask.shape[0], x_mask.shape[1], -1, -1, -1)
    .flatten(1, 3)
)
x_mask = x_mask.flatten(1)
max_image_len=200

if (
        max_image_len < 0
        or max_image_len is None
        or not isinstance(max_image_len, int)
):
    # suppose aug is 800 x 1333, then, maximum effective res is 800 x 1333 (if one side gets bigger, the other will be constrained and be shrinked)
    # (800 // self.patch_size) * (1333 // self.patch_size) is the maximum number of patches that single image can get.
    # if self.patch_size = 32, 25 * 41 = 1025
    # if res is 384 x 640, 12 * 20 = 240
    eff = x_h * x_w
    max_image_len = eff.max()
else:
    eff = x_h * x_w
    max_image_len = min(eff.max(), max_image_len)






valid_idx = x_mask.nonzero(as_tuple=False)
non_valid_idx = (1 - x_mask).nonzero(as_tuple=False)
unique_rows = valid_idx[:, 0].unique()
valid_row_idx = [valid_idx[valid_idx[:, 0] == u] for u in unique_rows]
non_valid_row_idx = [
    non_valid_idx[non_valid_idx[:, 0] == u] for u in unique_rows
]
valid_nums = [v.size(0) for v in valid_row_idx]
non_valid_nums = [v.size(0) for v in non_valid_row_idx]
pad_nums = [max_image_len - v for v in valid_nums]

select = list()
for i, (v, nv, p) in enumerate(zip(valid_nums, non_valid_nums, pad_nums)):
    if p <= 0:
        valid_choice = torch.multinomial(torch.ones(v).float(), max_image_len)
        select.append(valid_row_idx[i][valid_choice])
    else:
        pad_choice = torch.multinomial(
            torch.ones(nv).float(), p, replacement=True
        )
        select.append(
            torch.cat(
                [valid_row_idx[i], non_valid_row_idx[i][pad_choice]], dim=0,
            )
        )

select = torch.cat(select, dim=0)
x = x[select[:, 0], select[:, 1]].view(B, -1, C)
x_mask = x_mask[select[:, 0], select[:, 1]].view(B, -1)
patch_index = patch_index[select[:, 0], select[:, 1]].view(B, -1, 2)
pos_embed = pos_embed[select[:, 0], select[:, 1]].view(B, -1, C)

print(x_mask.shape)
print(patch_index.shape)
print(pos_embed.shape)