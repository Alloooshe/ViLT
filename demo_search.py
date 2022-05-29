import gradio as gr
import torch
import copy
import time
import requests
import io
import numpy as np
import re
from einops import rearrange

import ipdb

from PIL import Image

from vilt.config import ex
from vilt.modules import ViLTransformerSS

from vilt.modules.objectives import cost_matrix_cosine, ipot
from vilt.transforms import pixelbert_transform
from vilt.datamodules.datamodule_base import get_pretrained_tokenizer


@ex.automain
def main(_config):
    _config = copy.deepcopy(_config)

    loss_names = {
        "itm": 1,
        "mlm": 0.5,
        "mpp": 0,
        "vqa": 0,
        "imgcls": 0,
        "nlvr2": 0,
        "irtr": 1,
        "arc": 0,
    }
    tokenizer = get_pretrained_tokenizer(_config["tokenizer"])

    _config.update(
        {
            "loss_names": loss_names,
        }
    )

    model = ViLTransformerSS(_config)
    model.setup("test")
    model.eval()

    device = "cuda:0" if _config["num_gpus"] > 0 else "cpu"
    model.to(device)
    lst_imgs = [f"C:\\Users\\alimh\\PycharmProjects\\ViLT\\assets\\database\\{i}.jpg" for i in range(1,10)]


    def infer( mp_text, hidx =0 ):
        def get_image(path):
            image = Image.open(path).convert("RGB")
            img = pixelbert_transform(size=384)(image)
            return img.unsqueeze(0).to(device)

        imgs = [get_image(pth) for pth in lst_imgs]

        batch = []
        for img in imgs:
            batch.append({"text": [mp_text], "image": [img]})

        for dic in batch:
            encoded = tokenizer(dic["text"])

            dic["text_ids"] = torch.tensor(encoded["input_ids"]).to(device)
            dic["text_labels"] = torch.tensor(encoded["input_ids"]).to(device)
            dic["text_masks"] = torch.tensor(encoded["attention_mask"]).to(device)

        scores = []
        with torch.no_grad():

            for dic in batch:
                s = time.time()
                infer = model(dic)

                e = time.time()
                print("time ", round(e - s, 2))

                score = model.rank_output(infer["cls_feats"])
                scores.append(score.item())
            print(scores)
            img_idx =np.argmax(scores)
            print(np.argmax(scores) + 1 )
            selected_image =  Image.open(lst_imgs[img_idx]).convert("RGB")
            selected_image = np.asarray(selected_image)
            print(selected_image.shape)
            selected_token =""
            if hidx > 0 and hidx < len(encoded["input_ids"][0][:-1]):
                image = Image.open(lst_imgs[img_idx]).convert("RGB")
                selected_batch  = batch[img_idx]
                with torch.no_grad():
                    infer = model(selected_batch)
                    txt_emb, img_emb = infer["text_feats"], infer["image_feats"]
                    txt_mask, img_mask = (
                        infer["text_masks"].bool(),
                        infer["image_masks"].bool(),
                    )
                    for i, _len in enumerate(txt_mask.sum(dim=1)):
                        txt_mask[i, _len - 1] = False
                    txt_mask[:, 0] = False
                    img_mask[:, 0] = False
                    txt_pad, img_pad = ~txt_mask, ~img_mask

                    cost = cost_matrix_cosine(txt_emb.float(), img_emb.float())
                    joint_pad = txt_pad.unsqueeze(-1) | img_pad.unsqueeze(-2)
                    cost.masked_fill_(joint_pad, 0)

                    txt_len = (txt_pad.size(1) - txt_pad.sum(dim=1, keepdim=False)).to(
                        dtype=cost.dtype
                    )
                    img_len = (img_pad.size(1) - img_pad.sum(dim=1, keepdim=False)).to(
                        dtype=cost.dtype
                    )
                    T = ipot(
                        cost.detach(),
                        txt_len,
                        txt_pad,
                        img_len,
                        img_pad,
                        joint_pad,
                        0.1,
                        1000,
                        1,
                    )

                    plan = T[0]
                    plan_single = plan * len(txt_emb)
                    cost_ = plan_single.t()

                    cost_ = cost_[hidx][1:].cpu()

                    patch_index, (H, W) = infer["patch_index"]
                    heatmap = torch.zeros(H, W)
                    for i, pidx in enumerate(patch_index[0]):
                        h, w = pidx[0].item(), pidx[1].item()
                        heatmap[h, w] = cost_[i]

                    heatmap = (heatmap - heatmap.mean()) / heatmap.std()
                    heatmap = np.clip(heatmap, 1.0, 3.0)
                    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

                    _w, _h = image.size
                    overlay = Image.fromarray(np.uint8(heatmap * 255), "L").resize(
                        (_w, _h), resample=Image.NEAREST
                    )
                    image_rgba = image.copy()
                    image_rgba.putalpha(overlay)
                    selected_image =  image_rgba

                    selected_token = tokenizer.convert_ids_to_tokens(
                        encoded["input_ids"][0][hidx]
                    )


        return [selected_image,hidx]

    imgs = [Image.open(pth).convert("RGB") for pth in lst_imgs]
    inputs = [

        gr.inputs.Textbox(label="Caption with [MASK] tokens to be filled.", lines=5),
        gr.inputs.Slider(
            minimum=0,
            maximum=38,
            step=1,
            label="Index of token for heatmap visualization (ignored if zero)",
        ),
    ]
    outputs = [
        gr.outputs.Image(label="Image"),


        gr.outputs.Textbox(label="matching index "),
    ]


    interface = gr.Interface(
        fn=infer,
        inputs=inputs,
        outputs=outputs,
        server_name="localhost",
        server_port=8888,

    )

    interface.launch(debug=True,share=False)