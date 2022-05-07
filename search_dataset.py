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
        "mlm": 0,
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
    lst_imgs = [f"C:\\Users\\alimh\\PycharmProjects\\ViLT\\assets\\database\\{i}.jpg" for i in range(1,6)]

    def infer(url, mp_text, hidx):
        def get_image(path):
            image = Image.open(path).convert("RGB")
            img = pixelbert_transform(size=384)(image)
            return img.unsqueeze(0).to(device)

        imgs = [get_image(pth) for pth in lst_imgs]


        batch = []
        for img in imgs  :
            batch.append({"text": [mp_text], "image": [img]})

        for dic in batch:
            encoded = tokenizer(dic["text"])

            dic["text_ids"] = torch.tensor(encoded["input_ids"]).to(device)
            dic["text_labels"] = torch.tensor(encoded["input_ids"]).to(device)
            dic["text_masks"] = torch.tensor(encoded["attention_mask"]).to(device)

        scores = []
        with torch.no_grad():

            for dic in batch :
                s= time.time()
                infer = model(dic)
                e = time.time()
                print("time ",round(e-s,2))

                score = model.rank_output(infer["cls_feats"])
                scores.append(score.item())
            print(scores)
            print(np.argmax(scores)+1)


    infer(0,"group of men setting around a table in a room",0)
    infer(0,"a blue train with people standing on it",0)
    infer(0, "woman and a child standing in train station with bags", 0)
    infer(0, "banana shop", 0)
