token = "5017314876:AAHLfcYm2rSCAtA1Vi420uowzpaFV4tthjk"

import random
import time
import torch
import copy
from vilt.config import ex
from vilt.modules import ViLTransformerSS
from vilt.transforms import pixelbert_transform
from vilt.datamodules.datamodule_base import get_pretrained_tokenizer
from telegram import *
from telegram.ext import *
from PIL import Image
import numpy as np
from io import BytesIO
import string
import pandas as pd
import glob

bio = BytesIO()
bio.name = 'image.jpeg'
all_imgs = glob.glob("C:\\Users\\alimh\\Downloads\\val2017\\val2017\\*.jpg")
lst_imgs = random.sample(all_imgs, 100)

input_imgs = []

imgs = [Image.open(pth).convert("RGB") for pth in lst_imgs]

lst_response = []
randomImage = "get image"
saw_img = False
updater = Updater(token,
                  use_context=True)


@ex.automain
def load_model(_config):
    def get_image(path):
        image = Image.open(path).convert("RGB")
        img = pixelbert_transform(size=384)(image)
        return img.unsqueeze(0).to(device)

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
    global tokenizer
    tokenizer = get_pretrained_tokenizer(_config["tokenizer"])

    _config.update(
        {
            "loss_names": loss_names,
        }
    )

    global model
    model = ViLTransformerSS(_config)
    model.setup("test")
    model.eval()
    global device
    device = "cuda:0" if _config["num_gpus"] > 0 else "cpu"
    model.to(device)
    global input_imgs
    input_imgs = [get_image(pth) for pth in lst_imgs]



def infer(mp_text):
    batch = []
    print("running inference .... ")
    for img in input_imgs:
        batch.append({"text": [mp_text], "image": [img]})

    for dic in batch:
        encoded = tokenizer(dic["text"])
        dic["text_ids"] = torch.tensor(encoded["input_ids"]).to(device)
        dic["text_labels"] = torch.tensor(encoded["input_ids"]).to(device)
        dic["text_masks"] = torch.tensor(encoded["attention_mask"]).to(device)

    # global input_imgs
    # for im in input_imgs:
    #     print(im.shape)
    #
    #
    # batch["image"] = torch.stack(input_imgs).to(device)
    # print(batch["image"].shape)
    # batch["text"] = [mp_text]*len(imgs)
    # encoded = tokenizer(batch["text"])
    # batch["text_ids"] = torch.tensor(encoded["input_ids"]).to(device)
    # batch["text_labels"] = torch.tensor(encoded["input_ids"]).to(device)
    # batch["text_masks"] = torch.tensor(encoded["attention_mask"]).to(device)
    scores = []

    with torch.no_grad():
        # infer = model(batch)
        # scores = model.rank_output(infer["cls_feats"])

        for i,dic in enumerate(batch):
            # if i > 600 :
            #     dic["image"][0]= dic["image"][0].to(device)
            # print(f" text {mp_text}, image size {dic['image'][0].shape}")

            infer = model(dic)

            # print("time ",round(e-s,2))
            score = model.rank_output(infer["cls_feats"])
            scores.append(score.item())

        # # print(scores)
        answer = np.argmax(scores)
        sort_index = np.flip(np.argsort(scores))

    return answer, sort_index


def start(update: Update, context: CallbackContext):
    load_model()
    buttons = [[KeyboardButton(randomImage)], ]
    context.bot.send_message(chat_id=update.effective_chat.id, text="Welcome to Image Retrieval bot!",
                             reply_markup=ReplyKeyboardMarkup(buttons))
def reset(update: Update, context: CallbackContext):
    buttons = [[KeyboardButton(randomImage)], ]
    context.bot.send_message(chat_id=update.effective_chat.id, text="Welcome to Image Retrieval bot!",
                             reply_markup=ReplyKeyboardMarkup(buttons))

def help(update: Update, context: CallbackContext):
    update.message.reply_text("Your Message")


def unknown(update: Update, context: CallbackContext):
    update.message.reply_text("unkown command")


def unknown_text(update: Update, context: CallbackContext):
    update.message.reply_text("unkown command")


def getImg():
    indx = random.randint(0, len(lst_imgs) - 1)
    print(f"random image index {indx} image path {lst_imgs[indx]}")
    return imgs[indx], indx


def imgBio(sample_img):
    bio = BytesIO()
    name = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
    bio.name = name + '.jpeg'
    sample_img.save(bio, 'JPEG')
    bio.seek(0)
    return bio


def messageHandler(update: Update, context: CallbackContext):
    global saw_img
    if randomImage in update.message.text:
        print(f" getting image for {update.effective_chat.id}")
        image, idx = getImg()

        saw_img = True
        context.user_data["img_idx"] = idx
        context.user_data["img"] = bio

        if not image is None:
            context.bot.sendMediaGroup(chat_id=update.effective_chat.id,
                                       media=[InputMediaPhoto(imgBio(image), caption="please caption this image!")])
            context.bot.send_message(chat_id=update.effective_chat.id,
                                     text="waiting for your input ... ")

    else:

        if not saw_img:
            context.user_data["img_idx"] = -1

        saw_img = False

        context.user_data["user_input"] = update.message.text
        context.bot.send_message(chat_id=update.effective_chat.id,
                                 text="processing .... ")

        buttons = [[InlineKeyboardButton("👍", callback_data="like")],
                   [InlineKeyboardButton("👎", callback_data="dislike")]]
        cap = update.message.text
        s = time.time()
        ans, idx_rank = infer(cap)
        e = time.time()
        print("time ",round(e-s,2))
        image = imgs[ans]
        if context.user_data["img_idx"] == idx_rank[0]:
            context.user_data["rank"] = 1
        elif context.user_data["img_idx"] in idx_rank[:5]:
            context.user_data["rank"] = 5
        elif context.user_data["img_idx"] in idx_rank[:10]:
            context.user_data["rank"] = 10
        else:
            context.user_data["rank"] = -1

        context.bot.sendMediaGroup(chat_id=update.effective_chat.id,
                                   media=[InputMediaPhoto(imgBio(image), caption="this what model found!")])
        context.bot.send_message(chat_id=update.effective_chat.id,
                                 reply_markup=InlineKeyboardMarkup(buttons),
                                 text="Do you think the image matches your caption?")


def queryHandler(update: Update, context: CallbackContext):
    query = update.callback_query.data
    update.callback_query.answer()
    lst_response.append({"chat_id": update.effective_chat.id,
                         "img_indx": context.user_data["img_idx"],
                         "response": 1 if "like" == query else 0,
                         "rank": context.user_data["rank"],
                         "query": context.user_data["user_input"]
                         })
    df = pd.DataFrame.from_dict(lst_response)
    df.to_excel('results_1000k.xlsx')
    print(f"list {lst_response}")
    buttons = [[KeyboardButton(randomImage)], ]
    context.bot.send_message(chat_id=update.effective_chat.id, text="Do you want to try again?",
                             reply_markup=ReplyKeyboardMarkup(buttons))


updater.dispatcher.add_handler(CommandHandler('start', start))
updater.dispatcher.add_handler(CommandHandler('reset', start))
updater.dispatcher.add_handler(CommandHandler('help', help))
updater.dispatcher.add_handler(MessageHandler(Filters.text, messageHandler))
updater.dispatcher.add_handler(CallbackQueryHandler(queryHandler))
# updater.dispatcher.add_handler(EnvironmentError(errorHandler))


updater.start_polling()
