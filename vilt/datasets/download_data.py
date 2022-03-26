from img2dataset import download
import shutil
import os
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('data_path', type=str,
                    help='path to tsv file that contains urls and captions')
parser.add_argument('out_path', type=str, default="./",
                    help='path to write images')
parser.add_argument('dataset', type=str, default="images_val",
                    help='images_train or images_val')

args = parser.parse_args()


if __name__=="__main__" :
    output_dir = os.path.abspath(args.out_path)
    output_dir = os.path.join(output_dir,args.dataset)
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    download(
        processes_count=2,
        thread_count=4,
        url_list=args.data_path,
        image_size=256,
        resize_mode="no",
        output_folder=output_dir,
        output_format="files",
        input_format="tsv",
        url_col="url",
        caption_col="caption",
        enable_wandb=False,
        number_sample_per_shard=1000,
        distributor="multiprocessing",
    )
    lst = []
    for path, subdirs, files in os.walk(output_dir):
        for name in files:
            full_name_path = os.path.join(path, name)
            if "stats" in name or "parquet" in name or name.endswith(".txt") :
                os.remove(full_name_path)
            if "train_annot" in name or "val_annot" in name :
                continue
            elif name.endswith(".json"):
                with open(full_name_path, 'rb') as f:
                    tmp = json.load(f)
                    lst.append((full_name_path,tmp["caption"]))
                os.remove(full_name_path)
            else :
                base = full_name_path.split(".")[0]
                os.rename(full_name_path,base)

    json_name = "train_annot"
    if args.dataset.endswith("val"):
        json_name = "val_annot"

    with open(os.path.join(args.out_path, json_name), 'w') as f:
        json.dump(lst, f)
