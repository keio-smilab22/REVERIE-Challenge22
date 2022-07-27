# 生成されたパスを出力
# Args:
#   split: 
#   data: 
#   res: 
#   outputdir: 
# Output:
#   パス画像: 

import subprocess
import os
import argparse
import json
import cv2

def read_file(split, duet_dir, gt=False):
    # インデックスファイルの読み込み
    # GTの場合このファイルからパスを取得する
    if (split == 'psudo_test_unseen' or split == 'val_half_unseen' or split == 'val_unseen'):
        enc = "val_unseen"
    else:
        enc = "val_seen"
    file_name = "REVERIE_" + enc + "_enc.jsonl"
    idx_file = os.path.join(duet_dir, "VLN-DUET/datasets/REVERIE/annotations/pretrain/", file_name)
    idxs = []
    with open(idx_file, 'r') as fin:
        for line in fin:
            idxs.append(json.loads(line))
    
    file_name = "REVERIE_" + split + ".json"
    sents_file = os.path.join(duet_dir, "VLN-DUET/datasets/REVERIE/annotations/", file_name)
    sents = json.load(open(sents_file, 'r'))


    # 結果ファイルの読みこみ
    file_name = "submit_" + split + "_dynamic.json"
    # res_file = os.path.join(duet_dir, "VLN-DUET/datasets/REVERIE/exprs_map/finetune/dagger-vitbase-seed.0/preds", file_name)
    res_file = os.path.join("/home/vs22/Downloads/ID3/", file_name)
    res = json.load(open(res_file, 'r'))
    return idxs, res, sents


def make_interfile(outputdir, split, duet_dir, gt=False):
    # c++で引数とするための中間ファイルを作成
    idxs, res, sents = read_file(split, duet_dir, gt)
    o_filename = "scene-path.txt"
    inter_file = os.path.join(outputdir, o_filename)
    for res_path in res:
        tmp_idx = None
        for idx in idxs:
            if idx["instr_id"] == res_path["instr_id"]:
                tmp_idx = idx
                break
            if tmp_idx == None:
                continue
        tmp_id = idx["instr_id"][:-2]
        sent_idx = idx["instr_id"][-1]
        for sent in sents:
            if sent["id"] == tmp_id:
                inst = sent["instructions"][int(sent_idx)]
        if gt:
            with open(inter_file, "w") as f:
                for each in tmp_idx["path"]:
                    f.write(each + " ")
        else:
            with open(inter_file, "w") as f:
                for node in res_path["trajectory"]:
                    for each in node:
                        f.write(each + " ")
        make_img(outputdir, tmp_idx, inter_file, inst, split, gt)



def make_img(outputdir, idx, inter_file, inst, split, gt=False):
    with open("make_img.sh", "w") as f:
        f.write("#!/bin/sh" + "\n")
        input_house = "$MATTDATA/" + idx["scan"] + "/house_segmentations/*.house"
        input_scene = "$MATTDATA/" + idx["scan"] + "/matterport_mesh/*/*.obj"
        if (split == 'psudo_test_unseen' or split == 'val_half_unseen'):
            idx["instr_id"] = "un_" + idx["instr_id"]
        if gt:
            img_name = idx["instr_id"] + "_gt.jpg"
        else:
            img_name = idx["instr_id"] + ".jpg"
        output_img = os.path.join(outputdir + img_name)
        command = "../bin/x86_64/mpview -input_house " + input_house + " -input_scene " + input_scene + " -output_image " + output_img + " -path " + inter_file +" -batch"
        f.write(command + "\n")
    subprocess.run("chmod +x make_img.sh", shell=True)
    subprocess.run("./make_img.sh", shell=True)
    del_com = "rm -f " + inter_file
    subprocess.run(del_com, shell=True)
    img = cv2.imread(output_img)
    inst = inst.split()
    lines = 0
    num_words = 0
    text = ""
    for word in inst:
        text += word + " "
        num_words += 1
        if num_words % 12 == 0 or num_words == len(inst):
            where = num_words / 3 * 10
            cv2.putText(img, text, (0, int(where)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            text = ""
    cv2.imwrite(output_img, img)



def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', '-o', type=str, help="画像を格納するディレクトリ")
    parser.add_argument('--split', '-s', choices=['psudo_test_seen', 'psudo_test_unseen', 'val_half_seen', 'val_half_unseen', 'val_unseen', 'val_seen'])
    parser.add_argument('--duet_dir', '-d', type=str, help="DUETの親ディレクトリまでのパス")
    parser.add_argument('-gt', action='store_true')
    args = parser.parse_args() 
    return args


def main():
    args = parse()
    make_interfile(args.output_dir, args.split, args.duet_dir, args.gt)

if __name__ == '__main__':
    main()
