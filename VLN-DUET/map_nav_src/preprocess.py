# export
# PYTHONPATH=/root/mount/Matterport3DSimulator/build:/root/mount/VLN-DUET/map_nav_src

import argparse
import glob
import math
import os
import random
import re

import clip
import cv2 as cv
import h5py
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from reverie.data_utils import load_obj2vps
from utils.data import new_simulator


class Preprocess:
    def __init__(self, path, bboxes, seed=42):
        random.seed(seed)

        self.ix = 0
        self.connectivity_dir = "../datasets/R2R/connectivity"
        self.sim = new_simulator(self.connectivity_dir, scan_data_dir="./data/v1/scans")
        self.bboxes = bboxes

        model, preprocess = clip.load("ViT-L/14", device="cuda")
        self.clip = model
        self.clip_preprocess = preprocess
        self.input_resolution = model.visual.input_resolution

        self.path = path
        self.visualize = False

    @torch.no_grad()
    def get_feats(self, scanId, viewpointId):
        detected_images = []
        visual_feats = []
        objects = {}
        for ix in range(36):
            if ix == 0:
                self.sim.newEpisode([scanId], [viewpointId], [0], [math.radians(-30)])
            elif ix % 12 == 0:
                self.sim.makeAction([0], [1.0], [1.0])
            else:
                self.sim.makeAction([0], [1.0], [0])

            state = self.sim.getState()[0]
            heading = state.heading
            elevation = state.elevation

            rgb = np.array(state.rgb, copy=True)
            img_processed = self.clip_preprocess(Image.fromarray(rgb)).unsqueeze(0).cuda()
            clip_visual_feat = self.clip.encode_image(img_processed).squeeze(0)
            visual_feat = clip_visual_feat.detach().cpu().numpy()
            visual_feats.append(visual_feat)

            obj_ids = []
            _obj_ids = set()
            obj_sizes = []
            obj_directions = []
            if self.visualize and rgb.sum() > 1:
                cv.imwrite("rendering.png", rgb)
                cv.imshow('rendering', rgb)
                cv.waitKey(50)

            key = f"{scanId}_{viewpointId}_{state.viewIndex}"
            if key in self.bboxes:
                for id, bbox in self.bboxes[key]:
                    if id in _obj_ids:
                        continue

                    x, y, w, h = bbox
                    mlef = x == 0 or y == 0
                    mbtm = y + h == 480 or x + w == 640
                    if mlef or mbtm:
                        continue
                    img = rgb[y:y + h, x:x + w, :]
                    img_processed = self.clip_preprocess(Image.fromarray(img)).unsqueeze(0).cuda()
                    detected_clip = self.clip.encode_image(img_processed).squeeze(0)
                    detected_images.append(detected_clip.detach().cpu().numpy())
                    _obj_ids.add(id)
                    obj_ids.append(id)
                    obj_directions.append([heading, elevation])
                    obj_sizes.append([w, h])

                    if self.visualize:
                        cv.imshow(f"{id}", img)
                        cv.waitKey(50)

        return visual_feats, detected_images, obj_directions, obj_sizes, obj_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default="/root/mount/Matterport3DSimulator/data")
    parser.add_argument('--view_file', default="view.hdf5")
    parser.add_argument('--obj_file', default="obj.hdf5")
    # parser.add_argument('--angle_feat_size',default=4)

    print("preprare ...")
    args = parser.parse_args()
    obj2vps, bboxes = load_obj2vps(os.path.join('../datasets/', 'REVERIE', 'annotations', 'BBoxes_v2.json'))
    preprocess = Preprocess(args.data, bboxes)

    paths = glob.glob(f'{args.data}/v1/scans/*')
    pattern = re.compile(r'.+/([^_]+)_.+\..+')

    print("scan ...")
    metas = set()
    for key in tqdm(bboxes.keys()):
        scanId, viewpointId, _ = key.split("_")
        metas.add((scanId, viewpointId))

    print("preprocess ...")
    for (scanId, viewpointId) in tqdm(list(metas)):
        key = f"{scanId}_{viewpointId}"
        img_feats, objs, obj_directions, obj_sizes, obj_ids = preprocess.get_feats(scanId, viewpointId)
        if len(objs) == 0:
            objs = np.zeros((0,768))

        with h5py.File(args.view_file, 'a') as f:
            data = f.create_dataset(key, data=np.array(img_feats,dtype=np.float64))

        with h5py.File(args.obj_file, 'a') as f:
            data = f.create_dataset(key, data=np.array(objs,dtype=np.float64))
            data.attrs['directions'] = np.array(obj_directions,dtype=np.float64)
            data.attrs['sizes'] = np.array(obj_sizes,dtype=np.int64)
            data.attrs['obj_ids'] = np.array(obj_ids,dtype=np.int64)

        # print(key)
        # print(np.array(img_feats,dtype=np.float64).shape,np.array(objs,dtype=np.float64).shape)

        


if __name__ == "__main__":
    main()
