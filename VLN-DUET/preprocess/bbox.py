import os
import json

with open("ground_bbox.txt", "r") as tf:
    bbox_list = tf.read().split('\n')

result_dict = dict()

for i in bbox_list:
    new_dict = dict()
    if i == '':
        continue
    with open(i) as f:
        bbox_file = json.load(f)
    _, file_name = i.split('/')
    scan, vp_old = file_name.split('_')
    vp, _ = vp_old.split('.')
    for j in bbox_file:
        for k in bbox_file[j]:
            if k not in new_dict:
                new_dict[k] = dict()
                new_dict[k]['name'] = bbox_file[j][k]["name"]
                new_dict[k]['visible_pos'] = []
                new_dict[k]['visible_pos'].append(j)
                new_dict[k]['bbox2d'] = []
                new_dict[k]['bbox2d'].append(bbox_file[j][k]["bbox2d"])
            else:
                if j not in new_dict[k]['visible_pos']:
                    new_dict[k]['visible_pos'].append(j)
                new_dict[k]['bbox2d'].append(bbox_file[j][k]["bbox2d"])
    result_dict[scan + '_' + vp] = new_dict

with open('BBoxes_v2.json', 'w') as f:
    json.dump(result_dict, f, indent=4)