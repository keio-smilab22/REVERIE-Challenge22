import json

with open("submit_test_dynamic.json") as f:
    list_ = json.load(f)

for dict_ in list_:
    dict_["predObjId"] = int(dict_["pred_objid"])
    del dict_["pred_objid"]
    

with open("TeamKeio2_new_round5_ch2.json", 'w') as f2:
    json.dump(list_, f2)