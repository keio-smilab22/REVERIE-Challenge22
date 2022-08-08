# REVERIE-Challenge 2022

This repository is the official implementation of TeamKeio

## Requirements
The training and inference were conducted on a machine equipped with a GeForce RTX 3090 with 24 GB of GPU memory, and an Intel Corei9 10900K processor. 

0. Install docker, nvidia-docker2

1. Install Matterport3D simulators: follow instructions [here](https://github.com/peteanderson80/Matterport3DSimulator). We use the latest version instead of v0.1.
```
git clone --recursive git@github.com:keio-smilab22/REVERIE-Challenge22.git
cd REVERIE-Challenge22/Matterport3DSimulator
export MATTERPORT_DATA_DIR = <path to unzipped data>
docker build -t mattersim:9.2-devel-ubuntu18.04 .
sudo nvidia-docker run --runtime=nvidia -it --mount type=bind,source=$MATTERPORT_DATA_DIR,target=/root/mount/Matterport3DSimulator/data --volume `pwd`:/root/mount/Matterport3DSimulator mattersim:9.2-devel-ubuntu18.04

# in container
cd /root/mount/Matterport3DSimulator
mkdir build && cd build
cmake -DEGL_RENDERING=ON ..
make
cd ../
./scripts/downsize_skybox.py
./scripts/depth_to_skybox.py
exit
```

2. Download data from [Dropbox](https://www.dropbox.com/sh/u3lhng7t2gq36td/AABAIdFnJxhhCg2ItpAhMtUBa?dl=0). Put the data in `REVERIE-Challenge22/VLN-DUET/datasets' directory. We **only use 'R2R/connectivity'**.

3. Download annotation data and preprocessed feature from [Google Drive](https://drive.google.com/drive/folders/1svrvFcpfLarWh-DO1hH4O8ZeYRRuoZ7t?usp=sharing). We preprocessed [v2 data](https://github.com/YuankaiQi/REVERIE/tree/master/tasks/REVERIE/data_v2). もしpreprocessの方法を知りたいということであれば，お知らせください．
```
unzip annotaions.zip
mv add_instr_encodings/* BBox_v2.json <path to REVERIE-Challenge22>/VLN-DUET/datasets/REVERIE/annotations/
mv obj.hdf5 <path to REVERIE-Challenge22>/VLN-DUET/datasets/REVERIE/features/
mv view.hdf5 <path to REVERIE-Challenge22>/VLN-DUET/datasets/R2R/features/
```


3. Download pretrained lxmert
```
cd ../VLN-DUET
mkdir -p datasets/pretrained 
wget https://nlp.cs.unc.edu/data/model_LXRT.pth -P datasets/pretrained
```

## Training & Evaluation

```
cd ../
export REV_DIR=$(pwd)
cd Matterport3DSimulator
sudo nvidia-docker run -it --shm-size=256m -e="QT_X11_NO_MITSHM=1" -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix --mount type=bind,source=$MATTERPORT_DATA_DIR,target=/root/mount/Matterport3DSimulator/data,readonly --mount type=bind,source=$REV_DIR/VLN-DUET,target=/root/mount/VLN-DUET --volume `pwd`:/root/mount/Matterport3DSimulator mattersim:9.2-devel-ubuntu18.04

# in container
export PYTHONPATH=/root/mount/Matterport3DSimulator/build:/root/mount/VLN-DUET/map_nav_src
cd /root/mount/VLN-DUET/map_nav_src
pip install -r requirements.txt
bash scripts/run_reverie.sh
```

After evaluation, you can get a checkpoint file (`../datasets/REVERIE/exprs_map/finetune/dagger-vitbase-seed.0/ckpts/best_val_unseen`) and predictions (`../datasets/REVERIE/exprs_map/finetune/dagger-vitbase-seed.0/preds/submit_test_dynamic.json`)

You can get a checkpoint and sample predictions from [here](https://drive.google.com/drive/u/2/folders/1svrvFcpfLarWh-DO1hH4O8ZeYRRuoZ7t).

For REVERIE Challenge 2022, you have to modify `submit_test_dynamic.json` before submission. 
```
cd VLN-DUET/map_nav_src
python make_submit.py
```
