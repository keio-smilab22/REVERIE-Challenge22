export PYTHONPATH=/root/mount/Matterport3DSimulator/build:/root/mount/VLN-DUET/map_nav_src
pip install -r requirements.txt
# mkdir -p datasets/pretrained 
# wget https://nlp.cs.unc.edu/data/model_LXRT.pth -P datasets/pretrained
cd map_nav_src
bash scripts/run_reverie.sh

