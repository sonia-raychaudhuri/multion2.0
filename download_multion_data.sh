mkdir data
cd data
mkdir datasets
cd datasets
wget -O multinav.zip "http://aspis.cmpt.sfu.ca/projects/multion/multinav.zip"
unzip multinav.zip && rm multinav.zip
cd ../
wget -O multiON_objects.zip "http://aspis.cmpt.sfu.ca/projects/multion/multiON_objects.zip"
unzip multiON_objects.zip && rm multiON_objects.zip
wget -O default.phys_scene_config.json "http://aspis.cmpt.sfu.ca/projects/multion/default.phys_scene_config.json"
cd ../
mkdir oracle_maps
cd oracle_maps
wget -O map300.pickle "http://aspis.cmpt.sfu.ca/projects/multion/map300.pickle"
cd ../
mkdir model_checkpoints
wget -O model_checkpoints/ckpt.2.pth "http://aspis.cmpt.sfu.ca/projects/multion/model_checkpoints/ckpt.2.pth"
