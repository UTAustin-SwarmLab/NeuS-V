#/bin/bash
# These are the commands that I used to install the necessary packages for the project
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
pip install gradio
pip install transformers
pip install decord
pip install opencv-python
pip install joblib
pip install einops
pip install timm
pip install accelerate
pip install sentencepiece

# Carl
sudo apt install libboost-all-dev m4
sudo apt install libginac-dev libglpk-dev
# sudo apt install build-essential git cmake libboost-all-dev libcln-dev libgmp-dev libginac-dev automake libglpk-dev libhwloc-dev

cd FILLEMEUP
git clone https://github.com/moves-rwth/carl-storm
cd carl-storm
mkdir build
cd build
cmake ..
make lib_carl

# Storm
mkdir build
cd build
wget https://github.com/moves-rwth/storm/archive/stable.zip
unzip stable.zip
cd storm-stable
mkdir build
cd build
cmake ../ -DCMAKE_BUILD_TYPE=Release -DSTORM_DEVELOPER=OFF -DSTORM_LOG_DISABLE_DEBUG=ON -DSTORM_PORTABLE=ON -DSTORM_USE_SPOT_SHIPPED=ON make
make -j12
# export PATH=$PATH:/opt/storm/build/bin

pip install stormpy
