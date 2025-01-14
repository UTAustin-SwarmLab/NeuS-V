#/bin/bash
# These are the commands that I used to install the necessary packages for the project
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
pip install gradio
pip install transformers
pip install decord
pip install opencv-python
pip install joblib

# Storm
sudo apt install libboost-all-dev m4

mkdir build
cd build
wget https://github.com/moves-rwth/storm/archive/stable.zip
unzip stable.zip
cd storm-stable
mkdir build
cd build
cmake ..

# Carl
cd FILLEMEUP
git clone https://github.com/moves-rwth/carl-storm
cd carl-storm
mkdir build
cd build
cmake ..
make lib_carl



pip install pycarl
pip install stormpy