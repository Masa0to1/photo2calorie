# photo2calorie

## Overview

Calculate and display calories by importing photos of foods.

## Requirement

- OS: Ubuntu 22.04.3 LTS
- GPU: RTX3060ti
- Lang: Python 3.10.12
- Tools: NVIDIA Driver, NVIDIA CUDA Toolkit, NVIDIA cuDNN

## Usage

```
# Install necessary drivers and tools.
git clone ...
cd photo2calorie
sudo ubuntu-drivers autoinstall
sudo apt-get install nvidia-cuda-toolkit

# Create an 'images' directory.
mkdir images # And save multiple arbitrary images here.

# Install necessary libraries within the virtual environment.
python3 -m venv myenv
source myenv/bin/activate
pip install tensorflow-gpu
pip install matplotlib
pip install Pillow

# Converting Learning Image Data to an npz File
python3 convert.py images

# Loading and Training with npz File
python3 cnn.py image_data/image_data.npz # /path/to/image_data.npz

# Calculate the calorie content of the target image using the trained model.
python3 photo2calorie.py /target_image_file

```

## Licence

[MIT](https://github.com/masaki0to1/photo2calorie/blob/main/LICENSE.md)
