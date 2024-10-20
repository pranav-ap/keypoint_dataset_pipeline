# Instructions

## Connect

```commandline
ssh ath@atcremers75.in.tum.de -p 58022
```

## Install Miniconda

```commandline
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
conda init --all
```

## Setup Repo

```commandline
git clone https://github.com/pranav-ap/keypoint_dataset_pipeline.git
git checkout ath
```

## Setup Environment

```commandline
conda create -n kd_pipeline python=3.13
conda activate kd_pipeline 
pip install -r requirements.txt
python --version
pip list
```

