## Getting Started

Install Miniconda if conda is not already installed. [Source](https://docs.anaconda.com/miniconda/install/).

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
```

Start a new conda environment.
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc
conda activate myenv
```

Clone this repo. Then setup using Python only build. [Source](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/index.html?device=cuda#build-wheel-from-source).

```bash
cd vllm
VLLM_USE_PRECOMPILED=1 pip install --editable .
```
This build may fail if there are certain packages that are not installed yet. Just install whatever the error output says to install via pip. In particular you may have to install
```bash
pip install --upgrade setuptools setuptools_scm
pip install librosa datasets gradio
```
Afterwards, start running the python web server.
```bash
python examples/offline_inference/whisper_evaluation_server.py  
```

