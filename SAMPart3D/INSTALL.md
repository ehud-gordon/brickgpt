### Installation

We test our model on a 24G RTX4090 GPU with Python 3.10, CUDA 12.1 and Pytorch 2.1.0.

1. Install basic modules: torch and packages in requirements.txt
    ```bash
    conda create -n sampart3d
    conda activate sampart3d
    pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
    pip install -r requirements.txt
    ```

2. Install modules for PTv3-object
    ```bash
    cd libs/pointops
    python setup.py install
    cd ../..

    # spconv (SparseUNet)
    # refer https://github.com/traveller59/spconv
    pip install spconv-cu120  # choose version match your local cuda
    ```

    Following [README](https://github.com/Dao-AILab/flash-attention) in Flash Attention repo and install Flash Attention for PTv3-object.


3. Install modules for acceleration (necessary in current version of code)
    ```bash
    pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

    # using GPU-based HDBSCAN clustering algorithm
    # refer https://docs.rapids.ai/install
    pip install --extra-index-url=https://pypi.nvidia.com cudf-cu11==24.6.* cuml-cu11==24.6.*
    ```
