# MPA-RAFT: Motion Perception Attention for Particle Image Velocimetry
This repository contains the source code for our PIV estimation method:
[MPA-RAFT: Motion Perception Attention for Particle Image Velocimetry](https://ieeexplore.ieee.org/document/11145125)

Zhipeng Yang;Jialiang Chi;Hao Du;Jinjun Wang;Zhiqiang Wang;Dongdong Zhang

# Environments
You will have to choose cudatoolkit version to match your compute environment. The code is tested on PyTorch 1.8.0 but other versions may also work.

```pip install -r requirements.txt```

# Train

```python train.py```

# Acknowledgements
This project relies on code from existing repositories: [RAFT](https://github.com/princeton-vl/RAFT), [GMA](https://github.com/zacjiang/GMA/). We thank the original authors for their excellent work.
