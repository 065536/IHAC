# [IHAC: a LLM-Directed Hierarchical Reinforcement Learning Framework]

## Abstract 
Hierarchical Reinforcement Learning (HRL) aims to alleviate the curse of dimensionality. Issue in real-world RL by introducing hierarchical structures into policies and breaking down original policies into high-level policies and low-level policies, aligning more closely with how humans learn complex tasks. Recent works introduced large language model (LLM) as a candidate to provide high-level policies due to its power of representation. However, deploying LLM is both time and resource-consuming. In this work, we study how to efficiently learn a controller model by abstracting the instructions from LLM. The key novelty of our work is a new alignment scheme that aligns the controller model with original HRL model with robustness and efficiency. We implement our framework in MiniGird environment. Experimental results show that our algorithm can quickly explore rewards in the early stages and significantly outperform existing baselines. Our codes are available at \url{https://github.com/065536/IHAC}.

<!-- ## Purpose
This repo is intended to serve as a foundation with which you can reproduce the results of the experiments detailed in our paper, [Large Language Model as a Policy Teacher for Training Reinforcement Learning Agents](https://arxiv.org/abs/2311.13373). -->


## Running experiments
### Setup the LLMs

For Vicuna models, please follow the instruction from [FastChat](https://github.com/lm-sys/FastChat) to install Vicuna model on local sever. Here are the commands to launch the API in terminal: 

```bash
python3 -m fastchat.serve.controller --host localhost         ### Launch the controller
python3 -m fastchat.serve.model_worker --model-name 'meta_controller' --model-path lmsys/vicuna-7b-v1.5  ### Launch the model worker
python3 -m fastchat.serve.openai_api_server --host localhost        ### Launch the API
```


### Train and evaluate the models
Any algorithm can be run from the main.py entry point.

To train on a SimpleDoorKey environment,

```bash
python main.py train --task SimpleDoorKey --savedir train
```

<!--to train with given query result from LLM as teacher,

```bash
python main.py train --task SimpleDoorKey --savedir train --offline_planner
```-->

To evaluate the trained model,

```bash
python main.py eval --task SimpleDoorKey --loaddir train --savedir eval
```

To evaluate the LLM-based teacher baseline,
```bash
python main.py eval --task SimpleDoorKey --loaddir train --savedir eval --eval_teacher
```

## Logging details 
Tensorboard logging is enabled by default for all algorithms. The logger expects that you supply an argument named ```logdir```, containing the root directory you want to store your logfiles

The resulting directory tree would look something like this:
```
log/                         # directory with all of the saved models and tensorboard 
└── ppo                                 # algorithm name
    └── simpledoorkey                   # environment name
        └── save_name                   # unique save name 
            ├── acmodel.pt              # actor and critic network for algo
            ├── events.out.tfevents     # tensorboard binary file
            └── config.json             # readable hyperparameters for this run
```

Using tensorboard makes it easy to compare experiments and resume training later on.

To see live training progress

Run ```$ tensorboard --logdir=log``` then navigate to ```http://localhost:6006/``` in your browser

<!-- ## Citation
If you find [our work](https://arxiv.org/abs/2311.13373) useful, please kindly cite: 
```bibtex
@article{zhou2023large,
  title={Large Language Model as a Policy Teacher for Training Reinforcement Learning Agents},
  author={Zhou, Zihao and Hu, Bin and Zhao, Chenyang and Zhang, Pu and Liu, Bin},
  journal={arXiv preprint arXiv:2311.13373},
  year={2023}
}
``` -->


