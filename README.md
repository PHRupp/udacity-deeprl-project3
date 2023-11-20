# udacity-deeprl-project3
Udacity: Deep Reinforcement Learning Project 3

This project trains an agent to interact with Udacity's Tennis environment where the agent controls a racket that gets +0.1 reward for hitting the ball across the net. Additionally, reward of -0.01 if the ball touches the ground or goes out of bounds. 

The code is written in PyTorch and Python 3.

## Environment
The below is a paraphrasing from the Udacity course's repo regarding this project's environment:

The goal of the agent is to keep the arm within the floating ball as long as possible. The goal is to get an average score of >=0.5 over 100 consecutive episodes.

The state space has a total 24 dimensions (8 dimensions across 3 different perspectives) and contains the agent's position/velocity as well as the ball's. The agent's action space consists of a 2 dimensional vector where each value is bounded [-1, 1] and represents velocity of the rackets.

## Getting Started

After following the instructions defined here for downloading and installing: https://github.com/udacity/deep-reinforcement-learning/tree/master

My installation was based on 
* Windows 11 x64
* Python 3.6.13 :: Anaconda, Inc.
* Reacher Unity Build: https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip

```bash
# Instructions from Deep RL course for environment setup
conda create --name drlnd python=3.6 
activate drlnd

# after creating new conda python environment
cd <path/to/dev/directory>
git clone https://github.com/udacity/deep-reinforcement-learning.git
git clone https://github.com/PHRupp/udacity-deeprl-project3.git
pushd deep-reinforcement-learning/python

# HACK: edit requirements.txt in deep-rl env to make "torch==0.4.1" since i couldnt get a working version for 0.4.0
pip install https://download.pytorch.org/whl/cu92/torch-0.4.1-cp36-cp36m-win_amd64.whl
pip install .

# install packages used specifically in my code
pip install matplotlib==3.3.4 numpy==1.19.5 pandas==1.1.5
popd
pushd udacity-deeprl-project3
```

## Usage

In order to run the code, we run it straight via python instead of using jupyter notebooks.

As depicted in the Report.pdf, you can change the paramters in main.py to get different results. Otherwise, you can run the code as-is to get the same results assuming a random seed = 8675309. 

```python
# run from 'udacity-deeprl-project3' directory
python .\src\main.py
```
