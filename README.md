# Intelligent Human-Computer interface. 
This repository is detailed extensively in the accompanied [Research Paper](https://docs.google.com/document/d/1aNBuz6AemphvAmFaWkQFWymGF39TojJe29UIhg6q1pc/edit?usp=sharing).

### How to Navigate this Repository
#### **_CustomDQN_**
The current version of the Deep Q-Learning Network implimentation. It provides a convience wrapper for training an agent in any enviornment.

#### **_gym_environment_tests_**
DQN algorithm applied to games from the popular benchmark Gym from OpenAI.

#### **_saved_scores_**
Visualizations of various model performance on different tasks.

#### **_future_models_** 
RL models still in production that have no guarantee of their effectiveness.

#### **_code_references_**
Code samples found elsewhere on github, they may be used as a reference when updating files in _future_models_

### How to get started
#### Install Python
```
sudo apt-get install python3.6
```
OpenAI's Universe environments are only supported on Linux and Mac distributions (does not work on Windows). If you only have a Windows OS you have the option to run CustomDQN on the OpenAI's Gym environments (see _gym_environment_tests_). I recommend using Python3.6 to execute the code in this repository.

#### Install Module Dependencies
Once you have python installed, you will need to install the required modules via `pip`:
```
pip install matplotlib numpy tensorflow keras gym universe
```
In order to use _gym_enviornment_tests/LunarLander_ you will need to `pip install gym[box2d] box2d-py`

#### Clone Repository Tree using Git
```
git clone https://github.com/nathanShepherd/Intelligent-Interface.git
```

#### Execute Agent in Linux Terminal 
```
sudo python mwob_Agent.py
```
Or you can observe the Agent control a Lunar Lander 
```
python gym_environment_tests/LunarLander/lunarLander-CustomDQN.py
```                                                                  

#### TODO:
- Use DenseNet to improve classification accuracy

- Use a CRF or RNN/LSTM to help estimate the Q-Function relative to the current point in time

- Augment memory for [_efficient_ and _prioritized_](https://arxiv.org/abs/1511.05952) experience replay
