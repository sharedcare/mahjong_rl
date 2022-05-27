# Mahjong
The objective of this project is to build a reasonable AI agent to play Mahjong with human players.

The enviroment of Mahjong is built by [RLCard](https://github.com/datamllab/rlcard).

# Requirements
* PyTorch
* RLCard

# Usage
Train PPO Agent
```
python main.py --log_dir 'experiments/mahjong_ppo_result/'
```
Play Demo with trained agent
```
python demo.py
```

## Todo
- [x] PPO Agent
- [x] Demo
- [ ] IPPO Agent
- [ ] MAPPO Agent
- [ ] SAC Agent
- [ ] Add other types of Mahjong games
