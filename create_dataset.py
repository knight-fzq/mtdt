import pickle
import os
import json
from collections import namedtuple
import numpy as np

config_path_dict = {
    'cheetah_vel': "cheetah_vel/cheetah_vel_40.json",
    'cheetah_dir': "cheetah_dir/cheetah_dir_2.json",
    'ant_dir': "ant_dir/ant_dir_50.json",
    'ML1-pick-place-v2': "ML1-pick-place-v2/ML1_pick_place.json",
}
data_save_path = 'data'
env = 'ML1-pick-place-v2'

task_config = os.path.join('config', config_path_dict[env])
with open(task_config, 'r') as f:
    task_config = json.load(f, object_hook=lambda d: namedtuple('X', d.keys())(*d.values()))
train_env_name_list, test_env_name_list = [], []
for task_ind in task_config.test_tasks:
    test_env_name_list.append(env +'-'+ str(task_ind))

for env_name in test_env_name_list:
  dataset_path = data_save_path+f'/{env}/{env_name}-expert.pkl'
  with open(dataset_path, 'rb') as f:
      trajectories = pickle.load(f)
  
  states, traj_lens, returns = [], [], []
  for path in trajectories:
    states.append(path['observations'])
    traj_lens.append(len(path['observations']))
    returns.append(path['rewards'].sum())
  traj_lens, returns = np.array(traj_lens), np.array(returns)

  sorted_inds = np.argsort(returns)  # lowest to highest

  traj256 = []
  batch_inds = np.random.choice(
      np.arange(len(traj_lens)),
      size=256,
      replace=True,
  )
  for i in range(256):
     traj256.append(trajectories[batch_inds[i]])
  
  traj2_path = data_save_path+f'/{env}/{env_name}-num256-expert.pkl'
  with open(traj2_path, 'wb') as f:
    pickle.dump(traj256, f)
  

  # random_prompt = []
  # for i in range(5):
  #   random_prompt.append(trajectories[sorted_inds[i]])
  #   print(random_prompt[i]['rewards'].sum())

  # dataset_random_path = data_save_path+f'/{env}/{env_name}-prompt-random.pkl'
  # with open(dataset_random_path, 'wb') as f:
  #   pickle.dump(random_prompt, f)
  
  # medium_prompt = []
  # for i in range(5):
  #   medium_prompt.append(trajectories[sorted_inds[50+i]])
  #   print(medium_prompt[i]['rewards'].sum())

  # dataset_medium_path = data_save_path+f'/{env}/{env_name}-prompt-medium.pkl'
  # with open(dataset_medium_path, 'wb') as f:
  #   pickle.dump(random_prompt, f)

  # expert_prompt = []
  # for i in range(5):
  #   expert_prompt.append(trajectories[sorted_inds[-(i+1)]])
  #   print(expert_prompt[i]['rewards'].sum())
  # print('------------')
  # print('----------')