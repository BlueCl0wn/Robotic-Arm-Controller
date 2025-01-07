import gymnasium as gym
import gymnasium_robotics
import numpy as np
import argparse
import torch
from importlib.metadata import version
from models.nn_model import NeuralNetworkModel
from solvers.dqn import optimize_model, initiate_stuff, select_action
from models import RandomModel
from train import flatten_dict
import pickle

# ['__annotations__', '__class__', '__class_getitem__', '__delattr__', '__dict__', '__dir__', '__doc__', '__enter__', '__eq__', '__exit__', '__format__', '__ge__', '__getattribute__', '__getstate__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__lt__', '__module__', '__ne__', '__new__', '__orig_bases__', '__parameters__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__setstate__', '__sizeof__', '__slots__', '__str__', '__subclasshook__', '__weakref__', '_clean_particles', '_create_particle', '_destroy', '_ezpickle_args', '_ezpickle_kwargs', '_is_protocol', '_np_random', '_np_random_seed', 'action_space', 'clock', 'close', 'continuous', 'crash_penalty', 'enable_wind', 'get_wrapper_attr', 'gravity', 'has_wrapper_attr', 'isopen', 'lander', 'metadata', 'moon', 'np_random', 'np_random_seed', 'observation_space', 'particles', 'prev_reward', 'render', 'render_mode', 'reset', 'reward_shaping', 'screen', 'set_wrapper_attr', 'spec', 'step', 'turbulence_power', 'unwrapped', 'wind_power', 'world']

def make_env():
    instance = gym.make("FetchReach-v3", max_episode_steps=200, reward_type="dense", render_mode="human")
    return instance


env = make_env()

observation, info = env.reset(seed=42)

parser = argparse.ArgumentParser(description='Stochastic Neural Network')

parser.add_argument("--resume", type=str, default=None, help="Parse URI of a trained model. If left empty a random_model is used for the simulation.")
parser.add_argument("--save_as_csv", type=bool, default=False, help="save as csv")

args = parser.parse_args()
model_path = args.resume

if model_path is None:
    model = RandomModel()
else:
    loaded_data = torch.load(model_path, weights_only=False)
    stuff, i, params = loaded_data

def get_action(observation, i):
    """

    :return:
    """
    observation = flatten_dict(observation)
    if model_path is None:
        return model.make_decision(4)
    else:
        state = torch.tensor(observation, dtype=torch.float32, device=params.device).unsqueeze(0)
        action = select_action(env, state, *stuff, i, params)
        return action.tolist()

# set duration of runtime loop
N = 100000

# stuff for saving observations and actions in csv
if args.save_as_csv:
    csv_data = ""
    N = 150 * 100

# stuff = initiate_stuff()

# loop
summed_reward = 0
for _ in range(N):
    #observation = torch.tensor(observation, dtype=torch.float32)
    # action = model(observation)
    action = get_action(observation, _)
    #action = np.argmax(action.detach().numpy())
    if args.save_as_csv:
        csv_data += ','.join([str(round(float(i), 2)) for i in observation.to('cpu').detach().numpy()]) + "," + str(
            action) + "\n"
        print(_) if _ % 100 == 0 else None

    # print(action, '\t'.join([str(round(float(i),2)) for i in observation.to('cpu').detach().numpy()]))
    observation, reward, terminated, truncated, info = env.step(action)

    summed_reward += reward

    if terminated or truncated:
        print("summed reward =", summed_reward)
        summed_reward = 0
        observation, info = env.reset()

# saving csv
if args.save_as_csv:
    with open("retain/decision_data" + model_path[7:-4] + ".csv", "w") as file:
        file.write(csv_data)

env.close()