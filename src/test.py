import gymnasium as gym
import gymnasium_robotics
import argparse
import torch
from solvers.dqn import select_action
from src.models import RandomModel
from train import flatten_dict

# Creating environment
gym.register_envs(gymnasium_robotics)
env = gym.make("FetchReachDense-v3", max_episode_steps=50, reward_type="dense", render_mode="human")
observation, info = env.reset(seed=42)

# Creating parser and adding arguments
parser = argparse.ArgumentParser(description='Stochastic Neural Network')
parser.add_argument("--resume", type=str, default=None, help="Parse URI of a trained model. If left empty a random_model is used for the simulation.")
parser.add_argument("--save_as_csv", type=bool, default=False, help="save as csv")
args = parser.parse_args()
model_path = args.resume

# Printing things
if model_path is None:
    print("Simulating with completely random actions")
else:
    print("Testing model", model_path)

# Creating or loading the model depending on parsed model_path
if model_path is None:
    model = RandomModel()
else:
    loaded_data = torch.load(model_path, weights_only=False)
    policy_net, target_net, optimizer, i, params = loaded_data
    stuff = policy_net, target_net, optimizer, None

def get_action(_observation, _i):
    """
    Returns the action for a specific observation.
    :return:
    """
    _observation = flatten_dict(_observation)
    if model_path is None:
        return model.make_decision(4)
    else:
        state = torch.tensor(_observation, dtype=torch.float32, device=params.device).unsqueeze(0)
        _action = select_action(env, state, *stuff, _i, params)
        return _action.tolist()

# set duration of runtime loop
N = 10000

# stuff for saving observations and actions in csv
# Handy in order to generate a decision tree.
if args.save_as_csv:
    csv_data = ""
    N = 150 * 100

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