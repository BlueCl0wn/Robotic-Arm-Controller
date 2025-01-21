import gymnasium as gym
import gymnasium_robotics
import argparse
import torch
from solvers.dqn import select_action
from src.models import RandomModel
from common import flatten_dict
import numpy as np


def run() -> None:
    """
    This method implements code to test a DQN model on the "FetchReachDense-v4" environment.
    It supports running an existing model or performing random actions.

    Code does not support call from python code but must be run through cmd / bash.
    Script parameters are:
    --resume [URI]: Model is loaded from parsed URI. If parameter is not used, random actions are performed.
                    Retained models are stored in folder "..retained_models".
    --save_as_csv bool : Settings this to True runs the simulation a bunch of times to gather data to create a
                         decision tree. Using this sets 'render_mode' to None.

    IMPORTANT: To run "FetchReachDense-v4" it's necessary to install gymnasium-robotics=1.3.1 directly from GitHub.
               Packege can be found under: https://github.com/Farama-Foundation/Gymnasium-Robotics

    :return:
    """

    # Creating parser and adding arguments
    parser = argparse.ArgumentParser(description='Stochastic Neural Network')
    parser.add_argument("--resume", type=str, default=None,
                        help="Parse URI of a trained model. If left empty a random_model is used for the simulation.")
    parser.add_argument("--save_as_csv", type=bool, default=False, help="save as csv")
    params = parser.parse_args()

    # Creating environment
    gym.register_envs(gymnasium_robotics)
    render_mode = None if params.save_as_csv else "human"
    env = gym.make("FetchReachDense-v4", max_episode_steps=50, reward_type="dense", render_mode=render_mode)
    observation, info = env.reset(seed=42)

    # Loading the model if URI was parsed.
    if params.resume is not None:
        # Load model data
        loaded_data = torch.load(params.resume, weights_only=False) # type: tuple

        resume, save_as_csv = params.resume, params.save_as_csv  # Create variable so info about resuming is not lost during unpacking of tuple
        policy_net, target_net, optimizer, i, params = loaded_data
        params.resume, params.save_as_csv = resume, save_as_csv

        # Creating variable 'stuff' to use during method calls.
        stuff = policy_net, target_net, optimizer, None
        del resume, loaded_data

    # Printing things
    if params.resume is None:
        print(" Simulating with completely random actions ".center(88, "="))
    else:
        print(f" Testing model {params.resume} ".center(88, "="))



    def get_action(_observation: dict[np.ndarray], _i: int) -> np.ndarray:
        """
        Returns the action for a specific observation. Supports random and resuming mode.

        :return:
        """
        _observation = flatten_dict(_observation)
        if params.resume is None:
            return np.random.uniform(low=-1, high=1, size=4)
        else:
            state = torch.tensor(_observation, dtype=torch.float32, device=params.device).unsqueeze(0)

            _action = policy_net(state).squeeze() #select_action(env, state, *stuff, _i, params)
            print(_action)
            return _action.tolist()

    # set duration of runtime loop
    N = 50 * 25

    # Adjust runtime if csv generation is turned on.
    # Create variable to store actions in.
    if params.save_as_csv:
        csv_data = ""
        N = 50 * 200

    # loop
    summed_reward = 0
    for _i in range(N):

        action = get_action(observation, _i)
        # action = np.argmax(action.detach().numpy())
        if params.save_as_csv:
            csv_data += (','.join([str(round(float(i), 2)) for i in observation.to('cpu').detach().numpy()]) + ","
                         + str(action) + "\n")
            print(_i) if _i % 100 == 0 else None

        # Generate next simulation step.
        observation, reward, terminated, truncated, info = env.step(action)

        # Add up total reward per simulation run.
        summed_reward += reward

        # Reset environment, print and reset 'summed_reward'  when a simulation ends.
        if terminated or truncated:
            print("summed reward =", summed_reward)
            summed_reward = 0
            observation, info = env.reset()

    # saving csv
    if params.save_as_csv:
        with open("retain/decision_data" + params.resume[7:-4] + ".csv", "w") as file:
            file.write(csv_data)

    env.close()



if __name__ == '__main__':
    run()