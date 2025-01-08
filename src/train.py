import argparse

import gymnasium as gym
import gymnasium_robotics
import torch
import tqdm
import numpy as np

from itertools import count
from common import get_file_descriptor, splash_screen
from solvers.dqn import optimize_model, initiate_stuff, select_action


# from solvers.nes_demo import NES, sample_distribution

def flatten_dict(d: dict[np.ndarray]) -> np.ndarray:
    """
    Flatten a dictionary of NumPy ndarrays into a single 1D NumPy ndarray.

    This function takes a dictionary where the values are NumPy ndarrays,
    flattens each array, and concatenates them into a single 1D ndarray.

    Parameters:
    -----------
    d : dict
        A dictionary where the values are NumPy ndarrays. The keys can be
        of any hashable type. The arrays can have any shape or dimension.

    Returns:
    --------
    numpy.ndarray
        A 1D NumPy ndarray containing all elements from the input arrays,
        concatenated in the order they appear in the dictionary.

    Examples:
    ---------
    >> import numpy as np
    >> d = {'a': np.array([[1, 2], [3, 4]]),
    ...      'b': np.array([5, 6, 7]),
    ...      'c': np.array([[8], [9], [10]])}
    >> flatten_dict_of_ndarrays(d)
    array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10])

    Notes:
    ------
    - The order of elements in the output array depends on the order of
      items in the input dictionary and the order of elements in each array.
    - This function uses numpy.concatenate(), which is efficient for
      combining multiple arrays.
    """
    return np.concatenate([arr.flatten() for arr in d.values()])

def run() -> None:
    """
    Run training of model.

    :return None:
    """
    # repo = Repo(search_parent_directories=True)

    parser = argparse.ArgumentParser(description='Stochastic Neural Network')

    parser.add_argument("--resume", type=str, default=None, help="Resume training from specified model.")

    args = parser.parse_args()

    params = argparse.Namespace()

    params.__dict__.update(args.__dict__)
    # Environment settings
    params.max_teps = 200
    params.env = ("FetchReachDense-v3" , dict(max_episode_steps=params.max_teps,  reward_type="dense")) #
    params.version = "v1"
    #params.commit = repo.head.commit.hexsha


    def make_env() -> gym.Env:
        """
        Create instance of Gymnasium environment. All arguments parsed to the parser are automatically parsed to gym.make().
        To change environment variables do so in this method according to comments.

        :return None:
        """
        gym.register_envs(gymnasium_robotics)
        instance = gym.make(params.env[0], **params.env[1])

        # Change environment variables:
        # instance.unwrapped.[variable_name] = [value]

        return instance
    env = make_env()

    def set_hyperparams() -> None:
        """
        Sets values for parameters on call. To add or change parameter value do so according to comments.

        :return None:
        """
        # Set size of neural network
        params.input_size = sum([i.shape[0] for i in env.observation_space.values()]) # This calculates the size of the
        #                                                                            observation space as a simple int.
        params.output_size = env.action_space.shape[0] if isinstance(env.action_space,
                                                                     gym.spaces.Box) else env.action_space.n
        #print("input_size = ",params.input_size)
        #print("output_size = ",params.output_size)
        params.hidden_layers = [64, 64]

        # Training parameters
        params.episode_start = 0
        params.repetitions = 100
        params.npop = 30
        params.episodes = 50_000

        # if GPU is to be used TODO This might be nice to implement because rn this is slow as fuck.
        params.device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else
            "cpu"
        )
        print("device = ", params.device)

        # ------ DQN Params -------
        params.BATCH_SIZE = 128  # number of transitions sampled from the replay buffer
        params.GAMMA = 0.99  # discount factor as mentioned in the previous section
        params.EPS_START = 0.9  # starting value of epsilon
        params.EPS_END = 0.05  # final value of epsilon
        params.EPS_DECAY = 1000  # controls the rate of exponential decay of epsilon, higher means a slower decay
        params.TAU = 0.005  # update rate of the target network
        params.LR = 1e-4  # learning rate of the ``AdamW`` optimizer
    set_hyperparams()

    # Logging hyperparameters.
    logger = splash_screen(params)
    logger.flush()

    # initiate stuff for DQN TODO: maybe find a better name than stuff lol
    stuff = policy_net, target_net, optimizer, memory = initiate_stuff(params)

    # Stuff in case resuming is enabled.
    if params.resume:
        resume = params.resume
        stuff, i, params = torch.load(params.resume)
        params.resume = resume
        params.episode_start = i
        print(f"Resuming training from episode {i} of {params.resume}")

    # TQDM loading bar stuff
    episodes = tqdm.trange(
        params.episode_start,
        params.episodes + params.episode_start,
        desc="Fitness",
    )

    # Training loop
    for i in episodes:

        # Initialize the environment and get its state
        state, info = env.reset()
        state = flatten_dict(state)
        #print("state: ", state, "\n", len(state))
        state = torch.tensor(state, dtype=torch.float32, device=params.device).unsqueeze(0)
        episode_reward_total = 0
        for t in range(params.max_teps+10):
            #      Seems more useful to me as this way there is no step_limit to the simulation
            action = select_action(env, state, *stuff, t, params, logger)
            # print("action: ", action)
            observation, reward_env, terminated, truncated, _ = env.step(action.tolist())
            observation = flatten_dict(observation)  # flatten observation from dict[np.ndarray] to np.ndarray
            reward = torch.tensor([reward_env], device=params.device)
            done = terminated or truncated
            episode_reward_total += reward_env


            if truncated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=params.device).unsqueeze(0)

            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            optimize_model(*stuff, params)

            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key] * params.TAU + target_net_state_dict[key] * (
                        1 - params.TAU)
            target_net.load_state_dict(target_net_state_dict)

            # log
            # logger.add_scalar("sigma", params.sigma, i)

            if done:
                break

        # This block saves the model every 100 episodes.
        if i % 100 == 0:
            # save w to disk
            descrp = get_file_descriptor(params, i)
            torch.save((stuff, i, params), descrp)

        # This block checks the performance of the model every 10 episodes and saves that value.
        if i % 10 == 0:
            # Change current reference_fitness shown in loading bar.
            episodes.set_description(f"Fitness: {episode_reward_total:.2f}")
            # log fitness
            logger.add_scalar("fitness", episode_reward_total, i)
        if i % 50 == 0:
            logger.add_histogram("policy_net_params", policy_net.get_parameters(), i)
            logger.add_histogram("target_net_params", target_net.get_parameters(), i)
    env.close()
    pass

if __name__ == '__main__':
    run()

