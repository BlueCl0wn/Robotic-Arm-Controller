import argparse, time, torch, tqdm
import gymnasium as gym
import gymnasium_robotics
import numpy as np
from collections import deque
from common import get_file_descriptor, splash_screen
from solvers.dqn import optimize_model, initiate_stuff, select_action


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
    params.max_steps = 50
    params.env = ("FetchReachDense-v3", dict(max_episode_steps=params.max_steps, reward_type="dense"))  #
    params.version = "v1"

    # Create instance of Gymnasium environment. All arguments parsed to the parser are automatically parsed to gym.make().
    # To change environment variables do so in this method according to comments.
    gym.register_envs(gymnasium_robotics)
    env = gym.make(params.env[0], **params.env[1])

    def set_hyperparams_fixed() -> None:
        """
        Sets values for parameters on call. To add or change parameter value do so according to comments.

        :return None:
        """
        # Set size of neural network
        params.input_size = sum([i.shape[0] for i in env.observation_space.values()])  # This calculates the size of the
        #                                                                            observation space as a simple int.
        params.output_size = env.action_space.shape[0] if isinstance(env.action_space,
                                                                     gym.spaces.Box) else env.action_space.n
        #print("input_size = ",params.input_size)
        #print("output_size = ",params.output_size)
        params.hidden_layers = [64, 64, 64]

        params.episode_start = 0
        # Training parameters
        params.repetitions = 100
        params.npop = 30

    def set_hyperparams_run() -> None:
        params.episodes = 100_000

        # ------ DQN Params -------
        params.replay_mem_size = 10_000  # Size of the replay memory
        params.BATCH_SIZE = 16  # number of transitions sampled from the replay buffer
        params.GAMMA = 0.5  # discount factor as mentioned in the previous section
        params.EPS_START = 2  # starting value of epsilon
        params.EPS_END = 0.02  # final value of epsilon
        params.EPS_DECAY = 200_000  # controls the rate of exponential decay of epsilon, higher means a slower decay
        params.TAU = 0.005  # update rate of the target network # 0.005 start value
        params.LR = 1e-3  # learning rate of the ``AdamW`` optimizer

        # if GPU is to be used
        params.device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else
            "cpu"
        )
        print("device = ", params.device)

    set_hyperparams_fixed()
    set_hyperparams_run()

    # Logging hyperparameters.
    logger = splash_screen(params)
    logger.flush()

    # Creation of stuff in case no filepath is provided  / resuming is not enabled.
    # TODO: maybe find a better name than stuff lol
    stuff = policy_net, target_net, optimizer, memory = initiate_stuff(params)

    # Stuff in case resuming is enabled.
    if params.resume:
        resume = params.resume
        policy_net, target_net, optimizer, i, params = torch.load(params.resume)
        #policy_net, target_net, optimizer, memory = stuff
        stuff = policy_net, target_net, optimizer, memory
        params.resume = resume
        #params.max_steps = params.max_teps
        params.episode_start = i
        print(f"Resuming training from episode {i} of {params.resume}")
        set_hyperparams_run()

    # Setting hyperparams which are not fixed for a specific network.
    # This needs to be here. Ensures that adjusting hyperparams still affects resumed training.
    set_hyperparams_run()


    # TQDM loading bar stuff
    episodes = tqdm.trange(
        params.episode_start,
        params.episodes + params.episode_start,
        desc="Fitness",
        )

    avg_rewards = deque([], maxlen=200)

    # Training loop
    for i in episodes:

        # Initialize the environment and get its state
        state, info = env.reset()
        state = flatten_dict(state)
        #print("state: ", state, "\n", len(state))
        state = torch.tensor(state, dtype=torch.float32, device=params.device).unsqueeze(0)
        episode_reward_total = 0

        for t in range(params.max_steps + 10):
            time_action_1 = time.time()
            # Select action using epsilon-greedy strategy
            action = select_action(env, state, *stuff, i, params, logger=logger)
            time_action_2 = time.time()

            time_observation_1 = time.time()
            # Interact with the environment
            observation, reward_env, terminated, truncated, _ = env.step(action.tolist())
            observation = flatten_dict(observation)  # flatten observation from dict[np.ndarray] to np.ndarray
            time_observation_2 = time.time()
            reward = torch.tensor([reward_env], device=params.device)
            done = terminated or truncated
            episode_reward_total += reward_env

            if truncated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=params.device).unsqueeze(0)

            time_memory_1 = time.time()
            # Store the transition in memory
            memory.push(state, action, next_state, reward)
            time_memory_2 = time.time()

            # Move to the next state
            state = next_state

            time_optimize_1 = time.time()
            # Perform one step of the optimization (on the policy network)
            GAMMA = 0.99 # discount factor for future rewards
            optimize_model(*stuff, GAMMA, params)
            time_optimize_2 = time.time()

            time_nets_1 = time.time()
            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key] * params.TAU + target_net_state_dict[key] * (
                        1 - params.TAU)
            target_net.load_state_dict(target_net_state_dict)
            time_nets_2 = time.time()

            if False:
                print("time action: ", time_action_1 - time_action_2)
                print("time memory: ", time_memory_1 - time_memory_2)
                print("time optimize: ", time_optimize_1 - time_optimize_2)
                print("time nets: ", time_nets_1 - time_nets_2)
                print("time observation: ", time_observation_1 - time_observation_2)

            # log
            # logger.add_scalar("sigma", params.sigma, i)

            if done:
                break

        # This block saves the model every 100 episodes and stores other values for use in tensorboard.
        if i % 250 == 0:
            # save w to disk
            descrp = get_file_descriptor(params, i)
            # print(stuff, i, params)
            torch.save((policy_net, target_net, optimizer, i, params), descrp)
            logger.add_histogram("policy_net_params", policy_net.get_parameters(), i)
            logger.add_histogram("target_net_params", target_net.get_parameters(), i)
            logger.add_scalar("replay_memory_length", len(memory), i)


        avg_rewards.append(episode_reward_total) # Add episode_reward to deque to compute running average of fitness.

        # This block checks the performance of the model every 10 episodes and saves that value.
        if i % 10 == 0:
            # Change current reference_fitness shown in loading bar.
            episodes.set_description(f"Fitness: {episode_reward_total:.2f}")
            # log fitness
            logger.add_scalar("fitness", episode_reward_total, i)
            logger.add_scalar("fitness_avg", 0 if (len(avg_rewards) == 0) else sum(avg_rewards)/len(avg_rewards) , i)
    env.close()
    pass


if __name__ == '__main__':
    run()
