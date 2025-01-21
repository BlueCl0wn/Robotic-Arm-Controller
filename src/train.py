import argparse, time, torch, tqdm
import gymnasium as gym
import gymnasium_robotics
import numpy as np
from collections import deque
from common import get_file_descriptor, splash_screen, flatten_dict
from solvers.dqn import optimize_model, initiate_stuff, select_action




def run() -> None:
    """
    Run training of model.

    :return None:
    """

    # Create parser and Namespace
    parser = argparse.ArgumentParser(description='Stochastic Neural Network')
    parser.add_argument("--resume", type=str, default=None, help="Resume training from specified model.")
    args = parser.parse_args()
    params = argparse.Namespace()
    params.__dict__.update(args.__dict__)

    # Environment settings
    params.max_steps = 50
    params.reward_type = "dense"

    # IMPORTANT: To use the version 4 of this environment (and in general the Fetch environments) gymnasium-robotics==1.3.1 must be
    # directly installed from GitHub. For some reason v2 and v4 do not exist in gymnasium-robotics=1.3.1 installed
    # through pip / pypi.
    params.env = ("FetchReachDense-v4", dict(max_episode_steps=params.max_steps, reward_type=params.reward_type))


    # Create instance of Gymnasium environment. All arguments parsed to the parser are automatically parsed to gym.make().
    gym.register_envs(gymnasium_robotics)
    env = gym.make(params.env[0], **params.env[1])

    def set_hyperparams_fixed() -> None:
        """
        Sets values for parameters on call. To add or change parameter value do so according to comments.
        The hparams set in this function are fixed for a model. Meaning these can not be changed when resuming.
        In practice, they are overridden during loading of resume model.

        :return None:
        """
        # Set size of neural network
        params.input_size = sum([i.shape[0] for i in env.observation_space.values()])  # This calculates the size of the
        #                                                                            observation space as a simple int.
        params.output_size = env.action_space.shape[0] if isinstance(env.action_space,
                                                                     gym.spaces.Box) else env.action_space.n

        params.hidden_layers = [128, 512, 128]

        params.episode_start = 0
        # Training parameters

    def set_hyperparams_run() -> None:
        """
        Sets values for parameters on call. To add or change parameter value do so according to comments.
        The hparams set in this function can be changed when resuming training of an existing model.

        :return:
        """
        params.episodes = 1_000_002

        # ------ DQN Params -------
        params.replay_mem_size = 100_000  # Size of the replay memory
        params.BATCH_SIZE = 30  # number of transitions sampled from the replay buffer
        params.GAMMA = 0.95  # discount factor as mentioned in the previous section
        params.EPS_START = 1  # starting value of epsilon
        params.EPS_END = 0.01  # final value of epsilon
        params.EPS_DECAY = 100_000  # controls the rate of exponential decay of epsilon, higher means a slower decay
        params.TAU = 0.01  # update rate of the target network # 0.005 start value
        params.LR = 1e-7  # learning rate of the ``AdamW`` optimizer

        # if GPU is to be used
        params.device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else
            "cpu"
        )
        params.device = torch.device("cpu")
        print("device = ", params.device)

    # set hyperparams
    set_hyperparams_fixed()
    set_hyperparams_run()

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

    # Setting hyperparams which are not fixed for a specific network again.
    # This needs to be here. Ensures that adjusting hyperparams still affects resumed training.
    set_hyperparams_run()

    # Logging hyperparameters.
    logger = splash_screen(params)
    logger.flush()

    # TQDM loading bar
    episodes = tqdm.trange(
        params.episode_start,
        params.episodes + params.episode_start,
        desc="Fitness",
        )

    # initiate deque used to log average fitness over a
    avg_rewards_200 = deque([], maxlen=200)
    avg_rewards_1000 = deque([], maxlen=1000)
    avg_rewards_5000 = deque([], maxlen=5000)
    avg_slope_5000 = deque([], maxlen=5000)

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
            action = select_action(env, state, *stuff, i, params, logger=logger)
            time_action_2 = time.time()
            time_observation_1 = time.time()
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
            optimize_model(*stuff, params)
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

            # Change to True to get timing data.
            if False:
                print("time action: ", time_action_1 - time_action_2)
                print("time memory: ", time_memory_1 - time_memory_2)
                print("time optimize: ", time_optimize_1 - time_optimize_2)
                print("time nets: ", time_nets_1 - time_nets_2)
                print("time observation: ", time_observation_1 - time_observation_2)

            if done:
                break

        # This block saves the model every 500 episodes and stores other values for use in tensorboard.
        if i % 500 == 0:
            # Create file descriptor
            descrp = get_file_descriptor(params, i)

            # Save all important parts of model
            torch.save((policy_net, target_net, optimizer, i, params), descrp)

            # Logging things
            logger.add_histogram("policy_net_params", policy_net.get_parameters(), i)
            logger.add_histogram("target_net_params", target_net.get_parameters(), i)
            logger.add_scalar("replay_memory_length", len(memory), i)


        # Creating deques to average the performance for logging purposes.
        avg_rewards_200.append(episode_reward_total) # Add episode_reward to deque to compute running average of fitness.
        avg_rewards_1000.append(episode_reward_total) # Add episode_reward to deque to compute running average of fitness.
        avg_rewards_5000.append(episode_reward_total) # Add episode_reward to deque to compute running average of fitness.
        avg_slope_5000.append(0 if (len(avg_rewards_5000) == 0) else sum(avg_rewards_5000)/len(avg_rewards_5000)) # Add episode_reward to deque to compute running average of fitness.

        # This block checks the performance of the model every 10 episodes and saves that value.
        if i % 10 == 0:
            # Change current reference_fitness shown in loading bar.
            episodes.set_description(f"Fitness_average: {0 if (len(avg_rewards_5000[-200:]) == 0) else sum(avg_rewards_5000[-200:])/len(avg_rewards_5000[-200:]):.2f}")
            # log fitness
            logger.add_scalar("fitness_avg_200",
                              0 if (len(avg_rewards_5000[-200:]) == 0) else sum(avg_rewards_5000[-200:])/len(avg_rewards_5000[-200:]),
                              i)
            logger.add_scalar("fitness_avg_1000",
                              0 if (len(avg_rewards_5000[-1000:]) == 0) else sum(avg_rewards_5000[-1000:])/len(avg_rewards_1500[-1000:]),
                              i)
            logger.add_scalar("fitness_avg_5000",
                              0 if (len(avg_rewards_5000) == 0) else sum(avg_rewards_5000)/len(avg_rewards_5000),
                              i)

            logger.add_scalar("fitness_slope_5000", (avg_slope_5000[-1] - avg_slope_5000[0])/min(5000, len(avg_slope_5000)) , i)
    env.close()
    pass


if __name__ == '__main__':
    run()
