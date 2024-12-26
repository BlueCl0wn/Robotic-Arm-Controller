import numpy as np
import argparse
import gymnasium as gym
import gymnasium_robotics
import tensorboardX
# from git.repo import Repo
import torch
import tqdm


from common import get_file_descriptor, splash_screen
from episode_runner import run_simulation
from models.nn_model import NeuralNetworkModel
from solver.nes_demo import NES, sample_distribution


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
    # turn off wind, make it easier
    params.env = ("FetchPickAndPlace-v3", dict(wind_power=0.1))
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
        # params.[parameter] =

        # Set size of neural network
        params.input_size = env.observation_space.shape[0]
        params.output_size = env.action_space.shape[0] if isinstance(env.action_space,
                                                                     gym.spaces.Box) else env.action_space.n
        params.hidden_layers = [16, 4]
        params.model_penalty = 0.01

        # Training parameters
        params.episode_start = 0
        params.batch_size = 10
        params.repetitions = 100
        params.max_steps = 300
        params.npop = 30
        params.episodes = 50_000

        # Exploration stuff
        params.step_randomness_to_w_small = 100
        params.step_randomness_to_w_big = 2000
        params.sigma_random_small = 0.001
        params.sigma_random_big = 0.02
        params.learning_rate = 0.15
        params.sigma = 1.5


    set_hyperparams()

    # Logging hyperparameters. Should this be after the if for resuming? Not sure rn. TODO?
    logger = splash_screen(params)
    logger.flush()

    # Our main neural network on which all work is done.
    # Currently, this model is reset to after each repetitions and new generation are computed by sampling around it.
    # Not sure if this is the best implementation with our training algorithm (don't even know which one that is).
    w = NeuralNetworkModel(params.input_size, params.output_size, params.hidden_layers) # TODO: implement model
    print(w.get_parameters().shape)

    def get_population():
        """
        Function for computing the first population. Not implemented.
        :return:
        """
        # [w.new_from_parameters(w.get_parameters()) for _ in range(params.npop)]
        raise NotImplementedError

    # Set first population
    population = get_population()

    # Stuff in case resuming is enabled. Resets w to saved model
    if params.resume:
        w = torch.load(params.resume)
        params.eposode_start = int(params.resume.split("_")[-1].split(".")[0])
        print(f"Resuming training from episode {params.eposode_start} of {params.resume}")


    def fitness_function(models : list, i: int) -> list[int]:
        """
        Compute reward functon of parsed models. Implementation not done (TODO).

        :param models: list of models for which to compute reward
        :param i: training step. necessary for logging purposes.
        :return: list of rewards
        """

        fitnesses, lengths = run_simulation(models,  # type: ignore
                                          params.env,
                                          params.max_steps,
                                          repetitions=params.repetitions,
                                          batch_size=params.batch_size,
                                          progress_bar=False,
                                          make_env=make_env,
                                          )

        model_penalties = np.array([model.get_model_penalty() * params.model_penalty for model in models])
        fitnesses -= model_penalties

        # logging stuff
        if i % 10 == 0:
            logger.add_histogram("fitness_hist", fitnesses, i)
            logger.add_histogram("model_penalties", model_penalties, i)

        logger.add_scalar("fitness_mean", fitnesses.mean(), i)
        logger.add_scalar("steps_mean", lengths.mean(), i)
        #  return fitness.mean(axis=0)
        raise NotImplementedError


    # TQDM loading bar stuff
    episodes = tqdm.trange(
        params.eposode_start,
        params.episodes + params.eposode_start,
        desc="Fitness",
    )

    # Training loop
    for i in episodes:

        # This computes new population to check. I am not sure anymore where "sample_distribution" comes from and what it does. TODO?
        w_tries_numpy = sample_distribution(w, population, params.sigma, params.npop)

        # Calculate fitnesses of w_tries_numpy
        fitness = fitness_function(population, i)

        # Do algorithm stuff
        # TODO: implement our algorithm (in director solvers)
        theta, delta = NES(w_tries_numpy, fitness, params.learning_rate, w.get_parameters(), params.npop, params.sigma)

        # Increases exploration by introducing randomness after some number of step.
        # To turn off just set the appropriate params to a really high number. I am to lazy to add an if rn.
        # Maybe it would be better to change the steps to some other condition (e.g. convergence) . TODO?
        if i % params.step_randomness_to_w_big == 0 and i > 1:
            theta += np.random.normal(loc=0, scale=params.sigma_random_big, size=theta.shape)
        elif i % params.step_randomness_to_w_small == 0 and i > 1:
            theta += np.random.normal(loc=0, scale=params.sigma_random_small, size=theta.shape)

        # Set reference model with new values for next episode.
        w.set_parameters(theta)

        # This block check the performance of the refernce model (w) every 10 episodes and saves that value.
        # Quite important to know as we are doing the whole training to get a good reference model.
        if i % 10 == 0:
            reference_fitness, _ = run_simulation([w],  # type: ignore
                                                  params.env,
                                                  params.max_steps,
                                                  repetitions=200,
                                                  batch_size=10,
                                                  progress_bar=False,
                                                  make_env=make_env,
                                                  ) # Compute reference_fitness

            episodes.set_description(f"Fitness: {reference_fitness.mean():.2f}") # Add current reference_fitness to loading bar.

            # log stuff
            logger.add_scalar("reference_fitness", reference_fitness.mean(), i)
            logger.add_histogram("w_delta", delta, i)
            parameters = w.get_parameters()
            logger.add_histogram("w_params", parameters, i)

        # This block saves the model every 100 episodes.
        if i % 100 == 0:
            # save w to disk
            descrp = get_file_descriptor(params, i)
            torch.save(w, descrp)

        # This is block also for exploration. It decreases the sigma and learning_rate of the sampling at the beginning of this loop in every episode.
        # When they are too low, they are reset to a higher value. This results in kind of a chain saw behavior when plotted to episodes.
        # We should discuss if this is a good idea for us. Of course also depends on the training algorithm. TODO?
        params.sigma *= 0.9995
        if params.sigma < 0.5:
            params.sigma = 1.5

        params.learning_rate *= 0.999
        if params.learning_rate < 0.05:
            params.learning_rate = 0.25

        # log
        logger.add_scalar("sigma", params.sigma, i)

    pass


if __name__ == '__main__':
    run()

