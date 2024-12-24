import itertools
from typing import Callable
import gymnasium as gym
from tqdm import tqdm
import numpy as np
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from models import Model


def run_once(model: Model, env: gym.Env, max_steps: int, show_observation: bool, show_action: bool) -> tuple:
    """
    Run simulation once for a specific model for specified number of steps.
    Has less Schnickschnack to play with than 'run_once()'

    :param model: Model with which to run the simulation.
    :param env: Environment
    :param max_steps: Maximum allowed number of steps to run the simulation.
    :param show_observation: ? TODO
    :param show_action: ? TODO
    :return: Tuple of type(fitness, max_steps)
    """
    observation, _ = env.reset()
    fitness = 0.0

    for i in range(max_steps):
        action = model.make_decision(observation)
        observation, reward, terminated, truncated, _ = env.step(action)

        if show_observation:
            print(f"Observation: {observation}")
        if show_action:
            print(f"Action: {action}")

        fitness += float(reward)

        if terminated or truncated:
            return fitness, i + 1
    return fitness, max_steps


def run_once_thin(model: Model, env: gym.Env, max_steps: int) -> tuple:
    """
    Run simulation once for a specific model for specified number of steps.
    Has less Schnickschnack to play with than 'run_once()'

    :param model: Model with which to run the simulation.
    :param env: Environment
    :param max_steps: Maximum allowed number of steps to run the simulation.
    :return: Tuple of type(fitness, max_steps)
    """
    observation, _ = env.reset()
    fitness = 0.0

    for i in range(max_steps):
        # Decision step
        action = model.make_decision(observation)
        observation, reward, terminated, truncated, done = env.step(action)

        # Reward shaping
        """
        # Unpack the observation vector
        x, y, v_x, v_y, angle, ang_vel, leg1, leg2 = observation
        
        # Penalize inaction unless vertical velocity is negligible
        # if decision == 0:
        #     if abs(v_y) > 0.20:
        #         reward -= 1
        #     else:
        #         reward += 0.5

        # Reward stabilizing rotation towards zero angle
        reward += (1.0 - abs(angle)) * 0.01

        # Reward reducing velocities (horizontal and vertical)
        reward += max(0, 1.0 - abs(v_x)) * 0.01
        reward += max(0, 1.0 - abs(v_y + 0.5)) * 0.1

        # penaly for hovering
        if abs(v_y) < 0.1:
            reward -= 0.0

        # Reward main engine steering for stabilizing both angle and position
        if (angle > 0 and x > 0 and decision == 2) or \
                (angle < 0 and x < 0 and decision == 2):
            reward += 0.05

        # Heavlily penalize going out of 2.0 rage off the center
        if abs(x) > 1.25:
            reward -= 1

        if abs(x) > 2.0:
            reward -= 3

        # Heavily reward achieving stability across all key metrics
        reward += max(0, 1.0 - abs(x)) * 0.05
        reward += max(0, 1.0 - abs(v_y)) * 0.05
        reward += max(0, 1.0 - abs(v_x)) * 0.05
        reward += max(0, 1.0 - abs(ang_vel)) * 0.05
        """

        fitness += reward

        if terminated or truncated:
            return fitness, i + 1

    return fitness, max_steps # TODO is it smarter to return actual number of steps here?


def run_once_thin_wrapper(args):
    return run_once_thin(*args)


def run_batch(args):
    model, env, max_steps, batch_size = args

    return [run_once_thin(model, env, max_steps) for _ in range(batch_size)]


# create executor
executor = ProcessPoolExecutor(max_workers=32)


def run_simulation(models: list[Model] | Model,
                   env: str | tuple[str, dict],
                   max_steps: int,
                   repetitions: int = 1,
                   batch_size=100,
                   render: bool = False,
                   show_observation: bool = False,
                   show_action: bool = False,
                   progress_bar: bool = True,
                   make_env: Callable[[], gym.Env] | None = None,
                   ) -> tuple[np.ndarray, np.ndarray]:
    """
    Run_simulation 'repetitions' times for parsed models.

    :param models: List of models for which the simulation is to be run.
    :param env: Name of the environment to run the simulation on.
    :param max_steps: Maximum allowed number of steps in the simulation.
    :param repetitions: Number of times to run the simulation to increase stability of mean.
    :param batch_size: Size of one batch. Important for multiprocessing.
    :param render:
    :param show_observation:
    :param show_action:
    :param progress_bar:
    :param make_env: Function with which to make to environment.
    :return: Returns a tuple of arrays. First array contains
    """
    global executor

    # Allow parsing of single model instead of list[models]
    if not isinstance(models, list):
        models = [models]

    # Separate environment name from running parameters.
    if isinstance(env, tuple):
        env, env_options = env
    else:
        env_options = {}

    # Create make_env function in case it is not parsed.
    if make_env is None:
        make_env = lambda: gym.make(env, render_mode=None, **env_options)

    if repetitions == 1:
        render_mode = "human" if render else None
        fitness, lenght = run_once(models[0], gym.make(env, render_mode=render_mode, **env_options), max_steps,
                                   show_observation, show_action)

        return np.array([fitness]), np.array([lenght])

    # Make sure batch_size is a multiple of the number of models.
    if (len(models) * repetitions) % batch_size != 0:
        raise ValueError(
            f"Batch size {batch_size} is not a multiple of the number of models {len(models) * repetitions}")

    batches = repetitions * len(models) // batch_size  # Number of batches

    # multiprocessing stuff
    tasks = [
        (model, make_env(), max_steps, batch_size)
        for model in itertools.islice(itertools.cycle(models), batches)
    ]

    results = tqdm(executor.map(run_batch, tasks), total=batches, disable=not progress_bar)
    fitnesses, lengths = zip(*itertools.chain.from_iterable(results))
    return np.array(fitnesses).reshape(repetitions, len(models)), np.array(lengths).reshape(repetitions, len(models))


if __name__ == "__main__":
    from models import RandomModel
    import time

    models = [RandomModel() for _ in range(500)]  # type: list[Model]

    start_time = time.time()
    fitness, lenghts = run_simulation(models, ("LunarLander-v3", dict(continuous=True)), 150, repetitions=100,
                                      batch_size=50)
    end_time = time.time()

    print(f"Execution time: {end_time - start_time} seconds")

    print(fitness, lenghts)
    print(np.mean(fitness), np.mean(lenghts))

    # model = RandomModel()
    # fitness, lenght = run_simulation([model], "LunarLander-v3", 1000, 1, render=True, show_observation=True, show_action=True)
    # print(fitness, lenght)