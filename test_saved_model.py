import gymnasium as gym
import gymnasium_robotics
from sb3_contrib import TQC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

gym.register_envs(gymnasium_robotics)

# Create the FetchReach-v3 environment
env = gym.make("FetchReach-v4", render_mode="human")

# Load the trained model (optional)
loaded_model = TQC.load("tqc_fetchreach_150k", env=env)
print("Loaded the saved model.")

# Evaluate the trained agent
print("Evaluation after training:")
mean_reward, std_reward = evaluate_policy(loaded_model, env, n_eval_episodes=30)
print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

env.close()