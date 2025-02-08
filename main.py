import gymnasium as gym
import gymnasium_robotics
import matplotlib.pyplot as plt
import numpy as np
from sb3_contrib import TQC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback

gym.register_envs(gymnasium_robotics)

# Create the FetchReach-v3 environment
env = gym.make("FetchReach-v4", render_mode="human")

# Custom callback to store rewards
class RewardLogger(BaseCallback):
    def __init__(self, check_freq: int = 1000):
        super().__init__()
        self.check_freq = check_freq
        self.rewards = []
        self.timesteps = []

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            mean_reward, _ = evaluate_policy(self.model, self.training_env, n_eval_episodes=10)
            self.rewards.append(mean_reward)
            self.timesteps.append(self.num_timesteps)
        return True

# Initialize model
model = TQC("MultiInputPolicy", 
            env,
            policy_kwargs={"net_arch": [512, 512, 512]},
            learning_starts=1000,
            target_update_interval=10,
            train_freq=(1, "step"),
            gradient_steps=1,
            verbose=1)

# Evaluate before training
print("Evaluation before training:")
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

# Train with logging callback
reward_logger = RewardLogger(check_freq=5000)
model.learn(total_timesteps=150000, callback=reward_logger)

# Save the trained model
model.save("tqc_fetchreach_150k")
print("Model saved as tqc_fetchreach_150k.zip")

# Load the trained model
loaded_model = TQC.load("tqc_fetchreach_150k", env=env)
print("Loaded the saved model.")

# Evaluate after training
print("Evaluation after training:")
mean_reward, std_reward = evaluate_policy(loaded_model, env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

# Plot learning curve
plt.plot(reward_logger.timesteps, reward_logger.rewards, marker='o', linestyle='-')
plt.xlabel("Timesteps")
plt.ylabel("Mean Reward")
plt.title("Training Progress")
plt.grid()
plt.savefig("training_progress.png")
plt.show()

env.close()
