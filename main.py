import gymnasium as gym
import gymnasium_robotics
from sb3_contrib import TQC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

gym.register_envs(gymnasium_robotics)

# Create the FetchReach-v3 environment
env = gym.make("FetchReach-v4", render_mode="human")

# Define and train the TQC agent
model = TQC("MultiInputPolicy", 
			env, 
    		policy_kwargs={"net_arch" : [512, 512, 512]},
			learning_starts=1000,
			target_update_interval=10,
            train_freq=(1,"step"),
            gradient_steps=1, 
            verbose=1,
            )

# Evaluate the model before training
print("Evaluation before training:")
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

# Train the agent for 100000 steps
model.learn(total_timesteps=100000)

# Evaluate the trained agent
print("Evaluation after training:")
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

env.close()



