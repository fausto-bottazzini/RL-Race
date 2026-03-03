# Modelo 

from stable_baselines3 import PPO
from env import TrackEnv
import os

env = TrackEnv(track_mask="assets/track_1-mask.png")

# MlpPolicy (red neuronal)
model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003, device="auto")

# Entrenamiento
print("Empezando entrenamiento")
model.learn(total_timesteps=500000)

# guardar
model.save("data/ppo_track_agent")
print("Modelo guardado")
