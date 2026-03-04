# Modelo 

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from env import TrackEnv
import pandas as pd
import os

class ProgressLoggerCallback(BaseCallback):
    def __init__(self, check_freq:int, log_path:str, verbose=1):
        super(ProgressLoggerCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_path = log_path
        self.data = []

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            current_progress = self.training_env.get_attr("last_progress")[0]
            self.data.append({"timesteps": self.num_timesteps, "progress": current_progress})
            pd.DataFrame(self.data).to_csv(self.log_path, index=False)
        return True

env = TrackEnv(track_mask="assets/track_1-mask.png")
os.makedirs("data/logs", exist_ok=True)

callback = ProgressLoggerCallback(check_freq=1000, log_path="data/logs/progress_log.csv")

# MlpPolicy (red neuronal)
model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.003, ent_coef= 0.01, device="auto")

# Entrenamiento
print("Empezando entrenamiento")
model.learn(total_timesteps=1000000, callback=callback)

# guardar
model.save("data/ppo_track_agent")
print("Modelo guardado")
