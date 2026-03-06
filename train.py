# Modelo 

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from env import TrackEnv
import pandas as pd
import os

import ctypes # fancy

class ProgressLoggerCallback(BaseCallback):
    def __init__(self, check_freq:int, log_path:str, verbose=1):
        super(ProgressLoggerCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_path = log_path
        self.data = []

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:        
            current_progress = self.training_env.get_attr("total_ep_prog")[0]  
            self.data.append({"timesteps": self.num_timesteps, "progress": current_progress})
            pd.DataFrame(self.data).to_csv(self.log_path, index=False)
        return True

checkpoint_callback = CheckpointCallback(save_freq=100000, save_path='./data/models/', name_prefix='ppo_track_v1')

env = TrackEnv(track_mask="assets/track_1-mask.png")
os.makedirs("data/logs", exist_ok=True)

callback = ProgressLoggerCallback(check_freq=1000, log_path="data/logs/progress_log.csv")

# Modelo - MlpPolicy (red neuronal)

model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003, device="auto", tensorboard_log="./data/ppo_tensorboard/")
# load_and_train.py
model = PPO.load("data/models/ppo_track_v1_100000_steps", env=env) 

# Entrenamiento
print("Empezando entrenamiento")
# model.learn(total_timesteps=3000000, callback=[callback, checkpoint_callback])
model.learn(total_timesteps=3000000, callback=[callback, checkpoint_callback], reset_num_timesteps=False) # continue

# guardar
model.save("data/models/ppo_track_agent")
# model.save("data/models/ppo_track_agent_v2")
print("Modelo guardado")

def mensaje_final():
    titulo = "¡ENTRENAMIENTO COMPLETADO!"
    texto = "El modelo terminó el entrenamiento.\n\nEl modelo ha sido guardado en data/ppo_track_agent.\nYa se puede cerrar la consola y correr enjoy.py."
    ctypes.windll.user32.MessageBoxW(0, texto, titulo, 0x40 | 0x1000)

mensaje_final()
