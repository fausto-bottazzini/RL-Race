# Modelo 
import pandas as pd
import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from env import TrackEnv
import ctypes # fancy

def make_env(rank, seed=0):
    "Crear una copia del entorno por nucleo"
    def _init():
        env = TrackEnv(track_mask="assets/track_1-mask.png")
        return env
    return _init

class ProgressLoggerCallback(BaseCallback):
    def __init__(self, check_freq:int, log_path:str, verbose=1):
        super(ProgressLoggerCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_path = log_path
        self.data = []

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:        
            all_progress = self.training_env.get_attr("total_ep_prog")  
            avg_progress = sum(all_progress) / len(all_progress)
            self.data.append({"timesteps": self.num_timesteps, "progress": avg_progress})
            pd.DataFrame(self.data).to_csv(self.log_path, index=False)
        return True

def train(model_path = "data/models/ppo_track_v1_400000_steps.zip" ,  total_timesteps = 3000000):
    "Entrenamiento de un modelo PPO"
    # Rutas
    os.makedirs("data/logs", exist_ok=True)
    os.makedirs("data/models", exist_ok=True)
    log_path = "data/logs/progress_log.csv"
    
    # Hardware
    n_cpu = 4
    
    # Entorno
    env = SubprocVecEnv([make_env(i) for i in range(n_cpu)])
    env = VecMonitor(env) # para trackeo
    
    # Callbakcs
    checkpoint_callback = CheckpointCallback(save_freq=25000, save_path='./data/models/', name_prefix='ppo_track_v2')
    progress_callback = ProgressLoggerCallback(check_freq=2500, log_path=log_path) 
   
    # Modelo
    if os.path.exists(model_path):
        print(f"Cargando modelo: {model_path}")
        model = PPO.load(model_path, env=env, device="auto")
    else:
        print("Creando nuevo modelo")
        model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0005, tensorboard_log="./data/ppo_tensorboard/")

    # Entrenamiento
    print(f"Empezando entrenamiento en paralelo, N#{n_cpu}")
    try: 
        model.learn(total_timesteps=total_timesteps, callback=[progress_callback, checkpoint_callback], reset_num_timesteps=False)
    except KeyboardInterrupt:
        print("Entrenamiento interrumpido")

    # Guardado
    model.save("data/models/ppo_track_agent")
    print("Modelo guardado")

    # Aviso
    txt = "Entrenamiento completado.\nModelo guardado en data/models/ppo_track_agent"
    ctypes.windll.user32.MessageBoxW(0, txt, "¡ÉXITO!", 0x40 | 0x1000)

if __name__ == "__main__":
    train()  # multiprocessing requiere el bloque if __name__ == "__main__":