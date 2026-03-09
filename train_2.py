# Modelo 
import numpy as np
import pandas as pd
import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from env import TrackEnv2
import ctypes # fancy

def make_env(rank, seed=0):
    "Crear una copia del entorno por nucleo"
    def _init():
        env = TrackEnv2(track_mask="assets/track_1-mask.png")
        return env
    return _init

class ProgressLoggerCallback(BaseCallback):
    def __init__(self, check_freq:int, log_path:str, best_lap_path:str, verbose=1):
        super(ProgressLoggerCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_path = log_path
        self.best_lap_path = best_lap_path
        self.best_time = float("inf")
        self.data = []

    def _on_step(self) -> bool:
        # Tiempos de vuelta
        for info in self.locals.get("infos", []):
            if info.get("is_lap_complete"):
                lap_time = info["lap_time"]
                if lap_time < self.best_time:
                    self.best_time = lap_time
                    df_best = pd.DataFrame({
                        "lap_time": [lap_time],
                        "s1": [info["sectors"][0] if len(info["sectors"])>0 else 0],
                        "s2": [info["sectors"][1] if len(info["sectors"])>1 else 0],
                        "s3": [info["sectors"][2] if len(info["sectors"])>2 else 0],
                        "timestamp": [self.num_timesteps]})
                    header = not os.path.exists(self.best_lap_path)
                    df_best.to_csv(self.best_lap_path, index=False, header=header, mode="a")

        # Progreso general
        if self.n_calls % self.check_freq == 0:        
            all_progress = self.training_env.get_attr("total_ep_prog")  
            mean_progress = np.mean(all_progress)
            std_progress = np.std(all_progress)
            self.data.append({"timesteps": self.num_timesteps, "progress": mean_progress, "std": std_progress})
            pd.DataFrame(self.data).to_csv(self.log_path, index=False)
      
        return True

def train(model_path = "data/models/ppo_track_v3_800000.zip" ,  total_timesteps = 2200000):
    "Entrenamiento de un modelo PPO"
    # Rutas
    os.makedirs("data/logs", exist_ok=True)
    os.makedirs("data/models", exist_ok=True)
    log_path = "data/logs/progress_log.csv"
    best_lap_path = "data/logs/best_lap.csv"
    # Hardware
    n_cpu = 4
    
    # Entorno
    env = SubprocVecEnv([make_env(i) for i in range(n_cpu)])
    env = VecMonitor(env) # para trackeo
    
    # Callbakcs
    checkpoint_callback = CheckpointCallback(save_freq=50000, save_path="./data/models/", name_prefix="ppo_track_v3")
    progress_callback = ProgressLoggerCallback(check_freq=5000, log_path=log_path, best_lap_path=best_lap_path) 
   
    # Modelo
    if os.path.exists(model_path):
        print(f"Cargando modelo: {model_path}")
        model = PPO.load(model_path, env=env, device="auto")
        model.learning_rate = 0.0001 # mas fino
    else:
        print("Creando nuevo modelo")
        model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003, tensorboard_log="./data/ppo_tensorboard/")

    # Entrenamiento
    print(f"Empezando optimización en paralelo, N#{n_cpu}")
    try: 
        model.learn(total_timesteps=total_timesteps, callback=[progress_callback, checkpoint_callback], reset_num_timesteps=False)  # OJO! CAMBIAR A FALSE/TRUE 
    except KeyboardInterrupt:
        print("\nEntrenamiento interrumpido")

    # Guardado
    model.save("data/models/ppo_track_agent_opt")
    print("Modelo guardado")

    # Aviso
    txt = "Optimización completada.\nModelo guardado en data/models/ppo_track_agent_opt"
    ctypes.windll.user32.MessageBoxW(0, txt, "¡ÉXITO!", 0x40 | 0x1000)

if __name__ == "__main__":
    train()  # multiprocessing requiere el bloque if __name__ == "__main__":