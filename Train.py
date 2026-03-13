import os 
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from env import TrackEnv, TrackEnv2
import ctypes

# Callbacks 
class CurriculumCallBack(BaseCallback):
    def __init__(self, total_steps, verbose=0):
        super(CurriculumCallBack, self).__init__(verbose)
        self.total_steps = total_steps

    def _on_step(self) -> bool:
        progress = self.num_timesteps / self.total_steps # progreso [0,1]
        newprob = max(0.0, 0.8 - progress)  # va reduciendo la proabilidad
        self.training_env.env_method("set_random_spawn", newprob)
        return True

class ProgressLoggerCallback(BaseCallback):
    def __init__(self, check_freq:int, log_path:str, best_lap_path=None, verbose=1):
        super().__init__()
        self.check_freq = check_freq
        self.log_path = log_path
        self.best_lap_path = best_lap_path
        self.best_time = float("inf")
        self.data = []

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            all_progress = self.training_env.get_attr("total_ep_prog")
            self.data.append({"timesteps": self.num_timesteps, "progress": np.mean(all_progress), "std": np.std(all_progress)})
            pd.DataFrame(self.data).to_csv(self.log_path, index=False)
        
        # Si estamos en Fase 2, registrar laps
        if self.best_lap_path:
            for info in self.locals.get("infos", []):
                if info.get("is_lap_completed"):
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
        return True

class BestLapCheckpointCallback(BaseCallback):
    def __init__(self, best_lap_path:str, save_dir:str = "data/models/", check_freq:int = 6007, verbose=1):
        super().__init__(verbose)
        self.best_lap_path = best_lap_path
        self.save_dir = save_dir
        self.check_freq = check_freq
        self.last_best_time = float('inf')
        os.makedirs(save_dir, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            if os.path.exists(self.best_lap_path):
                try:
                    with open(self.best_lap_path, 'r', newline='') as f:
                        f.seek(0, os.SEEK_END)
                        f.seek(max(0, f.tell() - 1024), os.SEEK_SET)
                        lines = f.readlines()
                        
                        if len(lines) > 1:
                            last_line = lines[-1].strip().split(',')
                            current_lap_time = float(last_line[0])
                            actual_step = int(last_line[4])

                            if current_lap_time < self.last_best_time:
                                self.last_best_time = current_lap_time
                                save_name = os.path.join(self.save_dir, f"ppo_track_v5_{actual_step}_steps")
                                self.model.save(save_name)

                except (Exception, IndexError, ValueError):
                    pass  # Si otro CallBack lo esta usando
        return True

# Pipeline

def make_env1(): return TrackEnv(track_mask="assets/track_1-mask.png")
def make_env2(): return TrackEnv2(track_mask="assets/track_1-mask.png")

def training(init_model:str = None, both:bool = True, 
             models_path:str = "data/models", logs_path = "data/logs",
             n_cpu:int = 4, ts1:int = 1000000, ts2:int = 1000000, 
             lr1:int = 0.0003, lr2:int = 0.000005, ent_coef = 0.005):   # esto no fue probado para el entrenamiento 1
    """Pipeline de entrenamiento en dos pasos, general y pulido \n
    init_model: ruta (completa) modelo previo de donde partir, both: si es False solo el T2 (refinamiento), \n
    paths: donde se van a guardar los callbacks y resultados, n_cpu: entrenamiento en paralelo, \n
    ts1/2: total timesteps de T1/2, lr1/2: learning ratio T1/2""" 

    model = None
    os.makedirs(logs_path, exist_ok=True)
    os.makedirs(models_path, exist_ok=True)

    best_lap_csv = os.path.join(logs_path, "best_laps.csv")
    progress_logs_csv = os.path.join(logs_path, "progress_log.csv")

    path_t1 = os.path.join(models_path, "ppo_T1")
    path_t2 = os.path.join(models_path, "ppo_T2")

    # Entrenamiento 1 (manejar) #
    if both:
        print(f">>> Train 1: Aprender a manejar | {ts1} steps")   
        env1 = SubprocVecEnv([make_env1 for _ in range(n_cpu)])
        env1 = VecMonitor(env1)

        if init_model:
            model = PPO.load(init_model, env=env1, learning_rate=lr1)
        else:
            model = PPO("MlpPolicy", env1, learning_rate=lr1, verbose=1)

        # callbacks t1
        curriculum_cv = CurriculumCallBack(total_steps = ts1) 
        logger_T1 = ProgressLoggerCallback(2500, progress_logs_csv)
        checkpoint_callback = CheckpointCallback(save_freq=max(ts1 // 10, 50000), save_path=models_path, name_prefix="ppo_t1_chkpt")

        try:
            model.learn(total_timesteps = ts1, callback=[curriculum_cv, logger_T1, checkpoint_callback])  
        except KeyboardInterrupt:
            print("\nEntrenamiento T1 interrumpido")
        finally:
            model.save(path_t1)
            env1.close()
    else:
        print(">>> Salteando T1")

    # Entrenamiento 2 (tiempos de vuelta) #
    print(F">>> Train 2: Mejorar los tiempos de vuelta | {ts2} steps")
    env2 = SubprocVecEnv([make_env2 for _ in range(n_cpu)])
    env2 = VecMonitor(env2)

    if both:  # cargar el resultado de 1
        model = PPO.load(path_t1, env=env2)
    else: 
        load_path = init_model if init_model else (path_t1 + ".zip")  # agarrar el inicial o un T1 previo
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"No se encontró el modelo base en {load_path}")
        model = PPO.load(load_path, env=env2)

    # hiperparametros
    model.learning_rate = lr2 # pulido  (0.0001 o 0.00005)
    model.ent_coef = ent_coef # entropia (disperción) (0.01)
    # model.clip_range = 0.1

    #callbacks t2
    logger_T2 = ProgressLoggerCallback(2500, progress_logs_csv, best_lap_csv)
    checkpoint_callback = BestLapCheckpointCallback(best_lap_csv, models_path)

    try:
        model.learn(total_timesteps = ts2, callback = [logger_T2, checkpoint_callback], reset_num_timesteps=False)  # Ojo True
    except KeyboardInterrupt:
        print("\nEntrenamiento T2 interrumpido")
    finally:
        model.save(path_t2)
        env2.close()

    # Notificación
    txt = f"Entrenamiento finalizado.\nModelo final: {path_t2}"
    print(txt)
    ctypes.windll.user32.MessageBoxW(0, txt, "RL-Race Training", 0x40 | 0x1000)



## ENTRENAMIENTO ##

if __name__ == "__main__":
    training(init_model="data/models/ppo_track_v5_1614928_steps.zip", both=False, ts2=2000000)


