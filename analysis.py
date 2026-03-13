import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

def format_time(t):
    minutes = int(t // 60)
    seconds = int(t % 60)
    millis  = int((t - int(t)) * 1000)
    return f"{minutes:02}:{seconds:02}.{millis:03}"

def get_best_rollout(model_path, env, n_laps=5):
    "Corre el modelo y devuelve la telemetria de la vuelta mas rápida"
    model = PPO.load(model_path)
    best_telemetry = None
    best_time = float("inf")

    for _ in range(n_laps):
        obs,_ = env.reset()
        lap_data = []
        done = False
        while not done:
            action,_ = model.predict(obs, deterministic=False) # son malos deterministas
            obs. reward, terminated, truncated, info = env.step(action)

            lap_data.append({"x": env.car.position.x, "y": env.car.position.y,
                             "speed": env.car.velocity.lenght(),
                             "progress": env.track.get_progress(env.car.position.x, env.car.position.y),
                             "action": action})
            if info.get("is_lap_completed"):
                if info["lap_time"] < best_time:
                    best_time = info["lap_time"]
                    best_telemetry = pd.DataFrame(lap_data)
                done = True
            if terminated or truncated: done = True

    return best_telemetry, best_time


def plot_learning_curve(t1_path, t2_path):
    df1 = pd.read_csv(t1_path)
    df2 = pd.read_csv(t2_path)
    
    offset = df1['timesteps'].max()
    df2['timesteps'] += offset
    full_df = pd.concat([df1, df2])
    
    plt.figure(figsize=(12, 5))
    plt.fill_between(full_df["timesteps"], full_df["progress"]-full_df["std"], full_df["progress"]+full_df["std"], alpha=0.2, color="blue")
    plt.plot(full_df["timesteps"], full_df['progress'], color="darkblue", label="Media de Progreso")
    plt.axvline(x=offset, color="black", linestyle="--", label="Inicio T2")
    plt.ylabel("Distancia Recorrida [px]")
    plt.title("Evolución del Aprendizaje: De Exploración (T1) a Optimización (T2)")
    plt.legend()


def plot_speed_profile(telemetry_df):
    plt.figure(figsize=(12, 4))
    plt.plot(telemetry_df["progress"], telemetry_df["speed"], color="blue", label="Perfil de Velocidad")
    
    brakes = telemetry_df[telemetry_df["action"].apply(lambda x: x[4] > 0.5)]
    plt.plot(brakes["progress"], brakes["speed"], color="red", s=2, label="Frenado")
    
    plt.xlabel("Progreso en pista (m)")
    plt.ylabel("Velocidad (px/s)")
    plt.title("Análisis de Carga Dinámica")
    plt.legend()

