import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os

def animate(i):
    if os.path.exists("data/logs/progress_log.csv"):
        data = pd.read_csv("data/logs/progress_log.csv")
        if len(data) > 1:
            x = data["timesteps"]
            y = data["progress"]
            std = data["std"]

            plt.cla() # Limpiar eje
            plt.errorbar(x, y, yerr=std, label="Progreso acumulado", color="#1f77b4", linewidth=2, errorevery=2, capsize=3, elinewidth=1, alpha=0.8)
            plt.fill_between(x, y, color="#1f77b4", alpha=0.1)
            
            plt.title("Aprendizaje en Tiempo Real", fontsize=14, fontweight="bold")
            plt.xlabel("Timesteps")
            plt.ylabel("Progreso (dist)")
            plt.legend(loc="upper left")
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.tight_layout()

fig = plt.figure(figsize=(10, 5))
ani = FuncAnimation(fig, animate, interval=10000) # 10s (0.1 fps)

print("Iniciando monitor de progreso... Mantener esta ventana abierta.")
plt.show()