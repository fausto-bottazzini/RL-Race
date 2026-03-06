import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os

def animate(i):
    if os.path.exists("data/logs/progress_log.csv"):
        data = pd.read_csv("data/logs/progress_log.csv")
        if len(data) > 1:
            x = data['timesteps']
            y = data['progress']
            
            plt.cla() # Limpiar eje
            plt.plot(x, y, label='Progreso acumulado', color='#1f77b4', linewidth=2)
            plt.fill_between(x, y, color="#1f77b4", alpha=0.1)
            
            plt.title("Aprendizaje en Tiempo Real", fontsize=14, fontweight='bold')
            plt.xlabel("Timesteps")
            plt.ylabel("Progreso (dist)")
            plt.legend(loc='upper left')
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()

fig = plt.figure(figsize=(10, 5))
ani = FuncAnimation(fig, animate, interval=5000) # 5s (0.2 fps)

print("Iniciando monitor de progreso... Mantener esta ventana abierta.")
plt.show()