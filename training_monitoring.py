import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec
import os

def format_time(t):
    if pd.isna(t) or t == 0: return "-"
    minutes = int(t // 60)
    seconds = int(t % 60)
    millis  = int((t - int(t)) * 1000)
    return f"{minutes:02}:{seconds:02}.{millis:03}"

def animate(i):
    plt.clf()
    gs = gridspec.GridSpec(1,2,width_ratios=[2,1])
    # Gráfico de progreso 
    ax1 = plt.subplot(gs[0])
    if os.path.exists("data/logs/progress_log.csv"):
        data = pd.read_csv("data/logs/progress_log.csv")
        if len(data) > 1:
            x = data["timesteps"]
            y = data["progress"]
            std = data["std"]

            ax1.cla() # Limpiar eje
            ax1.errorbar(x, y, yerr=std, label="Progreso acumulado (media)", color="#1f77b4", linewidth=2, errorevery=2, capsize=3, elinewidth=1, alpha=0.8)
            ax1.fill_between(x, y, color="#1f77b4", alpha=0.1)
            
            ax1.set_title("Aprendizaje en Tiempo Real", fontsize=12, fontweight="bold")
            ax1.set_xlabel("Timesteps")
            ax1.set_ylabel("Progreso (dist)")
            ax1.legend(loc="upper left")
            ax1.grid(True, linestyle='--', alpha=0.5)
    # Tabla tiempos
    ax2 = plt.subplot(gs[1])
    ax2.axis("off")
    if os.path.exists("data/logs/best_laps.csv"):
        try:
            laps_df = pd.read_csv("data/logs/best_laps.csv")
            if not laps_df.empty:
                best_lap_row = laps_df.loc[laps_df["lap_time"].idxmin()]
                tb_data = [
                    ["MÉTRICA", "TIEMPO"],
                    ["BEST LAP", format_time(best_lap_row['lap_time'])],
                    ["", ""], # Separador
                    ["Sector 1", format_time(best_lap_row.get('s1', 0))],
                    ["Sector 2", format_time(best_lap_row.get('s2', 0))],
                    ["Sector 3", format_time(best_lap_row.get('s3', 0))],
                    ["", ""],
                    ["Last Step", int(best_lap_row['timestamp'])]
                ]

            table = ax2.table(callText=tb_data, loc="center", cellLoc="center", colWidths=[0.5, 0.5])
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1,2)

            # colores
            table[(1,1)].set_facecolor("#77e992")
            ax2.set_title("RÉCORDS DE VUELTA", fontsize=11, fontweight="bold", color="darkred")
        except Exception as e:
            ax2.text(0.5,0.5,f"Error cargando tabla", ha="center")
    else:
        ax2.text(0.5, 0.5, "Esperando datos\nde vueltas...", ha='center', alpha=0.5)
    plt.tight_layout()


fig = plt.figure(figsize=(14, 6))
ani = FuncAnimation(fig, animate, interval=5000) # 10s (0.1 fps)

print("Monitor dual activado (Gráfico + Tabla).")
plt.show()