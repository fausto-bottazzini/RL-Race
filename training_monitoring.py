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
            ax1.axhline(3114.2294737798984, color = "red", linestyle = "dashed", alpha = 0.25, label = "Total Length")
            ax1.errorbar(x, y, yerr=std, label="Progreso acumulado (media)", color="#1f77b4", linewidth=2, elinewidth=1, capsize=3, errorevery=2, alpha=0.8)
            ax1.fill_between(x, y, color="#1f77b4", alpha=0.1)
            
            ax1.set_title("Aprendizaje en Tiempo Real", fontsize=12, fontweight="bold")
            ax1.set_xlabel("Timesteps")
            ax1.set_ylabel("Progreso (dist)")
            ax1.legend(loc="upper left")
            ax1.grid(True, linestyle='--', alpha=0.5)
    # Tabla tiempos
    ax2 = plt.subplot(gs[1])
    ax2.axis('off')

    if os.path.exists("data/logs/best_laps.csv"):
        try:
            laps_df = pd.read_csv("data/logs/best_lap.csv")
            if not laps_df.empty:
                b_s1 = laps_df["s1"].min()  # records sectores
                b_s2 = laps_df["s2"].min()
                b_s3 = laps_df["s3"].min()
                last_n = laps_df.tail(20).iloc[::-1] # ultimas 20 vueltas 

                tb_data = [["STEP / N°", "S1", "S2", "S3", "TOTAL (S4)"]] # header
                for _, r in last_n.iterrows():
                    tb_data.append([str(int(r['timestamp'])),
                        format_time(r.get('s1', 0)),
                        format_time(r.get('s2', 0)),
                        format_time(r.get('s3', 0)),
                        format_time(r['lap_time'])])

                table = ax2.table(cellText=tb_data, loc="center", cellLoc="center")
                table.auto_set_font_size(False)
                table.set_fontsize(8)
                table.scale(1, 1.3)

                # Colores
                for (r_idx, c_idx), cell in table.get_celld().items():
                    if r_idx == 0:  # Encabezado
                        cell.set_facecolor("#333333")
                        cell.get_text().set_color("white")
                        cell.get_text().set_weight('bold')
                    
                    elif r_idx > 0:
                        if r_idx == 1 and (c_idx == 0 or c_idx == 4):  # el ultimo siempre es record
                            cell.set_facecolor("#98fb98")
                            if c_idx == 4: cell.get_text().set_weight('bold')             
                        # Sectores 
                        current_data = last_n.iloc[r_idx - 1]
                        if c_idx == 1 and current_data['s1'] <= b_s1: cell.set_facecolor("#e0b0ff")
                        if c_idx == 2 and current_data['s2'] <= b_s2: cell.set_facecolor("#e0b0ff")
                        if c_idx == 3 and current_data['s3'] <= b_s3: cell.set_facecolor("#e0b0ff")

                ax2.set_title("Tiempos de Vuelta", fontsize=10, fontweight="bold")
            else:
                ax2.text(0.5, 0.5, "Esperando datos...", ha='center')
        except Exception as e:
            ax2.text(0.5, 0.5, f"Error: {e}", ha='center', fontsize=7)
    plt.tight_layout()


fig = plt.figure(figsize=(14, 6))
ani = FuncAnimation(fig, animate, interval=5000) # 10s (0.1 fps)

print("Monitor dual activado (Gráfico + Tabla).")
plt.show()