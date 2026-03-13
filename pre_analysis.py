# Gráficos
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import ast

from track import Track
track = Track("assets/track_1-mask.png")

# 3114.2294737798984 (total_length)

x_meta = track.start_line["x"]         # meta
y1 = track.start_line["y1"]
y2 = track.start_line["y2"]

def format_time(t):
    minutes = int(t // 60)
    seconds = int(t % 60)
    millis  = int((t - int(t)) * 1000)
    return f"{minutes:02}:{seconds:02}.{millis:03}"

def plot_telemetry(data_path, track_image_path):
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print("No se encontró el archivo de telemetría.")
        return

    # Extraer datos
    df["action"] = df["action"].apply(ast.literal_eval) 
    x = np.array(df["x"])
    y = np.array(df["y"])
    act = np.array(df["action"].tolist())
    th = act[:,0]
    br = act[:,4]
    rev = act[:,1]

    # Configurar el gráfico
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Cargar y mostrar la imagen de la pista
    img = mpimg.imread(track_image_path)
    # Reescalar la imagen según el TRACK_SCALE para que coincida con las coordenadas
    h, w = img.shape[:2]
    ax.imshow(img, extent=[0, w, h, 0], alpha=0.7)

    for s in track.sectors:  # (xy),(xy)
        plt.plot((s[0][0], s[1][0]),(s[0][1], s[1][1]), color = "b") 
    plt.plot([x_meta, x_meta], [y1,y2], color = "red", linewidth = 3)

    # Dibujar la trayectoria con colores dinámicos
    for i in range(len(x) - 1):
        # Color dinámico basado en inputs (R=Brake, G=Throttle, B=Reverse)
        color = (br[i], th[i], rev[i])
        
        # Si no hay ningún input, gris para la inercia
        if sum(color) == 0:
            color = (0.5, 0.5, 0.5)
        else:
            # Normalizar para que sea un color válido si hay combinación
            max_val = max(color)
            color = tuple(c/max_val for c in color)

        ax.plot(x[i:i+2], y[i:i+2], color=color, linewidth=2)

    # Añadir marcadores para los sectores (Coordenadas base * SCALE)
    sectors = [(226, 102), (205, 420), (467, 174)]

    # Título y tiempo total
    plt.title(f"Telemetría", fontsize=15)
    ax.set_xlabel("X (Píxeles)")
    ax.set_ylabel("Y (Píxeles)")
    
    # Leyenda manual
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color='green', lw=2),
                    Line2D([0], [0], color='red', lw=2),
                    Line2D([0], [0], color='blue', lw=2),
                    Line2D([0], [0], color='gray', lw=2)]
    ax.legend(custom_lines, ['Acelerando', 'Frenando', 'Reversa', 'Inercia'], loc='upper right')

    plt.show()

def plot_learning_curve(csv_path):
    df = pd.read_csv(csv_path)
    plt.figure(figsize=(10,6))
    plt.plot(df["timesteps"], df["progress"], ".-", label="Progreso por Step")
    plt.title("Curva de Progreso")
    plt.xlabel("Timesteps")
    plt.ylabel("Progreso en pista")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":

    # plot_learning_curve("data/logs/3M_progress_log.csv")
    # plot_learning_curve("data/logs/progress_log.csv")
    # plot_telemetry("data/manual_test.csv", "assets/track_1-mask.png")
    plot_telemetry("data/last_run.csv", "assets/track_1-mask.png")