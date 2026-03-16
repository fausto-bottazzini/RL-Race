import os, glob
import pandas as pd
import numpy as np
import pygame
import imageio
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.lines import Line2D
from matplotlib import colormaps 
from stable_baselines3 import PPO
import cv2

total_length = 3114.2294737798984 # px
sectores = [((212,109), (193,132)), ((201,396), (201, 368)), ((443, 129), (470, 140))]
x_meta, y1, y2 = 480, 487, 522


def format_time(t): 
    minutes = int(t // 60)
    seconds = int(t % 60)
    millis  = int((t - int(t)) * 1000)
    return f"{minutes:02}:{seconds:02}.{millis:03}"

def evaluate_models(models_folder, env, n_tries=3, seed=13):  
    "Barre los modelos y extrae la telemetria de las mejores vueltas."
    model_files = sorted(glob.glob(os.path.join(models_folder, "*.zip")))
    all_telemetries = []
    best_time = float("inf")
    best_model = ""

    print(f"Evaluando {len(model_files)} modelos")
    i = 1
    for model_path in model_files:
        print(f"Evaluando {i}/{len(model_files)}")
        model_name = os.path.basename(model_path).replace(".zip","")
        model = PPO.load(model_path)
        best_model_time = float("inf")
        best_model_telemetry = None

        for _ in range(n_tries):
            obs, info = env.reset(seed=seed)
            full_episode_data = []
            action_history = []
            lap_start_idx = 0
            done = False

            while not done:
                action, _ = model.predict(obs, deterministic=False) # son malos deterministas
                obs, reward, terminated, truncated, info = env.step(action)
                full_episode_data.append({"model_name": model_name, 
                                 "x": env.car.position.x, "y": env.car.position.y,
                                 "speed": env.car.velocity.length(),
                                 "progress": env.track.get_progress(env.car.position.x, env.car.position.y),
                                 "thr": action[0], "brk": action[4], "rev": action[1]})
                action_history.append(action)

                if info.get("is_lap_completed"):
                    current_time = info["lap_time"]
                    if current_time < best_model_time:
                        best_model_time = current_time
                        best_model_telemetry = pd.DataFrame(full_episode_data[lap_start_idx:])

                        if best_model_time < best_time:
                            best_time = best_model_time
                            best_model = model_name
                    lap_start_idx = len(full_episode_data)
                if terminated or truncated:
                    done = True

        if best_model_telemetry is not None:
            all_telemetries.append(best_model_telemetry)
        i += 1
    print(f"\nVuelta Rapida Global: {best_time:.3f}s ({model_name})")

    df_all = pd.concat(all_telemetries, ignore_index=True)
    df_all.to_csv("data/analysis/all_telemetries.csv", index=False)
    return df_all, best_model

def scan_best_lap(model_path, env, n_tries=10, fps=25, seed=13):
    "Itera un modelo para encontrar su mejor vuelta y graba el gif."
    model = PPO.load(model_path)
    model_name = os.path.basename(model_path).replace(".zip", "")
    
    best_time = float('inf')
    best_telemetry = None
    best_frames = []

    pygame.init()
    pygame.font.init()
    font_main = pygame.font.SysFont("monospace", 20, bold=True)
    font_mono = pygame.font.SysFont("monospace", 14, bold=True)
    WIDHT, HEIGHT = 1088, 720  #1080
    screen = pygame.display.set_mode((WIDHT,HEIGHT))
    track_img = pygame.image.load("assets/track_1-mask.png").convert()

    os.makedirs("data/analysis", exist_ok=True)
    os.makedirs("plots", exist_ok=True)

    print(f"Buscando la mejor vuelta para {model_name}...")
    for attempt in range(n_tries):
        print(f"Run {attempt+1}/{n_tries}")
        obs, _ = env.reset(seed= seed + attempt) # variable
        done = False
        n_lap = 1
        current_telemetry = []
        current_frames = []

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return best_time, best_telemetry
                
            action, _ = model.predict(obs, deterministic=False)
            obs, reward, terminated, truncated, info = env.step(action)

            u_env = env.unwrapped # para leer las variables internas
            if u_env.timer_started:
                state_data = {'model_name': model_name,
                    'x': u_env.car.position.x, 'y': u_env.car.position.y,
                    'speed': u_env.car.velocity.length(), 'progress': u_env.track.get_progress(u_env.car.position.x, u_env.car.position.y),
                    'thr': action[0], 'rev': action[1], 'brk': action[4],
                    'time': u_env.current_lap_time}
                current_telemetry.append(state_data)
                
                # Renderizar y capturar Frame
                # circuito
                screen.fill((30, 30, 30))
                screen.blit(track_img, (0, 0))
                start_x = u_env.track.start_line["x"]
                pygame.draw.line(screen, (255, 0, 0), (start_x, u_env.track.start_line["y1"]), (start_x, u_env.track.start_line["y2"]), 3)
                for s in u_env.track.sectors:
                    pygame.draw.line(screen, (0, 0, 255), s[0], s[1], 2)

                # auto
                pygame.draw.circle(screen, (255, 0, 0), (int(u_env.car.position.x), int(u_env.car.position.y)), 5)

                lidar_angles = [-90, -45, -20, -10, 0, 10, 20, 45, 90]
                for i, rel_angle in enumerate(lidar_angles):
                    dist = obs[i+7] * 500 
                    angle = np.radians(-(u_env.car.angle + rel_angle))
                    end_x = u_env.car.position.x + dist * np.cos(angle)
                    end_y = u_env.car.position.y + dist * np.sin(angle)
                    pygame.draw.line(screen, (255, 0, 0), u_env.car.position, (end_x, end_y), 1)

                right_indicator = u_env.car.position + pygame.Vector2(0,5).rotate(-u_env.car.angle)
                front_indicator = u_env.car.position + pygame.Vector2(10,0).rotate(-u_env.car.angle)
                pygame.draw.line(screen, (0, 0, 0), u_env.car.position, (front_indicator.x, front_indicator.y), 2)
                pygame.draw.line(screen, (0, 0, 0), u_env.car.position, (right_indicator.x, right_indicator.y), 2)

                # crono
                right_margin = WIDHT - 20
                time_str = format_time(u_env.current_lap_time)
                time_surface = font_main.render(f"TIEMPO: {time_str}", True, (255, 255, 255))
                text_rect = time_surface.get_rect(topright=(right_margin, 20))
                screen.blit(time_surface, text_rect)
                for i, s_time in enumerate(u_env.sector_times):
                    s_surface = font_main.render(f"S{i+1}: {format_time(s_time)}", True, (200, 200, 255))
                    s_rect = s_surface.get_rect(topright=(right_margin, 50 + (i * 25)))
                    screen.blit(s_surface, s_rect)

                # obs y act
                lbls = ["VLng", "VLat", "Algn", "CosF", "SinF", "SDFn", "Trck", "L-90", "L-45", "L-20", "L-10", "L_00", "R+10", "R+20", "R+45", "R+90"]
                obs_lbl_str = "OBS: [ " + " ".join([f"{l:>5}" for l in lbls]) + " ]"
                obs_val_str = "     [ " + " ".join([f"{v:>5.2f}" for v in obs]) + " ]"
                lbl_surf = font_mono.render(obs_lbl_str, True, (180, 180, 180))
                val_surf = font_mono.render(obs_val_str, True, (0, 255, 255))
                screen.blit(lbl_surf, lbl_surf.get_rect(midbottom=(WIDHT//2, HEIGHT - 100)))
                screen.blit(val_surf, val_surf.get_rect(midbottom=(WIDHT//2, HEIGHT - 80)))

                act_y = HEIGHT - 60
                act_names = ["THR", "REV", "LFT", "RGT", "BRK"]
                act_str_width = font_mono.size("ACT:  ")[0] + sum([font_mono.size(f"[{a}] ")[0] for a in act_names])
                act_x = (WIDHT - act_str_width) // 2
                act_title = font_mono.render("ACT:  ", True, (255, 255, 255))
                screen.blit(act_title, (act_x, act_y))
                act_x += act_title.get_width()
                for i, act_lbl in enumerate(act_names):
                    color = (0, 255, 0) if action[i] else (70, 70, 70) # Verde si apretado, Gris oscuro si suelto
                    surf = font_mono.render(f"[{act_lbl}] ", True, color)
                    screen.blit(surf, (act_x, act_y))
                    act_x += surf.get_width()

                # lap
                att_surface = font_main.render(f"Lap: {n_lap}", True, (150, 150, 150))
                att_rect = att_surface.get_rect(topright=(right_margin, HEIGHT - 40))
                screen.blit(att_surface, att_rect)

                # Capturar el array de la pantalla
                pygame.display.flip()
                frame = pygame.surfarray.array3d(screen)
                frame = np.transpose(frame, (1, 0, 2))  # Pygame usa (x, y, rgb), Imageio espera (y, x, rgb), por lo que transponemos.
                frame = cv2.resize(frame, (WIDHT//2, HEIGHT//2)) # reducir resolución
                current_frames.append(frame)

            done = terminated or truncated
            
            # Meta
            if info.get("is_lap_completed"):
                lap_time = info["lap_time"]
                if lap_time < best_time:
                    best_time = lap_time
                    best_telemetry = list(current_telemetry) 
                    best_frames = list(current_frames)
                    print(f"Nuevo Record, Lap: {n_lap}| Tiempo: {best_time:.3f}s")

                current_telemetry.clear()
                current_frames.clear()
                n_lap += 1

    if best_telemetry is not None:
        font_big = pygame.font.SysFont("monospace", 40, bold=True)  # aviso
        screen.fill((20, 20, 20))
        msg = font_big.render("GUARDANDO MEJOR VUELTA...", True, (255, 215, 0))
        screen.blit(msg, msg.get_rect(center=(WIDHT//2, HEIGHT//2)))
        pygame.display.flip()
        
        csv_name = f"data/analysis/telemetry_{model_name}_{best_time:.2f}s.csv"
        video_name = f"plots/lap_{model_name}_{best_time:.2f}s.mp4"
        
        pd.DataFrame(best_telemetry).to_csv(csv_name, index=False)
        writer = imageio.get_writer(video_name, format="ffmpeg", mode="I", fps=fps)
        for f in best_frames:
            writer.append_data(f)
        writer.close()
        print(f"\nEscaneo completo. Archivos guardados:\n- {csv_name}\n- {video_name}")
    else:
        print("\nEl modelo no logró completar ninguna vuelta válida en los intentos dados.")
    pygame.quit()

    return best_time, best_telemetry

##########

def plot_learning_curve(t1_path, t2_path, t3_path=None, save=False):    
    "progreso vs timesteps"
    df1 = pd.read_csv(t1_path)
    df2 = pd.read_csv(t2_path)
    
    # offset1 = df1['timesteps'].max()
    offset1 = 2e6 - df2['timesteps'].min()
    df2['timesteps'] += offset1
    dfs = [df1, df2]
    
    if t3_path and os.path.exists(t3_path):
        df3 = pd.read_csv(t3_path)
        dfs.append(df3)

    plt.figure(figsize=(12, 5))
    plt.axvline(x=2e6, color="black", linestyle="--", label="Inicio T2")
    for i in (1,2,3):
        if i == 1:
            plt.axhline(y=total_length*i, color="red", linestyle="-", alpha=0.3/i, label="Vueltas Completadas")
        else: 
            plt.axhline(y=total_length*i, color="red", linestyle="-", alpha=0.3/i)
    for df in dfs:
        plt.fill_between(df["timesteps"], df["progress"]-df["std"], df["progress"]+df["std"], alpha=0.2, color="#0000FF")
        plt.plot(df["timesteps"], df['progress'], color="darkblue")

    if t3_path:
        plt.axvline(x=df2['timesteps'].max(), color="red", linestyle="--", label="Inicio T3 (Fine-Tuning)")

    plt.xlabel("Timesteps")
    plt.ylabel("Distancia Recorrida [px]")
    plt.title("Evolución del Aprendizaje: Exploración y Optimización Continua")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save:
        plt.savefig("plots/learning_curve.png", dpi=300)
    plt.show()

def plot_lap_times(best_laps_paths, save=False):   
    "Mejora de los tiempos de vuelta y sectores integrados en una sola figura."
    df12 = pd.read_csv(best_laps_paths[0])
    df3 = pd.read_csv(best_laps_paths[1])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={'width_ratios': [1.8, 1]})
    gap = (df12['lap_time'].max() - df3['lap_time'].min())

    # Tiempos de Vuelta (Principal)
    ax1.axvline(x=600000, color="r", alpha=0.3, label="Cambios en el modelo")
    ax1.axvline(x=1000000, color="r", alpha=0.3)
    ax1.axvline(x=1600000, color="r", alpha=0.3)
    
    ax1.plot(df12['timestamp'], df12['lap_time'], ".-", color='black', linewidth=2, label=f"Mejora total: -{gap:.2f} s")
    ax1.plot(df3['timestamp'], df3['lap_time'], ".-", color='black', linewidth=2)
    
    ax1.legend(loc='upper right')
    ax1.set_xlabel('Timestamp')
    ax1.set_ylabel('Tiempo [s]', color='black')
    ax1.set_title("Evolución General de Tiempos de Vuelta")
    ax1.grid(True, alpha=0.3)

    # Tiempos de Sectores
    ax2.axvline(x=600000, color="r", alpha=0.3, label="cambio")
    ax2.axvline(x=1000000, color="r", alpha=0.3)
    ax2.axvline(x=1600000, color="r", alpha=0.3)

    # ax2.fill_between([0, 2.9e6], df12['lap_time'].max(), df3['lap_time'].min(), color="grey", alpha=0.2, label=f"Gap: -{gap:.2f} s")

    # Datos df12
    ax2.plot(df12['timestamp'], df12['lap_time'], ".-", color='black', linewidth=2, label='Tiempo Total (S4)')
    ax2.plot(df12['timestamp'], df12['s1'], '+-', color="b", label='Sector 1')
    ax2.plot(df12['timestamp'], df12['s2'], '+-', color="#FFA500", label='Sector 2')
    ax2.plot(df12['timestamp'], df12['s3'], '+-', color="g", label='Sector 3')
    # Datos df3
    ax2.plot(df3['timestamp'], df3['lap_time'], ".-", color='black', linewidth=2)
    ax2.plot(df3['timestamp'], df3['s1'], 'x-', color="b")
    ax2.plot(df3['timestamp'], df3['s2'], 'x-', color="#FFA500")
    ax2.plot(df3['timestamp'], df3['s3'], 'x-', color="g")

    ax2.legend(loc='lower right', fontsize='small')
    ax2.set_xlabel('Timestamp')
    ax2.set_title("Desglose por Sectores")
    ax2.grid(True, alpha=0.3)
    
    # plot
    plt.tight_layout()
    if save:
        plt.savefig("plots/lap_times_combined.png", dpi=300)
    plt.show()

def plot_telemetry_vel(telemetry_df, track_img = "assets/track_1-mask.png", save=False):  
    "Trazada sobre el circuito, coloreado por velocidad"
    plt.figure(figsize=(10, 7))   
    # circuito
    img = mpimg.imread(track_img)
    h, w = img.shape[:2]
    plt.imshow(img, extent=[0, w, h, 0])
    for s in sectores:  # (xy),(xy)
        plt.plot((s[0][0], s[1][0]),(s[0][1], s[1][1]), color = "b") 
    plt.plot([x_meta, x_meta], [y1,y2], color = "red", linewidth = 3) 
    scatter = plt.scatter(telemetry_df['x'], telemetry_df['y'], c=telemetry_df['speed'], cmap='plasma', s=10, alpha=0.8)
    plt.colorbar(scatter, label='Velocidad [m/s]')
    plt.xlabel("Posición X")
    plt.ylabel("Posición Y")
    plt.title("Trazada y Mapa de Velocidad")
    plt.axis('equal')
    plt.tight_layout()
    if save: plt.savefig("plots/telemetry_vel.png", dpi=300)
    plt.show()

def plot_telemetry_act(telemetry_df, track_img = "assets/track_1-mask.png", save=False):  
    "Trazada sobre el circuito, coloreado por acciones"
    # data
    x = telemetry_df["x"].values
    y = telemetry_df["y"].values
    thr = telemetry_df["thr"].values
    brk = telemetry_df["brk"].values
    rev = telemetry_df["rev"].values
    tiempo_s = len(telemetry_df) * 1/25
    tiempo_m = format_time(tiempo_s)

    fig, ax = plt.subplots(figsize=(10, 8))
    # circuito
    img = mpimg.imread(track_img)
    h, w = img.shape[:2]
    ax.imshow(img, extent=[0, w, h, 0])
    for s in sectores:  # (xy),(xy)
        ax.plot((s[0][0], s[1][0]),(s[0][1], s[1][1]), color = "b") 
    ax.plot([x_meta, x_meta], [y1,y2], color = "red", linewidth = 3)
    # actions
    for i in range(len(x) - 1):
        color = (brk[i], thr[i], rev[i])
        if sum(color) == 0:
            color = (0.5, 0.5, 0.5)
        else:
            max_val = max(color)
            color = tuple(c/max_val for c in color)
        ax.plot(x[i:i+2], y[i:i+2], color=color, linewidth=2)
    # estilo
    ax.text(0.05, 0.95, f"TIEMPO: {tiempo_m}", transform=ax.transAxes, fontsize=10, fontweight='bold', va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    custom_lines = [Line2D([0], [0], color='green', lw=2),
                    Line2D([0], [0], color='red', lw=2),
                    Line2D([0], [0], color='blue', lw=2),
                    Line2D([0], [0], color='gray', lw=2)]
    ax.legend(custom_lines, ['Acelerando', 'Frenando', 'Reversa', 'Inercia'], loc=(0.8,0.8))
    ax.set_xlabel("Posición X")
    ax.set_ylabel("Posición Y")
    ax.set_title("Mejor Trazada")
    ax.axis('equal')
    plt.tight_layout()
    if save: plt.savefig("plots/telemetry_act.png", dpi=300)
    plt.show()

def plot_telemetry_evol(all_telemetries, step=5, models_folder = "data/models", track_img = "assets/track_1-mask.png", save=False):   
    # models_folder = "data/analysis/models_to_analyse"  # cambiar si es otra

    rutas_completas = sorted(glob.glob(os.path.join(models_folder, "*.zip")))
    nombres = [os.path.basename(f).replace(".zip", "") for f in rutas_completas]
    
    # reordenar (ver segun la carpeta de modelos)
    t1, t2, t3 = nombres.pop(0), nombres.pop(0), nombres.pop(0) 
    nombres.insert(len(nombres)-1, t1) # los movemos al fondo
    nombres.insert(len(nombres)-1, t2)
    nombres.insert(len(nombres)-1, t3)
    # print(nombres)
    # en caso de que desde un principio se pongan nombre en orden (alfabetico), no hace falta

    model_names = [m for m in nombres if m in all_telemetries['model_name'].unique()]
    selected_models = model_names[::step]

    fig, ax = plt.subplots(figsize=(11, 6))
    img = mpimg.imread(track_img)
    h, w = img.shape[:2]
    extent = [0, w, h, 0]
    ax.imshow(img, extent=extent)
    for s in sectores:  # (xy),(xy)
        ax.plot((s[0][0], s[1][0]),(s[0][1], s[1][1]), color = "b") 
    ax.plot([x_meta, x_meta], [y1,y2], color = "red", linewidth = 3)

    norm = plt.Normalize(0, len(selected_models))
    cmap = colormaps['viridis']
    for idx, name in enumerate(selected_models):
        df_model = all_telemetries[all_telemetries['model_name'] == name]
        if not df_model.empty:
            ax.plot(df_model["x"], df_model["y"], color=cmap(norm(idx)), alpha=0.6, linewidth=1.5)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, ax=ax, label='Evolución del Entrenamiento (Modelos)')
    cbar.set_ticks([])

    ax.set_title("Evolución de la Trazada")
    ax.set_aspect('equal')
    plt.tight_layout()
    if save: plt.savefig("plots/telemetry_evol.png", dpi=300)
    plt.show()

def plot_speed_profile(telemetry_df, save=False):  
    "Perfil de velocidad con zonas de frenado/reversa vs progreso en pista."
    df = telemetry_df[:-1]
    progress = df["progress"]
    speed = df["speed"]
    thr = df["thr"]
    brk = df["brk"]
    rev = df["rev"]

    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(progress, speed, color="black", linewidth=1.5, label="Velocidad")

    ax1.fill_between(progress, 0, speed, where=thr, color='green', alpha=0.3, label="Acelerando")
    ax1.fill_between(progress, 0, speed, where=brk, color='red', alpha=0.3, label="Freno")
    ax1.fill_between(progress, 0, speed, where=rev, color='blue', alpha=0.3, label="Reversa")

    ax1.axvline(x=total_length, color="k", linestyle="--", alpha=0.3)
    ax1.set_xlabel("Progreso en pista [px]")
    ax1.set_ylabel("Velocidad [m/s]")
    ax1.set_title("Perfil de Velocidad")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save: plt.savefig("plots/speed_profile.png", dpi=300)
    plt.show()

def plot_test_track(all_telemetries, step=5, track_img = "assets/track_1-mask.png", save=False):  
    from track import Track
    track = Track(track_img)
    cl = track.centerline

    model_names = [m for m in all_telemetries['model_name'].unique()]
    selected_models = model_names[::step]

    fig, axs = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={'width_ratios': [2, 1]})

    # Circuito
    axs[0].set_title("Circuito - SDF")
    axs[0].imshow(track.binary, cmap="gray", alpha=0.3)
    sdf_plot = axs[0].imshow(track.sdf, cmap="coolwarm", alpha=0.6) # campo SDF
    plt.colorbar(sdf_plot, ax=axs[0], label="Valor SDF")
    for s in track.sectors:  # (xy),(xy)
        axs[0].plot((s[0][0], s[1][0]),(s[0][1], s[1][1]), color = "b") 
    axs[0].plot([x_meta, x_meta], [y1,y2], color = "red", linewidth = 3)
    axs[0].plot(cl[:,0], cl[:,1], color="green", label="Centerline")
    axs[0].scatter(cl[0,0], cl[0,1], color="yellow", s=50, label="Meta (S=0)")
    axs[0].legend()

    X, Y = np.meshgrid(np.arange(0, 799, 25), np.arange(0, 554, 25))
    v_get_direc = np.vectorize(track.get_track_direction) # chequeo de la dirección
    angs = np.radians(v_get_direc(X,Y))
    U = np.cos(angs) * 10
    V = np.sin(angs) * 10
    Q = axs[0].quiver(X, Y, U, -V, color='white', units="width")

    # Progreso
    steps = np.arange(len(track.arc_lengths))
    for idx, name in enumerate(selected_models):  # para poder poner varios
        df_model = all_telemetries[all_telemetries['model_name'] == name]
        if not df_model.empty:
            if idx == 0:
                axs[1].plot(df_model["progress"][:-2].values, color="g", label="Model Progress")
            else:
                axs[1].plot(df_model["progress"][:-2].values) # index  (cuidado desfasaje)
    # xx = np.linspace(0,1500,2)
    # axs[1].plot(xx,2.1*xx,"r")
    axs[1].plot(steps, track.arc_lengths, color="blue", label = "Track Progress")
    axs[1].axhline(y=total_length, color="grey", linestyle="--", alpha=0.3)
    axs[1].grid()
    axs[1].set_title("Continuidad del Progreso")
    axs[1].set_xlabel("Índice del punto")
    axs[1].set_ylabel("Distancia acumulada (px)")


    plt.legend()
    plt.tight_layout()
    plt.show()


## Plots ## 

if __name__ == "__main__":

    # Gráficos de Entrenamiento #
    plot_learning_curve("data/logs/train1.progress.csv", "data/logs/progress_log.csv")
    plot_lap_times(["data/logs/best_laps_T2.csv", "data/logs/best_laps.csv"])
    
    # Obtener telemetrías #
    from env import TrackEnv2
    env = TrackEnv2(track_mask="assets/track_1-mask.png")

    # generar las telemetrias
    # all_telems, best_model_name = evaluate_models("data/models", env)  
    # best_model_path = os.path.join("data/models/", best_model_name + ".zip")
    # tiempo, best_telem = scan_best_lap(best_model_path, env, n_tries=10, fps = 50)  #x2

    # cargarlas
    all_telems = pd.read_csv("data/analysis/all_telemetries.csv") # cargar el archivo
    best_model_name = "ppo_T3"
    best_telem = pd.read_csv(f"data/analysis/telemetry_{best_model_name}_51.60s.csv")

    # modelos_disponibles = all_telems['model_name'].unique()

    # Gráficos de Telemetría #

    plot_telemetry_evol(all_telems, step=1)
    plot_telemetry_act(best_telem)
    plot_telemetry_vel(best_telem)
    plot_speed_profile(best_telem)
    plot_test_track(all_telems, step = 5)

    plt.close("all")

        

