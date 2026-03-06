# pruebas de la mask
import numpy as np
import matplotlib.pyplot as plt
from track import Track

# cargar pista
track = Track("assets/track_1-mask.png")

x_meta = track.start_line["x"]         # meta
y1 = track.start_line["y1"]
y2 = track.start_line["y2"]
cl = track.centerline

fig, axs = plt.subplots(1, 2, figsize=(16, 7))

# Circuito
axs[0].set_title("Circuito - SDF")
axs[0].imshow(track.binary, cmap="gray", alpha=0.3)
sdf_plot = axs[0].imshow(track.sdf, cmap="coolwarm", alpha=0.6) # campo SDF
plt.colorbar(sdf_plot, ax=axs[0], label="Valor SDF (Centro = cresta roja)")
for s in track.sectors:  # (xy),(xy)
    axs[0].plot((s[0][0], s[1][0]),(s[0][1], s[1][1]), color = "b") 
axs[0].plot([x_meta, x_meta], [y1,y2], color = "red", linewidth = 3)
axs[0].plot(cl[:,0], cl[:,1], color="green", label="Centerline")
axs[0].scatter(cl[0,0], cl[0,1], color="yellow", s=50, label="Meta (S=0)")

X, Y = np.meshgrid(np.arange(0, 799, 25), np.arange(0, 554, 25))
v_get_direc = np.vectorize(track.get_track_direction) # chequeo de la dirección
angs = np.radians(v_get_direc(X,Y))
U = np.cos(angs) * 10
V = np.sin(angs) * 10
Q = axs[0].quiver(X, Y, U, -V, color='white', units="width")

# Progreso
steps = np.arange(len(track.arc_lengths))
# xx = np.linspace(0,1500,2)
# axs[1].plot(xx,2.1*xx,"r")
axs[1].plot(steps, track.arc_lengths, color="blue")
axs[1].grid()
axs[1].set_title("Continuidad del Progreso")
axs[1].set_xlabel("Índice del punto")
axs[1].set_ylabel("Distancia acumulada (px)")

# Checkeo de SDF en la propia centerline 
sample_pts = cl[::50]
sdf_values = [track.get_lateral_distance(p[0], p[1]) for p in sample_pts]
print(f"SDF promedio en centerline: {np.mean(sdf_values):.2f} (Más alto posible)")

# plt.legend()
plt.tight_layout()
plt.show()

# testeo progreso

print("testeo progreso")
sample_indices = np.linspace(0,len(cl)-1, 20).astype(int)
for i in sample_indices:
    x,y = cl[i]
    s = track.get_progress(x,y)
    print(f"Index {i:5d} | progress: {s:.2f}")
print("\nTotal track lenght:", track.total_length)