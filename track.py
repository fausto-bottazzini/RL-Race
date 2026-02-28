# Circuito
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt, gaussian_filter1d
from skimage.measure import find_contours
from skimage import io
from scipy.spatial import cKDTree 


class Track: 
    def __init__(self, track_mask):
        img = io.imread(track_mask, as_gray=True)   # cargado del circuito
        treshold = img.mean()
        binary = img > treshold     # convertir a binario 
        self.binary = binary.astype(np.uint8)
        self.height, self.width = self.binary.shape

        self.start_line = {"x": 480, "y1": 487, "y2": 522}  #  ((x,y1),(x,y2))
        self.sectors = [((212,109), (193,132)),   # gate 1  # ((x1,y1),(x2,y2))
                        ((201,396), (201, 368)),  # gate 2
                        ((443, 129), (470, 140))] # gate 3

        # SDF
        self.dist_inside = distance_transform_edt(self.binary)  # distancia a bordes internos
        self.dist_outside = distance_transform_edt(1 - self.binary)  # distancia a bordes externos
        self.sdf = self.dist_inside - self.dist_outside    

        # Centerline
        center_level = 13.6 # np.max(self.sdf) * 0.7  # tomamos el maximo del sdf (un valor que este en toda la vuelta) (H)
        center_isolines = find_contours(self.sdf, level=center_level)
        
        if center_isolines:  # ordenarla
            cl = sorted(center_isolines, key=lambda x: len(x), reverse=True)[0] # la mas larga por si hay ruido
            self.centerline = cl[:, ::-1] # (x, y)
        else:
            raise ValueError("No se encontró una cresta en el SDF.")

        self.centerline = self._resample_closed_curve(self.centerline, 1500)  # aseguramos que sea unica
        self.centerline = gaussian_filter1d(self.centerline, sigma=5, axis=0) # suavizado

        # Roll desde la meta
        meta_pos = np.array([self.start_line["x"], (self.start_line["y1"] + self.start_line["y2"])/2])
        _, start_idx = cKDTree(self.centerline).query(meta_pos)
        self.centerline = np.roll(self.centerline, -start_idx, axis=0)

        self.kd_tree = cKDTree(self.centerline) # KD-Tree para búsqueda rápida

        # Parametrización 
        diffs = np.diff(self.centerline, axis=0, append=self.centerline[:1]) 
        self.arc_lengths = np.concatenate([[0], np.cumsum(np.linalg.norm(diffs, axis=1))])  # longitud acumulada 
        self.total_length = self.arc_lengths[-1]                          #  self.arc_lengths = np.cumsum(seg_lengths)

        # Telemetria 
        self.telemetry = {"step": [], "x": [], "y": [], "action": [], "progress": [], "on_track": []}

    # Tele
    def record_telemetry(self, step, action, car):
        "Guarda los datos del auto en el paso actual"
        self.telemetry["step"].append(step)
        self.telemetry["x"].append(car.position.x)
        self.telemetry["y"].append(car.position.y)
        self.telemetry["action"].append(action)
        self.telemetry["progress"].append(self.get_progress(car.position.x,car.position.y))
        self.telemetry["on_track"].append(self.is_inside(car.position.x,car.position.y))

    def export_telemetry(self, filename = "telemetry.csv"):
        pd.DataFrame(self.telemetry).to_csv(filename, index=False)
        # print(f"Telemetria guardada en {filename}")

    # Pista
    def is_inside(self, x, y):
        "Verificar si el coche está dentro de los límites de la pista"
        x_idx = int(np.clip(x, 0, self.width-1))
        y_idx = int(np.clip(y, 0, self.height-1))
        return self.binary[y_idx,x_idx] == 1

    def get_progress(self, x, y):
        "Progreso basado en la posición a lo largo del esqueleto"
        _, idx = self.kd_tree.query([x,y])
        return self.arc_lengths[idx]

    def get_lateral_distance(self, x, y):
        "Devuelve qué tan lejos está el auto del centro (SDF)"
        x_idx = int(np.clip(x, 0, self.width-1))
        y_idx = int(np.clip(y, 0, self.height-1))
        return self.sdf[y_idx, x_idx]

    # Sectors
    def check_gate_crossing(self, prev_pos, curr_pos, gate):
        """
        Detecta si el segmento (prev_pos -> curr_pos) intersecta con la gate ((x1,y1), (x2,y2))
        """
        p1, p2 = np.array(gate[0]), np.array(gate[1])
        a, b = np.array([prev_pos.x, prev_pos.y]), np.array([curr_pos.x, curr_pos.y])
        
        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

        # Intersección de segmentos
        return ccw(a, p1, p2) != ccw(b, p1, p2) and ccw(a, b, p1) != ccw(a, b, p2)

    def check_finish_crossing(self, prev_pos, curr_pos):
        # Simplificado usando la lógica de gates
        gate_meta = ((self.start_line["x"], self.start_line["y1"]), 
                     (self.start_line["x"], self.start_line["y2"]))
        return self.check_gate_crossing(prev_pos, curr_pos, gate_meta)

    def _resample_closed_curve(self, curve, n_points):
        diffs = np.diff(curve, axis=0)
        seg_lengths = np.linalg.norm(diffs, axis=1)
        
        arc = np.zeros(len(curve))
        arc[1:] = np.cumsum(seg_lengths)
        total_length = arc[-1]
        uniform_arc = np.linspace(0, total_length, n_points)

        new_x = np.interp(uniform_arc, arc, curve[:, 0])
        new_y = np.interp(uniform_arc, arc, curve[:, 1])
        
        return np.stack([new_x, new_y], axis=1)