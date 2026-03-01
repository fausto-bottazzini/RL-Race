# Entorno 
import numpy as np
import pygame
import gymnasium as gym 
from gymnasium import spaces
from car import Car
from track import Track

# Utilidades 

# ojos para el modelo
def get_eyes(track, pos, angle, max_dist = 500):
    "Tira rayos hacia adelante hasta encontrar el borde del circuito"
    for dist in range(0, max_dist, 1):
        dx = dist * np.cos(np.radians(angle))
        dy = dist * np.sin(np.radians(angle))
        check_x = int(pos.x + dx)
        check_y = int(pos.y + dy)

        if not track.is_inside(check_x,check_y):
            return dist / max_dist  # normalizado
    return 1.0

def get_observation(car,track):
    lidar_angles = [-45,-20, 0, 20, 45]
    distances = []
    for rel_angle in lidar_angles:
        total_angle = -(car.angle + rel_angle) # sistema de la pantalla
        d = get_eyes(track, car.position, total_angle)
        distances.append(d)
    observation = np.array([car.velocity.length() / car.max_speed,
                            track.get_lateral_distance(car.position.x,car.position.y) / 20,
                            *distances], dtype=np.float32)
    return observation


# Entorno
class TrackEnv(gym.Env):
    def __init__(self, track_mask):
        super(TrackEnv, self).__init__()

        self.track = Track(track_mask)
        self.car = None    # necesitamos varios
        self.step_count = 0

        # acciones
        self.action_space = spaces.MultiBinary(5)  # thr # rev # lft # rgt # brk  
        # inputs
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.car = Car(x=self.track.start_line["x"], y=512)
        self.step_count = 0
        self.last_progress = 0

        return get_observation(self.car, self.track), {}


    def step(self, action):
        # control     
        on_track = self.track.is_inside(self.car.position.x,self.car.position.y)
        self.car.update(action, dt=1/60, on_track=on_track)

        # progreso y reward
        current_progress = self.track.get_progress(self.car.position.x, self.car.position.y)
        reward = (current_progress - self.last_progress) # solo si mejora

        # meta
        if reward < -self.track.total_length / 2: # si se resetea
            reward += 100  # bono por vuelta

        self.last_progress = current_progress

        # terminar
        terminated = not on_track
        if terminated:
            reward -= 20.0 # Castigo por chocar
            
        truncated = self.step_count > 5000 # Límite de tiempo
        self.step_count += 1

        return get_observation(self.car, self.track), reward, terminated, truncated, {}
