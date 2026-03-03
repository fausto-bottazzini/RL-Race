# Entorno 
import numpy as np
import pygame
import gymnasium as gym 
from gymnasium import spaces
import time
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
    "Toda la información necesaria para manejar"
    angle_rad = np.radians(-car.angle)
    cos_a = np.cos(angle_rad)
    sen_a = np.sin(angle_rad)

    # vel
    vel_long = (car.velocity.x * cos_a + car.velocity.y * sen_a) / car.max_speed
    vel_lat = (-car.velocity.x * sen_a + car.velocity.y * cos_a) / car.max_speed
    # ang
    p_future, angle_to_future = track.get_future(car.position.x, car.position.y, look_ahead = 300)
    diff_future = (angle_to_future - car.angle + 180) % 360 - 180
    cos_future = np.cos(np.radians(diff_future))
    sen_future = np.sin(np.radians(diff_future))

    track_dir = track.get_track_direction(car.position.x, car.position.y)
    velocity_angle = np.degrees(np.arctan2(car.velocity.y, car.velocity.x))
    alignment = np.cos(np.radians(velocity_angle - track_dir))

    # sdf
    on_track = 1.0 if track.is_inside(car.position.x,car.position.y) else 0.0
    sdf_raw = track.get_lateral_distance(car.position.x, car.position.y)
    sdf_norm = np.clip(sdf_raw / 25, -1, 1)
    # ojos
    lidar_angles = [-90, -45, -20, -10, 0, 10, 20, 45, 90]
    distances = []
    front_pos = car.position + pygame.Vector2(10, 0).rotate(-car.angle)
    for rel_angle in lidar_angles:
        total_angle = -(car.angle + rel_angle) # sistema de la pantalla
        d = get_eyes(track, front_pos, total_angle)
        distances.append(d)
    observation = np.array([vel_long, vel_lat, alignment, cos_future, sen_future, sdf_norm, on_track, *distances], dtype=np.float32)
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
        # inputs                          # [Vel_X, Vel_Y, SDF, Error_Angular, 5 Lidars]
        self.observation_space = spaces.Box(low=-1, high=1, shape=(16,), dtype=np.float32)
        self.out_track_timer = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.car = Car(x=self.track.start_line["x"], y=512)
        self.step_count = 0
        self.last_progress = 0

        return get_observation(self.car, self.track), {}

    def step(self, action):
        # control     
        obs = get_observation(self.car, self.track)
        on_track = obs[6]
        old_pos = pygame.Vector2(self.car.position.x, self.car.position.y)
        self.car.update(action, dt=1/60, on_track=on_track)

        # progreso y reward
        current_progress = self.track.get_progress(self.car.position.x, self.car.position.y)
        progress_reward = current_progress - self.last_progress # solo si mejora
        
        if progress_reward <= 0:
            reward = -0.01    # quiero o reversa
        else:
            reward = progress_reward * (0.5 + obs[2] * 0.5)  # bonus por alineado

        if action[2] and action[3]:
            reward -= 0.5  # forma no comun de manejar

        future_curvature = abs(obs[4])  #sen_future
        speed_norm = self.car.velocity.length() / self.car.max_speed
        if future_curvature > 0.6 and speed_norm > 0.5:
            if action[4]: # frenar antes de la curva
                reward += 0.02
            if speed_norm > 0.8: # sigue rapido
                reward -= 0.3 


        if self.car.velocity.length() > (self.car.max_speed * 0.85):
            reward -= (self.car.velocity.length() / self.car.max_speed) * 0.05

        # reward += obs[5]  # sdf
        if abs(obs[1]) > 0.5 and (current_progress - self.last_progress) < 0.01:
            reward -= 0.05   # pensalizacion por derrapar y no avanzar

        # Salirse - 5s 
        terminated = False 
        if not on_track: 
            reward -= 0.5
            if self.out_track_timer is None:
                self.out_track_timer = time.time()
            elif time.time() - self.out_track_timer > 2.0:
                terminated = True
                reward -= 10.0 
        else:
            self.out_track_timer = None 

        # meta
        if reward < -self.track.total_length / 2: # si se resetea
            reward += 100  # bono por vuelta
        
        self.last_progress = current_progress
        self.step_count += 1
        truncated = self.step_count > 5000 # Límite de tiempo


        return obs, reward, terminated, truncated, {}
