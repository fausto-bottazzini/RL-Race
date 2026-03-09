# Entorno 
import numpy as np
import random as rm
import pygame
import gymnasium as gym 
from gymnasium import spaces
from car import Car
from track import Track

# Utilidades 

# ojos para el modelo
def get_eyes(track, pos, angle, max_dist = 500):
    "Tira rayos hacia adelante hasta encontrar el borde del circuito"
    for dist in range(0, max_dist, 5):   # manejar la resolución de los ojos 5px/paso
        dx = dist * np.cos(np.radians(angle))
        dy = dist * np.sin(np.radians(angle))
        check_x = int(pos.x + dx)
        check_y = int(pos.y + dy)

        if not track.is_inside(check_x,check_y):
            return dist / max_dist  # normalizado
    return 1.0

def get_observation(car,track):
    "Toda la información necesaria para manejar"

    forward = pygame.Vector2(1,0).rotate(-car.angle)
    right = pygame.Vector2(0,1).rotate(-car.angle)

    # vel
    vel_long = car.velocity.dot(forward) / car.max_speed
    vel_lat = car.velocity.dot(right) / car.max_speed

    # ang
    p_future, angle_to_future = track.get_future(car.position.x, car.position.y, look_ahead = 150)
    car_angle_pyg = -car.angle # cuidado diferente forma de tomar angulos
    diff_future = (angle_to_future - car_angle_pyg + 180) % 360 - 180
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

#####################

# Primer entrenamietno / Completar una vuelta

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
        self.out_track_counter = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if rm.random() < 0.8:     # 80% inicio aleatorio para aprender toda la pista
            random_progress = rm.uniform(0, self.track.total_length - 1)
            pos = self.track.get_point_at_dist(random_progress)
            angle = self.track.get_track_direction(pos.x,pos.y)
            self.car = Car(x=pos.x, y=pos.y, angle=-angle)
            self.last_progress = random_progress
        else:
            self.car =  Car(x=self.track.start_line["x"], y=512)
            self.last_progress = 0

        self.total_ep_prog = 0 # para el callback
        self.step_count = 0
        self.out_track_counter = 0
        return get_observation(self.car, self.track), {}

    def step(self, action):
        # control     
        obs = get_observation(self.car, self.track)
        on_track = obs[6]
        old_pos = pygame.Vector2(self.car.position.x, self.car.position.y)
        self.car.update(action, dt=1/25, on_track=on_track)  #  1/30 o 25 (no hacen falta tantos fps)

        # PROGRESO Y REWARD #
        current_progress = self.track.get_progress(self.car.position.x, self.car.position.y)
        progress_diff = current_progress - self.last_progress # solo si mejora
    
        # meta
        if progress_diff < -self.track.total_length / 2:
            progress_diff += self.track.total_length
        elif progress_diff > self.track.total_length / 2:
            progress_diff -= self.track.total_length

        self.total_ep_prog += progress_diff

        if progress_diff <= 0:
            reward = -0.05 # no avanzar o reversa
        else:
            reward = progress_diff * (0.5 + obs[2] * 0.5)

        # sdf 
        if progress_diff > 0:
            reward += obs[5] * 0.05

        # vuelta completada
        if (current_progress - self.last_progress) < -self.track.total_length / 2:
            reward += 100

        if (action[0] and action[1]) or (action[2] and action[3]):
            reward -= 0.05  # forma no comun de manejar

        if abs(obs[1]) > 0.5 and progress_diff < 0.01:
            reward -= 0.05   # pensalizacion por derrapar y no avanzar

        # Salirse - 5s 
        terminated = False 
        if not on_track: 
            reward -= 0.5
            self.out_track_counter += 1
            if self.out_track_counter > 50:  # 2s
                terminated = True
                reward -= 5.0 
        else:
            self.out_track_counter = 0 
    
        self.last_progress = current_progress
        self.step_count += 1
        truncated = self.step_count > 5000 # Límite de tiempo


        return obs, reward, terminated, truncated, {}


#####################

# Segundo entrenamiento / Tiempo de vuelta 

class TrackEnv2(gym.Env):
    def __init__(self, track_mask):
        super(TrackEnv2, self).__init__()
        self.track = Track(track_mask)
        self.car = None    
        # acciones
        self.action_space = spaces.MultiBinary(5)  # thr # rev # lft # rgt # brk  
        # inputs                          # [Vel_X, Vel_Y, SDF, Error_Angular, 5 Lidars]
        self.observation_space = spaces.Box(low=-1, high=1, shape=(16,), dtype=np.float32)
        # tiempo 
        self.best_lap_time = float("inf")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        spawn = pygame.Vector2(495, 512)
        self.car = Car(x=spawn.x, y=spawn.y, angle=180)
        # progreso
        self.last_progress = self.track.get_progress(spawn.x, spawn.y)
        self.total_ep_prog = 0
        self.step_count = 0
        self.out_track_counter = 0
        # estados de vuelta
        self.current_lap_time = 0
        self.lap_counter = 0
        self.timer_started = False
        self.sector_times = []
        self.lap_rewards = 0 ##

        return get_observation(self.car, self.track), {}
    
    def step(self, action):
        obs = get_observation(self.car, self.track)
        on_track = obs[6]
        prev_pos = pygame.Vector2(self.car.position.x, self.car.position.y)

        dt = 1/25
        self.car.update(action, dt=dt, on_track=on_track)
        curr_pos = self.car.position

        if self.timer_started:
            self.current_lap_time += dt

        # reward 
        current_progress = self.track.get_progress(curr_pos.x, curr_pos.y)
        progress_diff = current_progress - self.last_progress

        # meta
        if progress_diff < -self.track.total_length / 2:
            progress_diff += self.track.total_length
        elif progress_diff > self.track.total_length / 2:
            progress_diff -= self.track.total_length

        # tiempo de vuelta
        if self.timer_started:    
            self.total_ep_prog += progress_diff
            reward = progress_diff * 1.5
            # reward -= 0.01  # tiempo 
        else:
            reward = progress_diff * 0.5  # los primeros metros hasta que cruce la meta
            self.total_ep_prog = 0

        # sectores
        for i, gate in enumerate(self.track.sectors):
            if self.track.check_gate_crossing(prev_pos, curr_pos, gate):
                if self.timer_started:
                    sector_time = self.current_lap_time 
                    self.sector_times.append(sector_time)
                    reward += 5.0

        if (action[2] and action[3]): # izq y der
            reward -= 0.1  # forma no comun de manejar

        terminated = False 
        info = {}
        if self.track.check_finish_crossing(prev_pos, curr_pos):
            if not self.timer_started: # empieza la vuelta cronometrada
                self.timer_started = True
                self.current_lap_time = 0
                self.sector_times = []
                reward += 10
            else:  
                self.lap_counter +=1
                lap_reward = 60 / (self.current_lap_time + 1) # premio por hacerla rapido
                reward += lap_reward
                # print(F"Vuelta terminada, tiempo: {self.current_lap_time:.2f}s")
                info = {"is_lap_completed": True,
                         "Lap_time": self.current_lap_time,
                         "sectors": self.sector_times} 

                if self.current_lap_time < self.best_lap_time:
                    self.best_lap_time = self.current_lap_time
                
                if self.lap_counter > 2:  # primeros metros + warm up + vuelta buena
                    terminated = True
                else:
                    self.current_lap_time = 0
                    self.sector_times = []

        # Salirse
        if not on_track: 
            reward -= 1.0
            self.out_track_counter += 1
            if self.out_track_counter > 50:  # 2s
                terminated = True
                reward -= 10.0 
        else:
            self.out_track_counter = 0 

        self.last_progress = current_progress
        self.step_count += 1
        truncated = self.step_count > 10000

        return obs, reward, terminated, truncated, info