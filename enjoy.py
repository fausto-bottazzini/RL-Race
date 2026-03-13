import pygame
import numpy as np
from stable_baselines3 import PPO
from env import TrackEnv, TrackEnv2

pygame.init()
WIDHT, HEIGHT = 1080, 720
screen = pygame.display.set_mode((WIDHT,HEIGHT))
clock = pygame.time.Clock()

env = TrackEnv2(track_mask="assets/track_1-mask.png")
# Selecconar modelo
model = PPO.load("data/models/ppo_T2")
# model = PPO.load("data/models/ppo_track_v5_2891680_steps")  

track_img = pygame.image.load("assets/track_1-mask.png").convert()

def run_ai_lap():
    obs,_ = env.reset()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        action, _states = model.predict(obs, deterministic=True) # el modelo decide
        # action, _states = model.predict(obs, deterministic=False) # sigue probando cosas
        obs, reward, terminated, truncated, info = env.step(action) # evol

        # render
        screen.fill((30,30,30))
        screen.blit(track_img, (0,0))
        car = env.car
        car_pos = (int(car.position.x), int(car.position.y))

        # auto
        pygame.draw.circle(screen, (255, 0, 0), car_pos, 5)

        # futuro
        current_prog = env.track.get_progress(car.position.x, car.position.y)
        p_future = env.track.get_point_at_dist(current_prog + 150)
        pygame.draw.circle(screen, (0, 255, 0), (int(p_future.x), int(p_future.y)), 4) # Punto verde

        # ojos
        lidar_angles = [-90, -45, -20, -10, 0, 10, 20, 45, 90]
        for i, rel_angle in enumerate(lidar_angles):
            dist = obs[i+7] * 500 # Desnormalizar para dibujar
            angle = np.radians(-(car.angle + rel_angle))
            end_x = car.position.x + dist * np.cos(angle)
            end_y = car.position.y + dist * np.sin(angle)
            pygame.draw.line(screen, (255, 0, 0), car.position, (end_x, end_y), 1)

        right_indicator = car.position + pygame.Vector2(0,5).rotate(-car.angle)
        front_indicator = car.position + pygame.Vector2(10,0).rotate(-car.angle)
        pygame.draw.line(screen, (0, 0, 0), car.position, (front_indicator.x, front_indicator.y), 2)
        pygame.draw.line(screen, (0, 0, 0), car.position, (right_indicator.x, right_indicator.y), 2)

        pygame.display.flip()
        clock.tick(60) 

        if terminated or truncated:
            print(f"Fin del intento. Recompensa acumulada: {env.total_ep_prog}")   
            obs, _ = env.reset()

        env.track.record_telemetry(env.step_count, [bool(a) for a in action], car)

    env.track.export_telemetry("data/last_run.csv") 
    pygame.quit()

if __name__ == "__main__":
    run_ai_lap()