import pygame
import numpy as np
from stable_baselines3 import PPO
from env import TrackEnv

pygame.init()
WIDHT, HEIGHT = 1080, 720
screen = pygame.display.set_mode((WIDHT,HEIGHT))
clock = pygame.time.Clock()

env = TrackEnv(track_mask="assets/track_1-mask.png")
model = PPO.load("data/ppo_track_agent")

track_img = pygame.image.load("assets/track_1-mask.png").convert()

def run_ai_lap():
    obs,_ = env.reset()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        action, _states = model.predict(obs, deterministic=True) # el modelo decide
        obs, reward, terminated, truncated, info = env.step(action) # evol

        # render
        screen.fill((30,30,30))
        screen.blit(track_img, (0,0))
        car = env.car
        car_pos = (int(car.position.x), int(car.position.y))

        # auto
        pygame.draw.circle(screen, (255, 0, 0), car_pos, 5)
        front_indicator = car.position + pygame.Vector2(10, 5).rotate(-car.angle)
        pygame.draw.line(screen, (0, 0, 0), car_pos, (front_indicator.x, front_indicator.y), 2)
        # futuro
        current_prog = env.track.get_progress(car.position.x, car.position.y)
        p_future = env.track.get_point_at_dist(current_prog + 150)
        pygame.draw.circle(screen, (0, 255, 0), (int(p_future.x), int(p_future.y)), 4) # Punto verde

        # ojos
        lidar_angles = [-90, -45, -20, -10, 0, 10, 20, 45, 90]
        for i, rel_angle in enumerate(lidar_angles):
            dist = obs[i+7] * 500 # Des-normalizamos para dibujar
            angle = np.radians(-(car.angle + rel_angle))
            start_x = car.position.x + 10 * np.cos(np.radians(-car.angle))
            start_y = car.position.y + 10 * np.sin(np.radians(-car.angle))
            end_x = start_x + dist * np.cos(angle)
            end_y = start_y + dist * np.sin(angle)
            pygame.draw.line(screen, (255, 0, 0), (start_x, start_y), (end_x, end_y), 1)

        pygame.display.flip()
        clock.tick(60) 

        if terminated or truncated:
            print(f"Fin del intento. Recompensa acumulada: {reward}")
            obs, _ = env.reset()

        env.track.record_telemetry(env.step_count, [bool(a) for a in action], car)

    env.track.export_telemetry("data/last_run.csv") 
    pygame.quit()

if __name__ == "__main__":
    run_ai_lap()