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
        pygame.draw.circle(screen, (255, 0, 0), car_pos, 5)

        # ojos
        lidar_angles = [-45, -20, 0, 20, 45]
        for i, rel_angle in enumerate(lidar_angles):
            dist = obs[i+2] * 500 # Des-normalizamos para dibujar
            angle = np.radians(-(car.angle + rel_angle))
            end_x = car.position.x + dist * np.cos(angle)
            end_y = car.position.y + dist * np.sin(angle)
            pygame.draw.line(screen, (255, 0, 0), car_pos, (end_x, end_y), 1)

        pygame.display.flip()
        clock.tick(60) 

        if terminated or truncated:
            print(f"Fin del intento. Recompensa acumulada: {reward}")
            obs, _ = env.reset()

    pygame.quit()

if __name__ == "__main__":
    run_ai_lap()