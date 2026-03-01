# Cheuqueo del juego y telemetria 

import pygame
from car import Car
from track import Track

import numpy as np
from env import get_eyes

track = Track("assets/track_1-mask.png")
car = Car(x=475,y=512) # 495, 512

# config 
pygame.init()
screen = pygame.display.set_mode((1080,720))
clock = pygame.time.Clock()
track_image = pygame.image.load("assets/track_1-mask.png").convert()

running = True
step_count = 0
while running:
    dt = clock.tick(60)/1000

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            print(f"Progreso: {track.get_progress(car.position.x,car.position.y)}")
            running = False

    keys = pygame.key.get_pressed()
    action = [keys[pygame.K_w], keys[pygame.K_s], keys[pygame.K_a], keys[pygame.K_d], keys[pygame.K_SPACE]]

    if keys[pygame.K_ESCAPE]:
        print(f"Progreso: {track.get_progress(car.position.x,car.position.y)}")
        running = False

    prev_pos = pygame.Vector2(car.position.x,car.position.y)
    
    on_track = track.is_inside(car.position.x,car.position.y)

    car.update(action,dt,on_track)
    curr_pos = car.position

    # telemetria
    step_count += 1
    track.record_telemetry(step_count, action, car)

    # sectores / meta
    for i in range(len(track.sectors)):
        if track.check_gate_crossing(prev_pos,curr_pos, track.sectors[i]):
            print(f"sector {i}")

    if track.check_finish_crossing(prev_pos,curr_pos):
        print(f"Vuelta Completada. Progreso Total: {track.get_progress(car.position.x,car.position.y)}")

    # render
    screen.fill((30,30,30))
    screen.blit(track_image, (0,0))
    pygame.draw.circle(screen,(255,0,0),(int(car.position.x),int(car.position.y)),5)
    for rel_angle in [-45, -20, 0, 20, 45]:
        total_angle = -(car.angle + rel_angle)
        dist = get_eyes(track, car.position, total_angle) * 500
        end_x = car.position.x + dist * np.cos(np.radians(total_angle))
        end_y = car.position.y + dist * np.sin(np.radians(total_angle))
        pygame.draw.line(screen, (255, 0, 0), (car.position.x, car.position.y), (end_x, end_y), 2)
    pygame.display.flip()

track.export_telemetry("data/last_run.csv")  # guardado
pygame.quit()
    