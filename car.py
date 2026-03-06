# Físicas del juego
import pygame

# config 
max_vel = 150 
min_vel = 30  
rot = 120  
accel = 6 
rev_accel = 2  

brk = 15
drag = 0.02
lateral_drag = 0.05 # -%
grass = 5.0

px_to_m = 0.3

class Car:
    def __init__(self,x,y,angle = 180):
        self.position = pygame.Vector2(x,y)
        self.velocity = pygame.Vector2()
        self.angle = angle
        
        self.max_speed = (max_vel / 3.6) / px_to_m
        self.min_speed = (min_vel / 3.6) / px_to_m
        self.acceleration = accel / px_to_m
        self.reverse_acceleration = rev_accel / px_to_m 
        self.rotation_speed = rot

    def update(self,action,dt,on_track=True):
        "actions: [w,s,a,d,space], [0,1]"

        # control 
        accel_input = action[0]
        reverse_input = action[1]
        left_input = action[2]
        right_input = action[3]
        brake_input = action[4]

        forward = pygame.Vector2(1,0).rotate(-self.angle)
        speed_forward = self.velocity.dot(forward)
        right = pygame.Vector2(0,1).rotate(-self.angle)
        speed_lateral = self.velocity.dot(right)

        # accion
        if accel_input:
            self.velocity += forward * self.acceleration * dt
        
        if reverse_input:
            self.velocity -= forward * self.reverse_acceleration * dt
            if speed_forward < -self.min_speed:
                self.velocity -= forward * (speed_forward + self.min_speed)
        
        if left_input and self.velocity.length() > 2:
            self.angle += self.rotation_speed * dt
        if right_input and self.velocity.length() > 2:
            self.angle -= self.rotation_speed * dt
        
        if brake_input and self.velocity.length() > 0:
            brake_accel = brk / px_to_m
            decel = brake_accel * dt
            if self.velocity.length() > decel:
                self.velocity -= self.velocity.normalize() * decel
            else:
                self.velocity = pygame.Vector2()

        # fricción
        self.velocity -= self.velocity * drag * dt  # drag

        # # fricción lateral
        self.velocity -= right * speed_lateral * lateral_drag

        if not accel_input and not reverse_input:
            rolling = 2.0 / px_to_m 
            decel = rolling * dt
            if self.velocity.length() > decel:
                self.velocity -= self.velocity.normalize() * decel
            else:
                self.velocity = pygame.Vector2()

        if not on_track:  # fuera de pista
            if self.velocity.length() > 0:
                self.velocity -= self.velocity * 5 * dt

        # limite
        if self.velocity.length() > self.max_speed:
            self.velocity.scale_to_length(self.max_speed)

        self.position += self.velocity * dt