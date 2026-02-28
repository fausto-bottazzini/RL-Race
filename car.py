# Físicas del juego
import pygame

class CarPhysics:
    def __init__(self,x,y,angle = 180):
        self.position = pygame.Vector2(x,y)
        self.velocity = pygame.Vector2()
        self.angle = angle
        
        self.max_speed = (150 / 3.6) / 0.03
        self.min_speed = (30 / 3.6) / 0.03
        self.acceleration = 6 / 0.03
        self.reverse_acceleration = 2 / 0.03 
        self.rotation_speed = 120
        
    def update(self,action,dt,track):
        "actions: [w,s,a,d,space], [0,1]"

        # control 
        accel_input = action[0]
        reverse_input = action[1]
        left_input = action[2]
        right_input = action[3]
        brake_input = action[4]

        forward = pygame.Vector2(1,0).rotate(-self.angle)
        speed_forward = self.velocity.dot(forward)

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
            brake_accel = 15 / 0.03
            decel = brake_accel * dt
            if self.velocity.length() > decel:
                self.velocity -= self.velocity.normalize() * decel
            else:
                self.velocity = pygame.Vector2()
        
        # fricción
        self.velocity -= self.velocity * 0.02 * dt  # drag

        if not accel_input and not reverse_input:
            rolling = 2.0 / 0.03
            decel = rolling * dt
            if self.velocity.length() > decel:
                self.velocity -= self.velocity.normalize() * decel
            else:
                self.velocity = pygame.Vector2()

        if not track.is_inside(self.position.x, self.position.y):
            if self.velocity.length() > 0:
                self.velocity -= self.velocity.normalize() * 5 * dt

        # limite
        if self.velocity.length() > self.max_speed:
            self.velocity.scale_to_length(self.max_speed)

        self.position += self.velocity * dt