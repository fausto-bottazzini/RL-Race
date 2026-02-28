# Función recompensa

reward = + progreso_delta * 100         # premio por progreso
         - abs(lateral_distance) * 2    # penalización por estar lejos del centro
         - 0.01                         # penalización por tiempo

reward = 0
reward += delta_progress * 5    # progreso
reward += velocity * 0.1        # velocidad
reward -= offtrack * 2          # salirse de pista
reward -= abs(lateral_error) * 0.5  # centerline

if crashed:
    reward -= 50
    done = True

reward += 100 # sector completo
reward += max(0, target_time - lap_time) * 50  # completar vuelta
