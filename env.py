# Entorno 

class CarRL(gym.Env):
    def __init__(self):
    def reset(self):
    def step(self, action):
    def _get_obs(self)
        
class RacingEnv:
    def __init__(self, track_mask):
        self.track = Track(track_mask)
        self.car = Car(...)  # misma física del juego

        self.dt = 1/60
        self.reset()

    def reset(self):
        self.car.reset_to_start()
        self.prev_pos = self.car.pos.copy()
        self.current_gate = 0
        self.lap_time = 0
        self.sector_times = []
        self.telemetry = {...}
        return self._get_obs()

    def step(self, action):

        inputs = self._decode_action(action)
        self.car.update(inputs, self.dt)

        self.lap_time += self.dt

        self._check_collisions()
        self._check_gates()
        self._update_progress()

        self._record_telemetry()

        obs = self._get_obs()
        reward = 0  # todavía no entrenamos

        done = False

        return obs, reward, done, {}