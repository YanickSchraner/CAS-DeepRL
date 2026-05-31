import numpy as np
import gymnasium as gym
from gymnasium import spaces


class HvacEnv(gym.Env):
    """
    HVAC control environment.

    The agent controls a heater to keep a room at 21°C while minimizing
    electricity cost (which varies by time of day — cheap at night, expensive
    at peak hours). A simple RC thermal model governs temperature dynamics.

    Real-world analog: Google DeepMind used RL to cut data-center cooling
    energy consumption by 40% (2016). Similar approaches are used for smart
    building HVAC in commercial real estate.

    Observation (5 values):
        room_temp    : current room temperature in °C  [10, 35]
        outdoor_temp : outdoor temperature in °C      [-10, 35]
        hour_sin     : sin encoding of hour of day    [-1, 1]
        hour_cos     : cos encoding of hour of day    [-1, 1]
        price        : current electricity price €/kWh [0, 1] normalised

    Actions (discrete, 4):
        0 → Heater OFF    (0.0 kW)
        1 → Heater LOW    (0.5 kW)
        2 → Heater MEDIUM (1.5 kW)
        3 → Heater HIGH   (3.0 kW)

    Reward per step (15 min):
        -(discomfort_penalty + cost_weight * electricity_cost)
        discomfort_penalty = |room_temp - 21°C| * discomfort_weight
        electricity_cost   = heater_power_kW * price_€/kWh * 0.25 h

    The raw per-step electricity cost (a few cents) is tiny next to the
    discomfort penalty, so without scaling the agent has almost no incentive
    to shift heating into cheap hours. `cost_weight` (default 10) lifts the
    cost term into the same order of magnitude as discomfort, making
    DeepMind-style load-shifting (pre-heat when cheap, coast through the price
    peak) a behaviour worth learning. `info["cost_eur"]` always reports the
    *raw* monetary cost in euros, regardless of `cost_weight`.
    """

    metadata = {"render_modes": ["human"]}

    HEATER_POWERS = [0.0, 0.5, 1.5, 3.0]   # kW
    TARGET_TEMP = 21.0                        # °C
    STEPS_PER_DAY = 96                        # 24h × 4 steps/h (15-min steps)

    # Thermal model constants
    _R = 5.0    # °C / kW  — building thermal resistance
    _C = 10.0   # kWh / °C — building thermal mass
    _DT = 0.25  # hours    — step duration

    def __init__(self, discomfort_weight: float = 2.0, cost_weight: float = 10.0,
                 episode_days: int = 1, seed: int = None):
        super().__init__()
        self.discomfort_weight = discomfort_weight
        self.cost_weight = cost_weight
        self.episode_length = episode_days * self.STEPS_PER_DAY
        self._rng = np.random.default_rng(seed)

        self.observation_space = spaces.Box(
            low=np.array([10.0, -10.0, -1.0, -1.0, 0.0], dtype=np.float32),
            high=np.array([35.0,  35.0,  1.0,  1.0, 1.0], dtype=np.float32),
        )
        self.action_space = spaces.Discrete(4)

    # ------------------------------------------------------------------
    # Environment dynamics helpers
    # ------------------------------------------------------------------

    def _electricity_price(self, hour: float) -> float:
        """Time-of-use price in €/kWh. Cheap at night, peak in early evening."""
        base = 0.10
        if 6.0 <= hour <= 21.0:
            return base + 0.18 * np.sin(np.pi * (hour - 6.0) / 15.0) ** 2
        return base

    def _outdoor_temp(self, hour: float) -> float:
        """Simple sinusoidal daily temperature cycle."""
        return self._base_outdoor + 5.0 * np.sin(np.pi * (hour - 6.0) / 12.0 - np.pi / 2.0)

    def _thermal_step(self, t_room: float, t_out: float, power: float) -> float:
        """RC thermal model with small Gaussian noise."""
        heat_loss = (t_room - t_out) / self._R
        dt = (power - heat_loss) * self._DT / self._C
        return t_room + dt + self._rng.normal(0.0, 0.05)

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._base_outdoor = float(self._rng.uniform(-5.0, 10.0))
        self._room_temp = float(self._rng.uniform(16.0, 26.0))
        self._step = 0
        self._hour = 0.0

        return self._obs(), {}

    def step(self, action: int):
        power = self.HEATER_POWERS[int(action)]
        price = self._electricity_price(self._hour)
        t_out = self._outdoor_temp(self._hour)

        discomfort = abs(self._room_temp - self.TARGET_TEMP) * self.discomfort_weight
        cost = power * price * self._DT
        reward = float(-(discomfort + self.cost_weight * cost))

        self._room_temp = float(np.clip(self._thermal_step(self._room_temp, t_out, power), 10.0, 35.0))
        self._step += 1
        self._hour = (self._hour + self._DT) % 24.0

        terminated = self._step >= self.episode_length
        info = {
            "room_temp": self._room_temp,
            "outdoor_temp": t_out,
            "heater_kw": power,
            "price_eur_kwh": price,
            "discomfort": discomfort,
            "cost_eur": cost,
        }
        return self._obs(), reward, terminated, False, info

    def _obs(self) -> np.ndarray:
        price_norm = (self._electricity_price(self._hour) - 0.10) / 0.18
        t_out = self._outdoor_temp(self._hour)
        return np.array([
            self._room_temp,
            t_out,
            np.sin(2.0 * np.pi * self._hour / 24.0),
            np.cos(2.0 * np.pi * self._hour / 24.0),
            float(np.clip(price_norm, 0.0, 1.0)),
        ], dtype=np.float32)

    def render(self):
        hour_int = int(self._hour)
        price = self._electricity_price(self._hour)
        t_out = self._outdoor_temp(self._hour)
        action_names = ["OFF", "LOW", "MED", "HIGH"]
        print(
            f"Hour {hour_int:02d}:00 | "
            f"Room {self._room_temp:5.1f}°C | "
            f"Outdoor {t_out:5.1f}°C | "
            f"Price €{price:.2f}/kWh"
        )
