import numpy as np
import gymnasium as gym
from gymnasium import spaces


class PricingEnv(gym.Env):
    """
    Dynamic pricing environment.

    The agent sets the price for a product each day, balancing revenue against
    stockout risk and excess inventory. Demand responds to price elasticity,
    competitor pricing, and weekday patterns.

    Real-world analogs:
    - Amazon reprices millions of products multiple times per day.
    - Airlines use RL-assisted revenue management to fill seats profitably.
    - Ride-hailing apps apply surge pricing in real time.

    Observation (5 values):
        inventory_frac   : stock as fraction of capacity     [0, 1]
        competitor_norm  : competitor price, normalised      [0, 1]
        day_sin          : sin encoding of day of week       [-1, 1]
        day_cos          : cos encoding of day of week       [-1, 1]
        demand_signal    : EMA of recent daily sales / max   [0, 1]

    Actions (discrete, 4):
        0 → BUDGET   — 70 % of base price
        1 → STANDARD — 100 % of base price
        2 → PREMIUM  — 130 % of base price
        3 → LUXURY   — 170 % of base price

    Demand elasticity is regime-dependent: weekday shoppers are very
    price-sensitive (elasticity ≈ -2.2), weekend shoppers far less so
    (≈ -0.7). This makes the profit-maximising price swing with the day of
    week, so no single fixed price is optimal and a state-aware policy has
    real headroom over the fixed baselines.

    Reward per day:
        revenue - stockout_penalty - holding_cost
        revenue          = units_sold × price
        stockout_penalty = unmet_demand × BASE_PRICE × 0.25
        holding_cost     = inventory × 0.5 €/unit/day
    """

    metadata = {"render_modes": ["human"]}

    BASE_PRICE = 100.0
    PRICE_MULT = [0.70, 1.00, 1.30, 1.70]
    MAX_INVENTORY = 100
    MAX_DAILY_DEMAND = 20   # at base price, typical weekday

    def __init__(self, episode_length: int = 30, seed: int = None):
        super().__init__()
        self.episode_length = episode_length
        self._rng = np.random.default_rng(seed)

        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, -1.0, -1.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0,  1.0,  1.0, 1.0], dtype=np.float32),
        )
        self.action_space = spaces.Discrete(4)

    # ------------------------------------------------------------------
    # Demand model
    # ------------------------------------------------------------------

    def _demand(self, price: float, competitor_price: float, day_of_week: int) -> float:
        """
        Stochastic demand with price elasticity, competitor effect, and weekday pattern.

        Elasticity is regime-dependent: weekdays are highly elastic (-2.2, bargain
        hunters), weekends are nearly inelastic (-0.7, leisure buyers who will pay
        a premium). This is what makes the optimal price swing by day of week, so a
        state-aware policy can beat any single fixed price.
        """
        elasticity = -0.7 if day_of_week >= 5 else -2.2
        price_ratio = price / self.BASE_PRICE
        comp_effect = 0.35 * (competitor_price - price) / self.BASE_PRICE
        weekend_boost = 1.5 if day_of_week >= 5 else 1.0

        mean = self.MAX_DAILY_DEMAND * (price_ratio ** elasticity) * weekend_boost * (1.0 + comp_effect)
        mean = float(np.clip(mean, 0.0, self.MAX_DAILY_DEMAND * 3))
        # Lower observation noise (10% vs 20%) raises the signal-to-noise ratio so
        # the state-dependent pricing strategy is actually learnable.
        return max(0.0, float(self._rng.normal(mean, mean * 0.10 + 1e-6)))

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._inventory = int(self._rng.integers(20, 80))
        self._day = 0
        self._dow = 0   # day of week (0 = Monday)
        self._competitor_price = self.BASE_PRICE * float(self._rng.uniform(0.85, 1.25))
        self._demand_signal = 0.5

        return self._obs(), {}

    def step(self, action: int):
        price = self.BASE_PRICE * self.PRICE_MULT[int(action)]

        demand = self._demand(price, self._competitor_price, self._dow)
        units_sold = min(int(demand), self._inventory)
        unmet = max(0.0, demand - self._inventory)

        revenue = units_sold * price
        stockout_penalty = unmet * self.BASE_PRICE * 0.25
        holding_cost = self._inventory * 0.5
        reward = float(revenue - stockout_penalty - holding_cost)

        # Update inventory (simple restock when running low). The restock is
        # deliberately modest so that pricing too cheap genuinely risks a stockout
        # — this is what makes inventory-aware pricing pay off.
        self._inventory = max(0, self._inventory - units_sold)
        if self._inventory < 20:
            self._inventory = min(self._inventory + int(self._rng.integers(10, 25)), self.MAX_INVENTORY)

        # Competitor price random walk ±5 % per day
        self._competitor_price *= float(self._rng.uniform(0.95, 1.05))
        self._competitor_price = float(np.clip(
            self._competitor_price,
            self.BASE_PRICE * 0.6,
            self.BASE_PRICE * 1.6,
        ))

        # EMA demand signal
        self._demand_signal = 0.8 * self._demand_signal + 0.2 * (units_sold / self.MAX_DAILY_DEMAND)

        self._day += 1
        self._dow = (self._dow + 1) % 7
        terminated = self._day >= self.episode_length

        info = {
            "price": price,
            "units_sold": units_sold,
            "unmet_demand": unmet,
            "revenue": revenue,
            "stockout_penalty": stockout_penalty,
            "holding_cost": holding_cost,
            "inventory": self._inventory,
            "competitor_price": self._competitor_price,
        }
        return self._obs(), reward, terminated, False, info

    def _obs(self) -> np.ndarray:
        comp_norm = float(np.clip(
            (self._competitor_price - self.BASE_PRICE * 0.6) / (self.BASE_PRICE * 1.0),
            0.0, 1.0
        ))
        return np.array([
            self._inventory / self.MAX_INVENTORY,
            comp_norm,
            np.sin(2.0 * np.pi * self._dow / 7.0),
            np.cos(2.0 * np.pi * self._dow / 7.0),
            float(np.clip(self._demand_signal, 0.0, 1.0)),
        ], dtype=np.float32)

    def render(self):
        dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        print(
            f"Day {self._day:3d} ({dow_names[self._dow]}) | "
            f"Stock {self._inventory:3d} | "
            f"Competitor €{self._competitor_price:.0f} | "
            f"Demand EMA {self._demand_signal:.2f}"
        )
