import numpy as np
import gymnasium as gym
from gymnasium import spaces


class InventoryEnv(gym.Env):
    """
    Inventory management environment — tabular-friendly, Q-Learning compatible.

    Each day the agent decides how many units to reorder. Orders arrive the next
    day (lead time = 1). The goal is to meet stochastic customer demand while
    minimising holding costs, stockout penalties, and order costs.

    This is a canonical operations research problem now increasingly solved with
    RL. Amazon, Zalando, and Zara all use ML-assisted inventory systems.

    Observation (4 values — all normalised to [0, 1] or [-1, 1]):
        stock_norm    : inventory / MAX_STOCK
        pending_norm  : pending order / MAX_ORDER_QTY
        demand_norm   : yesterday's demand / MAX_DEMAND
        day_sin       : sin encoding of day of week

    Actions (discrete, 7):
        Order quantity: 0, 5, 10, 15, 20, 25, 30 units

    Reward per day:
        revenue  - holding_cost - stockout_penalty - order_fixed_cost
        revenue          = units_sold × SELL_PRICE
        holding_cost     = stock × 1 €/unit/day
        stockout_penalty = unmet_demand × 30 €/unit
        order_fixed_cost = 20 € if order_qty > 0 else 0
    """

    metadata = {"render_modes": ["human"]}

    MAX_STOCK = 100
    MAX_DEMAND = 25      # units per day (hard ceiling for observation space)
    SELL_PRICE = 50.0    # €/unit
    HOLDING_COST = 1.0   # €/unit/day
    STOCKOUT_COST = 30.0 # €/unit of unmet demand
    ORDER_FIXED = 20.0   # € per order placed (fixed cost, regardless of qty)

    ORDER_QUANTITIES = [0, 5, 10, 15, 20, 25, 30]

    def __init__(self, episode_length: int = 30, seed: int = None):
        super().__init__()
        self.episode_length = episode_length
        self._rng = np.random.default_rng(seed)

        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0,  1.0], dtype=np.float32),
        )
        self.action_space = spaces.Discrete(len(self.ORDER_QUANTITIES))

    # ------------------------------------------------------------------
    # Demand model
    # ------------------------------------------------------------------

    def _sample_demand(self) -> int:
        base = 12
        weekend_boost = 5 if self._dow >= 5 else 0
        d = int(self._rng.normal(base + weekend_boost, 4.0))
        return int(np.clip(d, 0, self.MAX_DEMAND))

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._stock = int(self._rng.integers(20, 60))
        self._pending = 0
        self._last_demand = 12
        self._day = 0
        self._dow = 0

        return self._obs(), {}

    def step(self, action: int):
        order_qty = self.ORDER_QUANTITIES[int(action)]

        # Receive yesterday's order
        self._stock = min(self._stock + self._pending, self.MAX_STOCK)
        self._pending = order_qty  # will arrive tomorrow

        # Realise demand
        demand = self._sample_demand()
        units_sold = min(demand, self._stock)
        unmet = max(0, demand - self._stock)
        self._stock -= units_sold
        self._last_demand = demand

        revenue = units_sold * self.SELL_PRICE
        holding = self._stock * self.HOLDING_COST
        stockout = unmet * self.STOCKOUT_COST
        order_cost = self.ORDER_FIXED if order_qty > 0 else 0.0
        reward = float(revenue - holding - stockout - order_cost)

        self._day += 1
        self._dow = (self._dow + 1) % 7
        terminated = self._day >= self.episode_length

        info = {
            "stock": self._stock,
            "demand": demand,
            "units_sold": units_sold,
            "unmet_demand": unmet,
            "revenue": revenue,
            "holding_cost": holding,
            "stockout_penalty": stockout,
        }
        return self._obs(), reward, terminated, False, info

    def _obs(self) -> np.ndarray:
        return np.array([
            self._stock / self.MAX_STOCK,
            self._pending / max(self.ORDER_QUANTITIES),
            self._last_demand / self.MAX_DEMAND,
            float(np.sin(2.0 * np.pi * self._dow / 7.0)),
        ], dtype=np.float32)

    def render(self):
        dow = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][self._dow]
        print(
            f"Day {self._day:3d} ({dow}) | "
            f"Stock {self._stock:3d} | "
            f"Pending {self._pending:3d} | "
            f"Last demand {self._last_demand:2d}"
        )


class DiscreteInventoryEnv(InventoryEnv):
    """
    Tabular Q-Learning compatible wrapper for InventoryEnv.

    Converts the continuous observation into a single integer state by
    discretizing each dimension into bins, making it directly usable with
    a Q-table (no neural network needed).

    State encoding (4 dimensions → 1 integer):
        stock_bin   : 5 levels  (0-19 | 20-39 | 40-59 | 60-79 | 80+)
        pending_bin : 7 levels  (corresponds to each ORDER_QUANTITIES value)
        demand_bin  : 5 levels  (0-4 | 5-9 | 10-14 | 15-19 | 20+)
        dow_bin     : 2 levels  (weekday=0, weekend=1)

    Total discrete states: 5 × 7 × 5 × 2 = 350
    """

    N_STOCK = 5
    N_PENDING = 7    # matches len(ORDER_QUANTITIES)
    N_DEMAND = 5
    N_DOW = 2
    N_STATES = N_STOCK * N_PENDING * N_DEMAND * N_DOW  # 350

    def __init__(self, episode_length: int = 30, seed: int = None):
        super().__init__(episode_length=episode_length, seed=seed)
        self.observation_space = spaces.Discrete(self.N_STATES)

    def _encode(self) -> int:
        stock_bin = min(self._stock // 20, self.N_STOCK - 1)
        pending_bin = min(self._pending // 5, self.N_PENDING - 1)
        demand_bin = min(self._last_demand // 5, self.N_DEMAND - 1)
        dow_bin = 1 if self._dow >= 5 else 0
        return int(
            stock_bin
            + self.N_STOCK * (
                pending_bin
                + self.N_PENDING * (
                    demand_bin + self.N_DEMAND * dow_bin
                )
            )
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        return self._encode(), {}

    def step(self, action: int):
        _, reward, terminated, truncated, info = super().step(action)
        return self._encode(), reward, terminated, truncated, info
