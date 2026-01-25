"""
Hybrid Two-Stage Framework for VRPTW
Stage 1: Attention-based Double DQN for initial solution construction
Stage 2: Fast-ALNS for solution refinement
Reference: He et al. (2021) arXiv:2103.05847
"""

import subprocess
subprocess.run(["pip", "install", "-q", "torch", "numpy", "pandas", "tqdm", "kagglehub", "matplotlib", "safetensors"], check=True)

import os
import math
import random
import time as time_module
import shutil
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

# ==============================================================================
# CONFIG
# ==============================================================================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

DATA_DIR = "/root/data/Solomon"
MODEL_DIR = "/root/models"
MODEL_PATH = "/root/models/dqn_model.safetensors"
OUTPUT_DIR = "/root/output"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

EMBED_DIM = 128
NUM_HEADS = 8
NUM_LAYERS = 3
LEARNING_RATE = 1e-4
GAMMA = 0.99
BUFFER_SIZE = 20000
BATCH_SIZE = 256
TARGET_UPDATE = 50
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.997
BETA_VEHICLE = 5.0
EVAL_TIME_LIMIT = 30.0

# Fast training: single phase, 100 customers, 800 episodes
CURRICULUM = [
    {"customers": 100, "episodes": 800},
]

# ==============================================================================
# DATA
# ==============================================================================

@dataclass
class Customer:
    id: int
    x: float
    y: float
    demand: float
    ready_time: float
    due_date: float
    service_time: float


@dataclass
class VRPTWInstance:
    depot: Customer
    customers: List[Customer]
    vehicle_capacity: float
    num_vehicles: int
    max_coord: float = 100.0
    max_time: float = 1000.0
    max_demand: float = 100.0


def download_kaggle_data():
    if os.path.exists(DATA_DIR) and len(os.listdir(DATA_DIR)) > 0:
        print(f"Data exists at {DATA_DIR}")
        return DATA_DIR
    try:
        import kagglehub
        path = kagglehub.dataset_download("senju14/vrptw-benchmark-datasets")
        print(f"Downloaded to: {path}")
        solomon_src = None
        for root, dirs, files in os.walk(path):
            if "Solomon" in dirs:
                solomon_src = os.path.join(root, "Solomon")
                break
        if solomon_src is None:
            for root, dirs, files in os.walk(path):
                if any(f.lower().startswith("rc") and f.endswith(".txt") for f in files):
                    solomon_src = root
                    break
        if solomon_src:
            if os.path.exists(DATA_DIR):
                shutil.rmtree(DATA_DIR)
            shutil.copytree(solomon_src, DATA_DIR)
            print(f"Copied to {DATA_DIR}")
            return DATA_DIR
        return path
    except Exception as e:
        print(f"Download failed: {e}")
        return None


def parse_solomon_file(filepath: str) -> VRPTWInstance:
    with open(filepath, 'r') as f:
        lines = f.readlines()
    vehicle_line = lines[4].split()
    num_vehicles = int(vehicle_line[0])
    vehicle_capacity = float(vehicle_line[1])
    customers = []
    for line in lines[9:]:
        parts = line.split()
        if len(parts) >= 7:
            cust = Customer(
                id=int(parts[0]),
                x=float(parts[1]),
                y=float(parts[2]),
                demand=float(parts[3]),
                ready_time=float(parts[4]),
                due_date=float(parts[5]),
                service_time=float(parts[6])
            )
            customers.append(cust)
    depot = customers[0]
    customers = customers[1:]
    max_coord = max(max(c.x for c in customers + [depot]), max(c.y for c in customers + [depot]))
    max_time = max(c.due_date for c in customers + [depot])
    max_demand = max(c.demand for c in customers) if customers else 1.0
    return VRPTWInstance(
        depot=depot, customers=customers, vehicle_capacity=vehicle_capacity,
        num_vehicles=num_vehicles, max_coord=max_coord, max_time=max_time,
        max_demand=max(max_demand, 1.0)
    )


def generate_synthetic_instance(num_customers: int, clustered: bool = False, rc1_style: bool = True) -> VRPTWInstance:
    max_coord = 100.0
    if rc1_style:
        vehicle_capacity, max_time, num_vehicles = 200.0, 240.0, 25
    else:
        vehicle_capacity, max_time, num_vehicles = 1000.0, 480.0, 10
    
    depot = Customer(id=0, x=max_coord/2, y=max_coord/2, demand=0.0, ready_time=0.0, due_date=max_time, service_time=0.0)
    customers = []
    
    if clustered:
        num_clusters = random.randint(3, 8)
        centers = [(random.uniform(10, max_coord-10), random.uniform(10, max_coord-10)) for _ in range(num_clusters)]
        for i in range(1, num_customers + 1):
            cx, cy = random.choice(centers)
            x = np.clip(np.random.normal(cx, 15), 0, max_coord)
            y = np.clip(np.random.normal(cy, 15), 0, max_coord)
            demand = random.uniform(5, 40)
            ready_time = random.uniform(0, max_time * 0.6)
            window = random.uniform(30, 100) if rc1_style else random.uniform(50, 200)
            due_date = min(ready_time + window, max_time)
            customers.append(Customer(id=i, x=float(x), y=float(y), demand=demand, ready_time=ready_time, due_date=due_date, service_time=random.uniform(5, 15)))
    else:
        for i in range(1, num_customers + 1):
            x, y = random.uniform(0, max_coord), random.uniform(0, max_coord)
            demand = random.uniform(5, 40)
            ready_time = random.uniform(0, max_time * 0.6)
            window = random.uniform(30, 100) if rc1_style else random.uniform(50, 200)
            due_date = min(ready_time + window, max_time)
            customers.append(Customer(id=i, x=x, y=y, demand=demand, ready_time=ready_time, due_date=due_date, service_time=random.uniform(5, 15)))
    
    return VRPTWInstance(depot=depot, customers=customers, vehicle_capacity=vehicle_capacity, num_vehicles=num_vehicles, max_coord=max_coord, max_time=max_time, max_demand=40.0)


def generate_curriculum_dataset(phase: Dict) -> List[VRPTWInstance]:
    instances = []
    for i in range(500):
        clustered = i >= 250
        rc1_style = random.random() < 0.5
        instances.append(generate_synthetic_instance(phase["customers"], clustered, rc1_style))
    return instances


def distance(c1: Customer, c2: Customer) -> float:
    return math.sqrt((c1.x - c2.x)**2 + (c1.y - c2.y)**2)


def normalize_instance(instance: VRPTWInstance) -> Tuple[np.ndarray, float, float]:
    n = len(instance.customers)
    features = np.zeros((n, 5), dtype=np.float32)
    for i, c in enumerate(instance.customers):
        features[i] = [c.x/instance.max_coord, c.y/instance.max_coord, c.demand/instance.max_demand, c.ready_time/instance.max_time, c.due_date/instance.max_time]
    return features, instance.depot.x/instance.max_coord, instance.depot.y/instance.max_coord

# ==============================================================================
# MODEL
# ==============================================================================

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int = 8):
        super().__init__()
        self.embed_dim, self.num_heads = embed_dim, num_heads
        self.head_dim = embed_dim // num_heads
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, S, _ = x.shape
        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        attn = torch.nan_to_num(F.softmax(scores, dim=-1), nan=0.0)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, S, self.embed_dim)
        return self.out_proj(out)


class GraphEncoder(nn.Module):
    def __init__(self, input_dim: int = 5, embed_dim: int = 128, num_heads: int = 8, num_layers: int = 3):
        super().__init__()
        self.linear_proj = nn.Linear(input_dim, embed_dim)
        self.layers = nn.ModuleList([nn.ModuleDict({
            'attn': MultiHeadSelfAttention(embed_dim, num_heads),
            'norm1': nn.LayerNorm(embed_dim),
            'ff': nn.Sequential(nn.Linear(embed_dim, embed_dim*4), nn.ReLU(), nn.Linear(embed_dim*4, embed_dim)),
            'norm2': nn.LayerNorm(embed_dim)
        }) for _ in range(num_layers)])

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.linear_proj(x)
        for layer in self.layers:
            x = layer['norm1'](x + layer['attn'](x, mask))
            x = layer['norm2'](x + layer['ff'](x))
        return x


class BahdanauAttentionDecoder(nn.Module):
    def __init__(self, state_dim: int = 3, embed_dim: int = 128):
        super().__init__()
        self.state_proj = nn.Linear(state_dim, embed_dim)
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v = nn.Linear(embed_dim, 1, bias=False)

    def forward(self, state: torch.Tensor, node_embeddings: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        query = self.state_proj(state).unsqueeze(1)
        scores = self.v(torch.tanh(self.W_q(query) + self.W_k(node_embeddings))).squeeze(-1)
        return scores.masked_fill(mask, float('-inf'))


class AttentionDQN(nn.Module):
    def __init__(self, input_dim: int = 5, state_dim: int = 3, embed_dim: int = 128, num_heads: int = 8, num_layers: int = 3):
        super().__init__()
        self.encoder = GraphEncoder(input_dim, embed_dim, num_heads, num_layers)
        self.decoder = BahdanauAttentionDecoder(state_dim, embed_dim)

    def forward(self, customer_features: torch.Tensor, state: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.decoder(state, self.encoder(customer_features, mask), mask)

# ==============================================================================
# ENVIRONMENT
# ==============================================================================

class VRPTWEnv:
    def __init__(self, instance: VRPTWInstance):
        self.instance = instance
        self.reset()

    def reset(self) -> Dict:
        self.current_vehicle, self.current_time, self.current_load = 0, 0.0, 0.0
        self.current_location = self.instance.depot
        self.unvisited = set(range(len(self.instance.customers)))
        self.routes = [[] for _ in range(self.instance.num_vehicles)]
        self.total_distance, self.vehicles_used = 0.0, 1
        return self._get_state()

    def _get_state(self) -> Dict:
        features, _, _ = normalize_instance(self.instance)
        mask = np.zeros(len(self.instance.customers), dtype=bool)
        for i in range(len(self.instance.customers)):
            if i not in self.unvisited:
                mask[i] = True
            else:
                c = self.instance.customers[i]
                arrival = max(self.current_time + distance(self.current_location, c), c.ready_time)
                if arrival > c.due_date or self.current_load + c.demand > self.instance.vehicle_capacity:
                    mask[i] = True
        state_vec = np.array([self.current_location.x/self.instance.max_coord, self.current_load/self.instance.vehicle_capacity, self.current_time/self.instance.max_time], dtype=np.float32)
        return {'customer_features': features, 'state': state_vec, 'mask': mask}

    def step(self, action: int) -> Tuple[Dict, float, bool]:
        if action not in self.unvisited:
            return self._get_state(), -100.0, True
        customer = self.instance.customers[action]
        travel_dist = distance(self.current_location, customer)
        arrival = max(self.current_time + travel_dist, customer.ready_time)
        if arrival > customer.due_date or self.current_load + customer.demand > self.instance.vehicle_capacity:
            return self._get_state(), -100.0, True

        self.total_distance += travel_dist
        self.current_time = arrival + customer.service_time
        self.current_load += customer.demand
        self.current_location = customer
        self.unvisited.remove(action)
        self.routes[self.current_vehicle].append(action)
        reward = -travel_dist / self.instance.max_coord

        if len(self.unvisited) == 0:
            self.total_distance += distance(self.current_location, self.instance.depot)
            return self._get_state(), reward - distance(self.current_location, self.instance.depot)/self.instance.max_coord, True

        feasible = any(max(self.current_time + distance(self.current_location, self.instance.customers[i]), self.instance.customers[i].ready_time) <= self.instance.customers[i].due_date and self.current_load + self.instance.customers[i].demand <= self.instance.vehicle_capacity for i in self.unvisited)
        if not feasible:
            self.total_distance += distance(self.current_location, self.instance.depot)
            self.current_vehicle += 1
            self.vehicles_used += 1
            reward -= BETA_VEHICLE / self.instance.max_coord
            if self.current_vehicle >= self.instance.num_vehicles:
                return self._get_state(), -100.0, True
            self.current_time, self.current_load, self.current_location = 0.0, 0.0, self.instance.depot
        return self._get_state(), reward, False

# ==============================================================================
# AGENT
# ==============================================================================

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    def sample(self, batch_size: int):
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(self, device: torch.device):
        self.device, self.gamma, self.batch_size, self.target_update = device, GAMMA, BATCH_SIZE, TARGET_UPDATE
        self.policy_net = AttentionDQN(5, 3, EMBED_DIM, NUM_HEADS, NUM_LAYERS).to(device)
        self.target_net = AttentionDQN(5, 3, EMBED_DIM, NUM_HEADS, NUM_LAYERS).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.buffer = ReplayBuffer(BUFFER_SIZE)
        self.steps, self.epsilon = 0, 1.0

    def select_action(self, state: Dict, training: bool = True) -> int:
        mask = state['mask']
        valid = np.where(~mask)[0]
        if len(valid) == 0:
            return -1
        if training and random.random() < self.epsilon:
            return random.choice(valid)
        with torch.no_grad():
            cf = torch.tensor(state['customer_features'], dtype=torch.float32).unsqueeze(0).to(self.device)
            sv = torch.tensor(state['state'], dtype=torch.float32).unsqueeze(0).to(self.device)
            m = torch.tensor(mask, dtype=torch.bool).unsqueeze(0).to(self.device)
            return self.policy_net(cf, sv, m).argmax(dim=-1).item()

    def update(self) -> float:
        if len(self.buffer) < self.batch_size:
            return 0.0
        batch = self.buffer.sample(self.batch_size)
        curr_cf, curr_sv, curr_m, next_cf, next_sv, next_m, actions, rewards, dones = [], [], [], [], [], [], [], [], []
        for s, a, r, ns, d in batch:
            curr_cf.append(s['customer_features']); curr_sv.append(s['state']); curr_m.append(s['mask'])
            next_cf.append(ns['customer_features']); next_sv.append(ns['state']); next_m.append(ns['mask'])
            actions.append(a); rewards.append(r); dones.append(d)
        
        cf_t = torch.tensor(np.array(curr_cf), dtype=torch.float32).to(self.device)
        sv_t = torch.tensor(np.array(curr_sv), dtype=torch.float32).to(self.device)
        m_t = torch.tensor(np.array(curr_m), dtype=torch.bool).to(self.device)
        a_t = torch.tensor(actions, dtype=torch.long).to(self.device)
        r_t = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        d_t = torch.tensor(dones, dtype=torch.bool).to(self.device)
        ncf_t = torch.tensor(np.array(next_cf), dtype=torch.float32).to(self.device)
        nsv_t = torch.tensor(np.array(next_sv), dtype=torch.float32).to(self.device)
        nm_t = torch.tensor(np.array(next_m), dtype=torch.bool).to(self.device)
        
        q_vals = self.policy_net(cf_t, sv_t, m_t).gather(1, a_t.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q_policy = self.policy_net(ncf_t, nsv_t, nm_t)
            next_q_policy[nm_t] = float('-inf')
            best_actions = next_q_policy.argmax(dim=-1)
            next_q_target = self.target_net(ncf_t, nsv_t, nm_t)
            max_next_q = torch.nan_to_num(next_q_target.gather(1, best_actions.unsqueeze(1)).squeeze(1), nan=0.0, posinf=0.0, neginf=0.0)
            max_next_q[d_t] = 0.0
            target_q = r_t + self.gamma * max_next_q
        
        loss = F.mse_loss(q_vals, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        self.steps += 1
        if self.steps % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)
        return loss.item()

    def save(self, path: str):
        from safetensors.torch import save_file
        save_file(self.policy_net.state_dict(), path)
        print(f"Model saved to {path}")

    def load(self, path: str):
        from safetensors.torch import load_file
        self.policy_net.load_state_dict(load_file(path))
        self.target_net.load_state_dict(self.policy_net.state_dict())
        print(f"Model loaded from {path}")

# ==============================================================================
# ALNS
# ==============================================================================

def calculate_route_cost(route: List[int], instance: VRPTWInstance) -> Tuple[float, bool]:
    if not route:
        return 0.0, True
    total_dist, current_time, current_load, current = 0.0, 0.0, 0.0, instance.depot
    for idx in route:
        c = instance.customers[idx]
        d = distance(current, c)
        total_dist += d
        arrival = current_time + d
        if arrival > c.due_date:
            return float('inf'), False
        current_time = max(arrival, c.ready_time) + c.service_time
        current_load += c.demand
        if current_load > instance.vehicle_capacity:
            return float('inf'), False
        current = c
    return total_dist + distance(current, instance.depot), True


def calculate_solution_cost(routes: List[List[int]], instance: VRPTWInstance) -> Tuple[float, bool]:
    total = 0.0
    for route in routes:
        cost, ok = calculate_route_cost(route, instance)
        if not ok:
            return float('inf'), False
        total += cost
    return total, True


def time_slack_insertion_cost(route: List[int], cust_idx: int, pos: int, instance: VRPTWInstance) -> Tuple[float, bool]:
    new_route = route[:pos] + [cust_idx] + route[pos:]
    new_cost, ok = calculate_route_cost(new_route, instance)
    if not ok:
        return float('inf'), False
    old_cost, _ = calculate_route_cost(route, instance)
    return new_cost - old_cost, True


class FastALNS:
    def __init__(self, instance: VRPTWInstance, initial_routes: Optional[List[List[int]]] = None):
        self.instance = instance
        self.routes = deepcopy(initial_routes) if initial_routes else self._greedy_init()
        self._ensure_all_visited()
        self.best_routes = deepcopy(self.routes)
        self.best_cost, _ = calculate_solution_cost(self.best_routes, instance)

    def _greedy_init(self) -> List[List[int]]:
        routes, unvisited = [], set(range(len(self.instance.customers)))
        while unvisited:
            route, current, current_time, current_load = [], self.instance.depot, 0.0, 0.0
            while True:
                best_cust, best_dist = None, float('inf')
                for idx in unvisited:
                    c = self.instance.customers[idx]
                    d = distance(current, c)
                    arr = max(current_time + d, c.ready_time)
                    if arr <= c.due_date and current_load + c.demand <= self.instance.vehicle_capacity and d < best_dist:
                        best_dist, best_cust = d, idx
                if best_cust is None:
                    break
                c = self.instance.customers[best_cust]
                current_time = max(current_time + distance(current, c), c.ready_time) + c.service_time
                current_load += c.demand
                current = c
                route.append(best_cust)
                unvisited.remove(best_cust)
            if route:
                routes.append(route)
        return routes

    def _ensure_all_visited(self):
        visited = {c for route in self.routes for c in route}
        for c in set(range(len(self.instance.customers))) - visited:
            inserted = False
            for route in self.routes:
                for pos in range(len(route) + 1):
                    cost, ok = time_slack_insertion_cost(route, c, pos, self.instance)
                    if ok and cost < float('inf'):
                        route.insert(pos, c)
                        inserted = True
                        break
                if inserted:
                    break
            if not inserted:
                self.routes.append([c])

    def _try_route_merge(self) -> bool:
        for i in range(len(self.routes)):
            for j in range(i + 1, len(self.routes)):
                if not self.routes[i] or not self.routes[j]:
                    continue
                merged = self.routes[i] + self.routes[j]
                cost, ok = calculate_route_cost(merged, self.instance)
                if ok and cost < float('inf'):
                    self.routes[i], self.routes[j] = merged, []
                    return True
        return False

    def run(self, time_limit: float = 30.0) -> Tuple[List[List[int]], float]:
        start = time_module.time()
        curr_cost, _ = calculate_solution_cost(self.routes, self.instance)
        temp = 100.0
        while time_module.time() - start < time_limit:
            if random.random() < 0.1:
                self._try_route_merge()
                self.routes = [r for r in self.routes if r]
            saved = deepcopy(self.routes)
            all_custs = [c for r in self.routes for c in r]
            if len(all_custs) <= 2:
                continue
            num_remove = random.randint(2, min(8, len(all_custs)//3 + 1))
            removed = random.sample(all_custs, num_remove)
            for r in self.routes:
                for c in removed:
                    if c in r:
                        r.remove(c)
            for cust in removed:
                best_cost_inc, best_route, best_pos = float('inf'), -1, -1
                for i, route in enumerate(self.routes):
                    for j in range(len(route) + 1):
                        cost_inc, ok = time_slack_insertion_cost(route, cust, j, self.instance)
                        if ok and cost_inc < best_cost_inc:
                            best_cost_inc, best_route, best_pos = cost_inc, i, j
                if best_route >= 0:
                    self.routes[best_route].insert(best_pos, cust)
                else:
                    self.routes.append([cust])
            new_cost, ok = calculate_solution_cost(self.routes, self.instance)
            if not ok:
                self.routes = saved
                continue
            if new_cost < self.best_cost:
                self.best_routes, self.best_cost, curr_cost = deepcopy(self.routes), new_cost, new_cost
            elif new_cost < curr_cost or random.random() < math.exp((curr_cost - new_cost) / max(temp, 0.01)):
                curr_cost = new_cost
            else:
                self.routes = saved
            temp *= 0.9995
        return self.best_routes, self.best_cost

# ==============================================================================
# TRAINING
# ==============================================================================

def dqn_construct_routes(agent: DQNAgent, instance: VRPTWInstance) -> List[List[int]]:
    env = VRPTWEnv(instance)
    state, done = env.reset(), False
    while not done:
        action = agent.select_action(state, training=False)
        if action == -1:
            break
        state, _, done = env.step(action)
    return [r for r in env.routes if r]


def train_curriculum(agent: DQNAgent) -> List[float]:
    all_rewards = []
    for phase_idx, phase in enumerate(CURRICULUM):
        print(f"\n[Phase {phase_idx+1}/{len(CURRICULUM)}] {phase['customers']} customers, {phase['episodes']} episodes")
        agent.buffer = ReplayBuffer(BUFFER_SIZE)
        instances = generate_curriculum_dataset(phase)
        phase_rewards = []
        pbar = tqdm(range(phase["episodes"]), desc=f"Phase {phase_idx+1}")
        for ep in pbar:
            instance = random.choice(instances)
            env = VRPTWEnv(instance)
            state, total_reward, done = env.reset(), 0.0, False
            while not done:
                action = agent.select_action(state, training=True)
                if action == -1:
                    break
                next_state, reward, done = env.step(action)
                agent.buffer.push(state, action, reward, next_state, done)
                agent.update()
                state, total_reward = next_state, total_reward + reward
            phase_rewards.append(total_reward)
            all_rewards.append(total_reward)
            if (ep+1) % 100 == 0:
                pbar.set_postfix({'R': f'{np.mean(phase_rewards[-100:]):.2f}', 'E': f'{agent.epsilon:.3f}', 'B': len(agent.buffer)})
        print(f"Phase {phase_idx+1} complete. Avg Reward: {np.mean(phase_rewards[-100:]):.4f}")
    return all_rewards

# ==============================================================================
# EVALUATION
# ==============================================================================

def evaluate_all(agent: DQNAgent, time_limit: float = 30.0) -> List[Dict]:
    print("\n" + "="*70 + "\nEVALUATION ON RC BENCHMARK\n" + "="*70)
    rc_files = [os.path.join(DATA_DIR, f) for f in sorted(os.listdir(DATA_DIR)) if f.lower().startswith("rc") and f.endswith(".txt")]
    if not rc_files:
        print("No RC instances found")
        return []
    print(f"Found {len(rc_files)} instances")
    results = []
    for filepath in tqdm(rc_files, desc="Evaluating"):
        name = os.path.basename(filepath).replace(".txt", "")
        instance = parse_solomon_file(filepath)
        alns_pure = FastALNS(instance, None)
        pure_routes, pure_cost = alns_pure.run(time_limit)
        init_routes = dqn_construct_routes(agent, instance)
        alns_hybrid = FastALNS(instance, init_routes)
        hybrid_routes, hybrid_cost = alns_hybrid.run(time_limit)
        impr = ((pure_cost - hybrid_cost) / pure_cost * 100) if pure_cost > 0 and pure_cost < float('inf') else 0
        results.append({'instance': name, 'alns': pure_cost, 'hybrid': hybrid_cost, 'impr': impr, 'alns_v': len([r for r in pure_routes if r]), 'hybrid_v': len([r for r in hybrid_routes if r]), 'routes': hybrid_routes})
        print(f"{name}: ALNS={pure_cost:.1f} ({results[-1]['alns_v']}v) | Hybrid={hybrid_cost:.1f} ({results[-1]['hybrid_v']}v) | {impr:+.1f}%")
    
    print("-"*70)
    valid = [r for r in results if r['alns'] < float('inf') and r['hybrid'] < float('inf')]
    if valid:
        print(f"Avg ALNS: {np.mean([r['alns'] for r in valid]):.1f}")
        print(f"Avg Hybrid: {np.mean([r['hybrid'] for r in valid]):.1f}")
        print(f"Avg Improvement: {np.mean([r['impr'] for r in valid]):+.2f}%")
        print(f"Win Rate: {sum(1 for r in valid if r['impr'] > 0)}/{len(valid)} instances")
    return results


def visualize_routes(results: List[Dict]):
    import matplotlib.pyplot as plt
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    valid = [r for r in results if r['impr'] != 0]
    if not valid:
        return
    sorted_results = sorted(valid, key=lambda x: x['impr'], reverse=True)
    to_plot = sorted_results[:3] + sorted_results[-3:]
    for result in to_plot:
        filepath = os.path.join(DATA_DIR, f"{result['instance']}.txt")
        if not os.path.exists(filepath):
            continue
        instance = parse_solomon_file(filepath)
        routes = result['routes']
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.scatter(instance.depot.x, instance.depot.y, c='red', s=200, marker='s', zorder=5, label='Depot')
        for c in instance.customers:
            ax.scatter(c.x, c.y, c='blue', s=50, alpha=0.7)
        colors = plt.cm.tab10(np.linspace(0, 1, len(routes)))
        for i, route in enumerate(routes):
            if not route:
                continue
            xs = [instance.depot.x] + [instance.customers[idx].x for idx in route] + [instance.depot.x]
            ys = [instance.depot.y] + [instance.customers[idx].y for idx in route] + [instance.depot.y]
            ax.plot(xs, ys, c=colors[i], linewidth=1.5, alpha=0.7)
        ax.set_title(f"{result['instance']}: Cost={result['hybrid']:.1f} ({result['impr']:+.1f}%)")
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"{result['instance']}_routes.png"), dpi=150)
        plt.close()
    print(f"Visualizations saved to {OUTPUT_DIR}")

# ==============================================================================
# RUN
# ==============================================================================

print("\n" + "="*70 + "\nHYBRID VRPTW EXPERIMENT\n" + "="*70)

download_kaggle_data()
print(f"Data: {DATA_DIR}")

agent = DQNAgent(DEVICE)

if os.path.exists(MODEL_PATH):
    agent.load(MODEL_PATH)
else:
    print("\n" + "="*70 + "\nTRAINING\n" + "="*70)
    train_curriculum(agent)
    agent.save(MODEL_PATH)

results = evaluate_all(agent, EVAL_TIME_LIMIT)
if results:
    visualize_routes(results)

print("\n" + "="*70 + "\nEXPERIMENT COMPLETE\n" + "="*70)
