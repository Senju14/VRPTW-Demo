import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import time
import numpy as np
from safetensors.torch import save_file, load_file
from .utils import VRPTWInstance, Solution
from .alns import ALNS

# --- FORCE GPU DETECTION ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n[SYSTEM] AI Engine is running on: {str(DEVICE).upper()}")
if str(DEVICE) == "cuda":
    print(f"[SYSTEM] GPU Model: {torch.cuda.get_device_name(0)}\n")
else:
    print("[SYSTEM] Warning: Running on CPU. Training/Inference will be slower.\n")

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        ).to(DEVICE) # Move model to GPU

    def forward(self, x):
        return self.network(x)

class DQNAgent:
    def __init__(self, instance: VRPTWInstance, max_episodes=500,
                 epsilon_start=1.0, epsilon_end=0.1, buffer_size=10000, batch_size=64):
        self.instance = instance
        self.max_episodes = max_episodes
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = (epsilon_start - epsilon_end) / max_episodes

        self.state_dim = 9
        self.action_dim = instance.n + 1

        self.q_network = QNetwork(self.state_dim, self.action_dim)
        self.target_network = QNetwork(self.state_dim, self.action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.001)
        self.replay_buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.gamma = 0.99

    def get_state(self, current_node, visited, current_load, current_time):
        inst = self.instance
        route_state = [current_load / inst.capacity, current_time / 1000.0, len(visited) / inst.n]

        unvisited = [i for i in range(1, inst.n + 1) if i not in visited]
        feasible_count = sum(1 for n in unvisited
                           if current_load + inst.demands[n] <= inst.capacity)

        context = [(inst.capacity - current_load) / inst.capacity,
                   len(unvisited) / inst.n,
                   feasible_count / max(len(unvisited), 1),
                   inst.coords[current_node, 0] / 100.0,
                   inst.coords[current_node, 1] / 100.0,
                   inst.demands[current_node] / inst.capacity]

        return np.array(route_state + context, dtype=np.float32)

    def get_feasible_actions(self, current_node, visited, current_load, current_time):
        inst = self.instance
        actions = [0]
        for node in range(1, inst.n + 1):
            if node not in visited:
                if current_load + inst.demands[node] <= inst.capacity:
                    arrival = current_time + inst.distance(current_node, node)
                    if arrival <= inst.due_times[node]:
                        actions.append(node)
        return actions

    def select_action(self, state, feasible_actions):
        if random.random() < self.epsilon:
            return random.choice(feasible_actions)

        with torch.no_grad():
            # Move state to GPU
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
            q_values = self.q_network(state_tensor).squeeze().cpu().numpy() # Move back to CPU for processing
            
            if q_values.ndim == 0: q_values = [q_values]

            feasible_q = [(q_values[a], a) for a in feasible_actions]
            return max(feasible_q)[1]

    def train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Move batches to GPU
        states = torch.FloatTensor(np.array(states)).to(DEVICE)
        actions = torch.LongTensor(np.array(actions)).to(DEVICE)
        rewards = torch.FloatTensor(np.array(rewards)).to(DEVICE)
        next_states = torch.FloatTensor(np.array(next_states)).to(DEVICE)
        dones = torch.FloatTensor(np.array(dones)).to(DEVICE)

        current_q = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q = self.target_network(next_states).max(1)[0].detach()
        target_q = rewards + self.gamma * next_q * (1 - dones)

        loss = nn.MSELoss()(current_q.squeeze(), target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def solve(self):
        best_solution = None
        best_cost = float('inf')
        episodes_to_run = min(self.max_episodes, 50)

        for episode in range(episodes_to_run):
            routes = []
            visited = set()
            while len(visited) < self.instance.n:
                route = []
                current_node = 0
                current_load = 0
                current_time = 0
                while True:
                    state = self.get_state(current_node, visited, current_load, current_time)
                    feasible = self.get_feasible_actions(current_node, visited, current_load, current_time)
                    if len(feasible) == 1: break
                    
                    action = self.select_action(state, feasible)
                    if action == 0: break
                    
                    route.append(action)
                    visited.add(action)
                    
                    dist = self.instance.distance(current_node, action)
                    reward = -dist
                    current_load += self.instance.demands[action]
                    current_time += dist
                    current_time = max(current_time, self.instance.ready_times[action]) + self.instance.service_times[action]
                    
                    next_state = self.get_state(action, visited, current_load, current_time)
                    done = len(visited) == self.instance.n
                    
                    self.replay_buffer.append((state, action, reward, next_state, done))
                    self.train_step()
                    current_node = action
                
                if route: routes.append(route)
                if not route and len(visited) < self.instance.n:
                     remaining = [i for i in range(1, self.instance.n+1) if i not in visited]
                     for r_node in remaining:
                         routes.append([r_node])
                         visited.add(r_node)
                     break
            
            solution = Solution(routes, self.instance)
            if solution.feasible and solution.cost < best_cost:
                best_cost = solution.cost
                best_solution = solution
            
            self.epsilon = max(self.epsilon_end, self.epsilon - self.epsilon_decay)
            if episode % 10 == 0: self.target_network.load_state_dict(self.q_network.state_dict())
        
        return best_solution if best_solution else Solution([], self.instance)

class DQNALNSAgent:
    def __init__(self, instance: VRPTWInstance, max_episodes=500, epsilon_start=1.0, epsilon_end=0.1):
        self.instance = instance
        self.max_episodes = max_episodes
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = (epsilon_start - epsilon_end) / max_episodes
        self.n_destroy = 3; self.n_repair = 2
        self.action_dim = self.n_destroy * self.n_repair
        self.state_dim = 10
        
        self.q_network = QNetwork(self.state_dim, self.action_dim)
        self.target_network = QNetwork(self.state_dim, self.action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.001)
        self.replay_buffer = deque(maxlen=10000)
        self.batch_size = 64
        self.gamma = 0.99
        self.alns = ALNS(instance, max_iterations=1)

    def get_state(self, solution: Solution):
        inst = self.instance
        n_routes = len(solution.routes)
        avg_len = np.mean([len(r) for r in solution.routes]) if solution.routes else 0
        loads = [sum(inst.demands[i] for i in r) for r in solution.routes]
        avg_load = np.mean(loads) if loads else 0
        load_std = np.std(loads) if loads else 0
        times = []
        for route in solution.routes:
            t = 0; curr = 0
            for node in route:
                t += inst.distance(curr, node)
                t = max(t, inst.ready_times[node]) + inst.service_times[node]
                curr = node
            times.append(t)
        avg_time = np.mean(times) if times else 0
        return np.array([solution.cost, n_routes, avg_len, avg_load, load_std, avg_time, inst.capacity, inst.n, 0, 0])[:10]

    def select_operator_pair(self, state):
        if random.random() < self.epsilon: return random.randint(0, self.action_dim - 1)
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
            q_values = self.q_network(state_tensor).squeeze().cpu()
            if q_values.ndim == 0: return 0
            return q_values.argmax().item()

    def train_step(self):
        if len(self.replay_buffer) < self.batch_size: return
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states)).to(DEVICE)
        actions = torch.LongTensor(np.array(actions)).to(DEVICE)
        rewards = torch.FloatTensor(np.array(rewards)).to(DEVICE)
        next_states = torch.FloatTensor(np.array(next_states)).to(DEVICE)
        dones = torch.FloatTensor(np.array(dones)).to(DEVICE)
        
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q = self.target_network(next_states).max(1)[0].detach()
        target_q = rewards + self.gamma * next_q * (1 - dones)
        loss = nn.MSELoss()(current_q.squeeze(), target_q)
        self.optimizer.zero_grad(); loss.backward(); self.optimizer.step()

    def solve(self):
        current = self.alns._create_initial_solution()
        best = current.copy()
        episodes = min(self.max_episodes, 50)
        
        for episode in range(episodes):
            state = self.get_state(current)
            action_idx = self.select_operator_pair(state)
            destroy_idx = action_idx // self.n_repair
            repair_idx = action_idx % self.n_repair
            
            temp_sol = current.copy()
            removed = self.alns.destroy_ops[destroy_idx](temp_sol)
            candidate = self.alns.repair_ops[repair_idx](removed)
            
            reward = current.cost - candidate.cost if candidate.feasible else -100
            if candidate.feasible:
                if candidate.cost < current.cost or random.random() < 0.1:
                    current = candidate
                    if candidate.cost < best.cost: best = candidate.copy()
            
            next_state = self.get_state(current)
            done = episode == episodes - 1
            self.replay_buffer.append((state, action_idx, reward, next_state, done))
            self.train_step()
            self.epsilon = max(self.epsilon_end, self.epsilon - self.epsilon_decay)
            if episode % 10 == 0: self.target_network.load_state_dict(self.q_network.state_dict())
            
        return best
    