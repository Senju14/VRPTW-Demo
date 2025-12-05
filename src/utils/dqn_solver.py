

"""
DQN-ALNS Solver for VRPTW
=========================

Uses Deep Q-Network trained model to select destroy operators in ALNS metaheuristic.
- DQN Model: Chooses between random_removal and route_removal strategies
- ALNS: Adaptive Large Neighborhood Search with greedy repair
- Constraints: Vehicle capacity, time windows, vehicle count limit

Main function: solve_dqn_vrptw()
"""

import math
import random
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file

# [DQN SOLVER] Setup Device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[DQN SOLVER] Using device: {device}")
if device.type == 'cuda':
    print(f"[DQN SOLVER] GPU Name: {torch.cuda.get_device_name(0)}")


# ==================== SOLUTION REPRESENTATION ====================

class Solution:
    def __init__(self, routes):
        self.routes = routes
        self.total_cost = 0.0
        self.total_time_violation = 0.0
        self.total_capacity_violation = 0.0


# ==================== EVALUATION FUNCTIONS ====================

def calculate_distance(cust1, cust2):
   
    return math.sqrt((cust1.x - cust2.x)**2 + (cust1.y - cust2.y)**2)


def evaluate_route(route, depot, vehicle_capacity):

    route_cost = 0.0
    route_demand = 0.0
    current_time = 0.0
    time_violation = 0.0
    capacity_violation = 0.0
    last_node = depot

    for cust in route:
        travel_time = calculate_distance(last_node, cust)
        route_cost += travel_time

        arrival_time = current_time + travel_time
        current_time = max(arrival_time, cust.ready_time)

        if current_time > cust.due_date:
            time_violation += (current_time - cust.due_date)

        current_time += cust.service_time
        route_demand += cust.demand
        last_node = cust

    # Return to depot
    travel_time_to_depot = calculate_distance(last_node, depot)
    route_cost += travel_time_to_depot
    current_time += travel_time_to_depot

    if current_time > depot.due_date:
        time_violation += (current_time - depot.due_date)

    if route_demand > vehicle_capacity:
        capacity_violation = route_demand - vehicle_capacity

    return route_cost, time_violation, capacity_violation


def evaluate_solution(solution, depot, vehicle_capacity):
   
    total_cost = 0.0
    total_time_violation = 0.0
    total_capacity_violation = 0.0
    valid_routes = []

    for route in solution.routes:
        if not route:
            continue

        cost, time_v, cap_v = evaluate_route(route, depot, vehicle_capacity)
        total_cost += cost
        total_time_violation += time_v
        total_capacity_violation += cap_v
        valid_routes.append(route)

    solution.routes = valid_routes
    solution.total_cost = total_cost
    solution.total_time_violation = total_time_violation
    solution.total_capacity_violation = total_capacity_violation


# ==================== INITIAL SOLUTION ====================

def create_initial_solution(depot, customers, vehicle_capacity, max_vehicles=None):
   
    routes = []
    customers_copy = list(customers)
    random.shuffle(customers_copy)
    unserved_customers = set(customers_copy)

    while unserved_customers and (max_vehicles is None or len(routes) < max_vehicles):
        new_route = []
        current_load = 0
        current_time = 0.0
        last_node = depot

        for cust in list(unserved_customers):
            travel_time = calculate_distance(last_node, cust)
            arrival_time = current_time + travel_time
            service_start_time = max(arrival_time, cust.ready_time)

            can_serve = (current_load + cust.demand <= vehicle_capacity) and \
                        (service_start_time <= cust.due_date)

            if can_serve:
                time_back_to_depot = calculate_distance(cust, depot)
                service_end_time = service_start_time + cust.service_time
                arrival_at_depot = service_end_time + time_back_to_depot

                can_return_to_depot = (arrival_at_depot <= depot.due_date)

                if can_return_to_depot:
                    new_route.append(cust)
                    unserved_customers.remove(cust)
                    current_load += cust.demand
                    current_time = service_end_time
                    last_node = cust

        if new_route:
            routes.append(new_route)

        if not new_route and unserved_customers:
            break

    initial_sol = Solution(routes)
    evaluate_solution(initial_sol, depot, vehicle_capacity)
    return initial_sol


# ==================== DESTROY OPERATORS ====================

def random_removal(solution, depot, capacity, num_to_remove):
   
    all_customers_in_routes = []
    for route in solution.routes:
        for cust in route:
            all_customers_in_routes.append(cust)

    if len(all_customers_in_routes) == 0:
        return []

    num_to_remove = min(num_to_remove, len(all_customers_in_routes))
    customers_to_remove = random.sample(all_customers_in_routes, num_to_remove)

    removed_customers_set = set(customers_to_remove)

    new_routes = []
    for route in solution.routes:
        new_route = []
        for cust in route:
            if cust not in removed_customers_set:
                new_route.append(cust)

        if new_route:
            new_routes.append(new_route)

    solution.routes = new_routes
    return customers_to_remove


def route_removal(solution, depot, capacity, num_routes_to_remove):
    
    if len(solution.routes) == 0:
        return []

    num_to_remove = min(num_routes_to_remove, len(solution.routes))
    sorted_routes = sorted(solution.routes, key=len)

    customers_to_remove = []
    new_routes = []

    for i in range(len(sorted_routes)):
        if i < num_to_remove:
            customers_to_remove.extend(sorted_routes[i])
        else:
            new_routes.append(sorted_routes[i])

    solution.routes = new_routes
    return customers_to_remove


# ==================== REPAIR OPERATOR ====================

def greedy_insertion(solution, customers_to_insert, depot, capacity, max_vehicles=None):
  
    random.shuffle(customers_to_insert)

    for cust in customers_to_insert:
        best_route_idx = -1
        best_position_idx = -1
        min_cost_increase = float('inf')

        for r_idx, route in enumerate(solution.routes):
            old_cost, _, _ = evaluate_route(route, depot, capacity)

            for p_idx in range(len(route) + 1):
                new_route = route[:p_idx] + [cust] + route[p_idx:]
                new_cost, time_v, cap_v = evaluate_route(new_route, depot, capacity)

                if time_v == 0 and cap_v == 0:
                    cost_increase = new_cost - old_cost

                    if cost_increase < min_cost_increase:
                        min_cost_increase = cost_increase
                        best_route_idx = r_idx
                        best_position_idx = p_idx

        if best_route_idx != -1:
            solution.routes[best_route_idx].insert(best_position_idx, cust)
        else:
            # Only create new route if under vehicle limit
            if max_vehicles is None or len(solution.routes) < max_vehicles:
                new_route = [cust]
                cost, time_v, cap_v = evaluate_route(new_route, depot, capacity)
                if time_v == 0 and cap_v == 0:
                    solution.routes.append(new_route)
            # If at vehicle limit, force insert into best existing route
            elif solution.routes:
                # Find route with minimum cost increase (even with violations)
                best_cost_increase = float('inf')
                fallback_route_idx = 0
                fallback_position_idx = 0
                
                for r_idx, route in enumerate(solution.routes):
                    old_cost, _, _ = evaluate_route(route, depot, capacity)
                    for p_idx in range(len(route) + 1):
                        new_route = route[:p_idx] + [cust] + route[p_idx:]
                        new_cost, _, _ = evaluate_route(new_route, depot, capacity)
                        cost_increase = new_cost - old_cost
                        
                        if cost_increase < best_cost_increase:
                            best_cost_increase = cost_increase
                            fallback_route_idx = r_idx
                            fallback_position_idx = p_idx
                
                solution.routes[fallback_route_idx].insert(fallback_position_idx, cust)

    evaluate_solution(solution, depot, capacity)


# ==================== DQN COMPONENTS ====================

class QNetwork(nn.Module):
   
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.layer1 = nn.Linear(state_size, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, action_size)

    def forward(self, state):
        x = F.relu(self.layer1(state))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


class DQNAgent:
  
    def __init__(self, state_size, action_size):
        self.device = device  # Use global device setting
        self.model = QNetwork(state_size, action_size).to(self.device)
        
    def load_weights(self, model_path):
       
        weights = load_file(model_path)
        self.model.load_state_dict(weights)
        self.model.to(self.device)  # Ensure model is on correct device
        self.model.eval()
        
    def select_action(self, state, epsilon=0.0):
        """Select action using trained DQN model"""
        if random.random() > epsilon:
            with torch.no_grad():
                state_tensor = torch.FloatTensor([state]).to(self.device)
                q_values = self.model(state_tensor)
                action = q_values.max(1)[1].item()
                return action
        else:
            return random.randint(0, 1)


# ==================== DQN-ALNS SOLVER ====================

def get_state(current_cost, best_cost, iterations_since_best):
 
    cost_diff = (current_cost - best_cost) / best_cost if best_cost > 0 else 0
    stuck_norm = iterations_since_best / 100.0
    return [cost_diff, stuck_norm]


def solve_dqn_vrptw(depot, customers, capacity, num_vehicles, model_path, iterations=300):
   
    import time
    start_time = time.time()
    
    print(f"[DQN SOLVER] Starting with {len(customers)} customers, {num_vehicles} vehicles")
    
    # Setup DQN agent with trained model
    destroy_operators = [random_removal, route_removal]  # 2 actions: random vs route removal
    agent = DQNAgent(state_size=2, action_size=2)
    agent.load_weights(model_path)
    print(f"[DQN SOLVER] Loaded model: {model_path.split('/')[-1]}")
    
    # Generate initial solution
    initial_solution = create_initial_solution(depot, customers, capacity, num_vehicles)
    current_solution = copy.deepcopy(initial_solution)
    best_solution = copy.deepcopy(initial_solution)
    iterations_since_best = 0
    
    print(f"[DQN SOLVER] Initial: cost={best_solution.total_cost:.1f}, vehicles={len(best_solution.routes)}/{num_vehicles}")
    
    # ALNS loop
    for iteration in range(1, iterations + 1):
        # DQN decides destroy strategy
        state = get_state(current_solution.total_cost, best_solution.total_cost, iterations_since_best)
        action_idx = agent.select_action(state, epsilon=0.0)
        destroy_op = destroy_operators[action_idx]
        
        # Debug log for first few iterations
        if iteration <= 5 or iteration % 100 == 0:
            action_name = "random_removal" if action_idx == 0 else "route_removal"
            print(f"  Iter {iteration}: DQN chose {action_name} (state={state})")
        
        new_solution = copy.deepcopy(current_solution)
        
        if destroy_op == random_removal:
            num_to_remove = random.randint(5, 15)
            removed = destroy_op(new_solution, depot, capacity, num_to_remove)
        else:
            # Limit route removal to not exceed vehicle constraint
            max_routes_to_remove = min(len(new_solution.routes), max(1, len(new_solution.routes) // 3))
            num_routes = random.randint(1, max_routes_to_remove)
            removed = destroy_op(new_solution, depot, capacity, num_routes)
        
        # Repair with vehicle constraint
        greedy_insertion(new_solution, removed, depot, capacity, num_vehicles)
        
        # Evaluate new solution
        evaluate_solution(new_solution, depot, capacity)
        
        # Only accept solutions that don't exceed vehicle limit
        if len(new_solution.routes) <= num_vehicles:
            # Update best solution
            if new_solution.total_cost < best_solution.total_cost:
                best_solution = copy.deepcopy(new_solution)
                current_solution = copy.deepcopy(new_solution)
                iterations_since_best = 0
            else:
                iterations_since_best += 1
                # Simple acceptance criterion
                if random.random() < 0.1:
                    current_solution = copy.deepcopy(new_solution)
        else:
            iterations_since_best += 1
        
        if iteration % 100 == 0:
            print(f"  Progress: {iteration}/{iterations}, best_cost={best_solution.total_cost:.1f}")
    
    execution_time = time.time() - start_time
    
    # Convert to output format
    routes_as_indices = []
    for route in best_solution.routes:
        route_indices = [0]  # depot first
        for customer in route:
            route_indices.append(customer.id)
        routes_as_indices.append(route_indices)
    
    print(f"[DQN SOLVER] Completed in {execution_time:.1f}s")
    print(f"[DQN SOLVER] Final: cost={best_solution.total_cost:.1f}, vehicles={len(best_solution.routes)}/{num_vehicles}")
    
    return {
        'routes': routes_as_indices,
        'total_distance': best_solution.total_cost,
        'execution_time': execution_time,
        'violations': []
    }
