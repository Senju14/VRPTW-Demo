import time
import random

def solve_placeholder(algo_name, instance, max_vehicles):
    """
    Mô phỏng thuật toán đang chạy. 
    Sau này bạn sẽ replace logic này bằng import từ alns.py hoặc dqn.py
    """
    # Giả lập thời gian tính toán khác nhau
    if "DQN" in algo_name:
        time.sleep(1.5) 
    else:
        time.sleep(0.8)

    # Giả lập kết quả trả về để frontend hiển thị được
    depot = instance['depot']
    customers = instance['customers']
    
    # Chia customers thành các route ngẫu nhiên (chỉ để demo UI)
    routes = []
    pool = customers.copy()
    random.shuffle(pool)
    
    while pool:
        route_nodes = []
        load = 0
        # Lấy tối đa 5 khách hoặc cho đến khi hết
        for _ in range(min(len(pool), 5)):
            node = pool.pop()
            route_nodes.append(node)
        routes.append({"nodes": route_nodes})
        if len(routes) >= max_vehicles:
            break
            
    # Tính distance giả
    total_dist = random.uniform(800, 1200)

    return {
        "algorithm": algo_name,
        "vehicles": len(routes),
        "distance": total_dist,
        "routes": routes,
        "depot": depot
    }
