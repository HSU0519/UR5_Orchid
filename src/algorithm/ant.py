import time
import math
import random
import numpy as np

# 计算两点间距离的函数
def distance(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

# 初始化距离矩阵
def create_distance_matrix(locations):
    n = len(locations)
    distance_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                distance_matrix[i][j] = distance(locations[i][1], locations[j][1])
    return distance_matrix

# 蚁群算法
def ant_colony_optimization(distance_matrix, n_ants=10, n_iterations=100, alpha=1.0, beta=2.0, evaporation_rate=0.5):
    start_time = time.time()
    
    n = len(distance_matrix)
    
    # 只有 0 或 1 個點時，不需要跑蟻群演算法
    if n <= 1:
        end_time = time.time()
        return list(range(n)), 0, end_time - start_time
    
    # 將距離矩陣中非對角線的 0 值替換為極小值，避免除以零
    safe_distance_matrix = np.array(distance_matrix, dtype=float)
    for i in range(n):
        for j in range(n):
            if i != j and safe_distance_matrix[i][j] == 0:
                safe_distance_matrix[i][j] = 1e-6
    
    pheromones = np.ones((n, n)) / n
    best_route = None
    best_distance = float('inf')

    for _ in range(n_iterations):
        all_routes = []
        all_distances = []
        
        for _ in range(n_ants):
            route = []
            visited = set()
            current_city = random.randint(0, n - 1)
            route.append(current_city)
            visited.add(current_city)
            
            for _ in range(n - 1):
                probabilities = []
                for next_city in range(n):
                    if next_city not in visited:
                        pheromone = pheromones[current_city][next_city] ** alpha
                        heuristic = (1.0 / safe_distance_matrix[current_city][next_city]) ** beta
                        probabilities.append((pheromone * heuristic, next_city))
                
                if not probabilities:
                    break
                    
                total_prob = sum([prob[0] for prob in probabilities])
                if total_prob == 0 or math.isnan(total_prob) or math.isinf(total_prob):
                    # 概率異常時，隨機選擇未造訪的城市
                    unvisited = [c for c in range(n) if c not in visited]
                    if unvisited:
                        next_city = random.choice(unvisited)
                        route.append(next_city)
                        visited.add(next_city)
                        current_city = next_city
                    continue
                    
                probabilities = [(prob[0] / total_prob, prob[1]) for prob in probabilities]
                probabilities.sort(reverse=True)
                
                rand_prob = random.random()
                cumulative_prob = 0.0
                selected = False
                for prob, next_city in probabilities:
                    cumulative_prob += prob
                    if rand_prob <= cumulative_prob:
                        route.append(next_city)
                        visited.add(next_city)
                        current_city = next_city
                        selected = True
                        break
                
                # 浮點誤差導致未選到時，選最後一個未造訪的
                if not selected:
                    next_city = probabilities[-1][1]
                    route.append(next_city)
                    visited.add(next_city)
                    current_city = next_city

            all_routes.append(route)
            route_len = len(route)
            total_distance = sum([distance_matrix[route[i - 1]][route[i]] for i in range(route_len)])
            all_distances.append(total_distance)

            if total_distance < best_distance:
                best_distance = total_distance
                best_route = route

        for i in range(n):
            for j in range(n):
                pheromones[i][j] *= (1.0 - evaporation_rate)

        for route, total_distance in zip(all_routes, all_distances):
            route_len = len(route)
            for i in range(route_len):
                pheromones[route[i - 1]][route[i]] += 1.0 / total_distance
                
    end_time = time.time()
    exetime = end_time - start_time
    return best_route, best_distance, exetime