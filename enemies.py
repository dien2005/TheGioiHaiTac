from settings import * 
from random import choice
from timer import Timer
import heapq
import numpy as np
from random import random

class Tooth(pygame.sprite.Sprite):
    def __init__(self, pos, frames, groups, collision_sprites, player, level, algorithm):
        super().__init__(groups)
        self.frames, self.frame_index = frames, 0
        self.image = self.frames[self.frame_index]
        self.rect = self.image.get_frect(topleft=pos)
        self.z = Z_LAYERS['main']

        self.player = player
        self.level = level
        self.algorithm = algorithm
        self.speed = 200
        self.path = []
        self.hit_timer = Timer(250)
        self.direction = 1
        self.search_direction = 1  
        self.base_y = int(pos[1] // TILE_SIZE)  # Lưu tầng ban đầu của Tooth
        self.vision_radius = 300         # Tầm nhìn của Tooth (pixel)
        self.state = 'patrol'            # Trạng thái ban đầu
        self.search_direction = 1        # Dò trái/phải khi tuần tra

        # Q-Learning parameters
        self.q_table = {}  # Từ điển lưu giá trị Q: {x: [Q_trái, Q_phải]}
        self.alpha = 0.1   # Tỷ lệ học
        self.gamma = 0.9   # Hệ số chiết khấu
        self.epsilon = 0.1 # Tỷ lệ thăm dò
        self.max_steps = 50  # Giới hạn số bước mỗi tập
        self.grid_width = None  # Trì hoãn khởi tạo

    def get_grid_pos(self, pos):
        return (int(pos[0] // TILE_SIZE), int(pos[1] // TILE_SIZE))

    def heuristic(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def has_floor_support(self, grid_x, grid_y):
        # Kiểm tra ô bên dưới có sàn không
        if not hasattr(self.level, 'grid'):
            print("Lỗi: self.level.grid chưa được khởi tạo trong has_floor_support")
            return False
        if grid_y + 1 >= len(self.level.grid):
            return False  # Ngoài biên dưới
        return self.level.grid[grid_y + 1][grid_x] == 0  # Có vật cản (sàn) bên dưới

    def build_grid(self):
        if not hasattr(self.level, 'grid'):
            print("Lỗi: self.level.grid chưa được khởi tạo trong build_grid")
            return {}
        grid = {}
        center_x, center_y = self.get_grid_pos(self.rect.center)
        for y in range(center_y - 8, center_y + 9):
            for x in range(center_x - 14, center_x + 15):
                if 0 <= x < len(self.level.grid[0]) and 0 <= y < len(self.level.grid):
                    grid[(x, y)] = self.level.grid[y][x] == 0  # True: chặn, False: trống
                else:
                    grid[(x, y)] = True  # Ngoài biên thì chặn
        return grid

    def dfs(self, start, target):
        visited = set()
        stack = [(start, [start])]
        directions = [(1, 0), (-1, 0)]  # Chỉ di chuyển ngang

        while stack:
            current, path = stack.pop()
            if current == target and current[1] == self.base_y:
                return path
            if current in visited:
                continue
            visited.add(current)

            for dx, dy in directions:
                next_pos = (current[0] + dx, current[1] + dy)
                if (0 <= next_pos[0] < len(self.level.grid[0]) and 
                    0 <= next_pos[1] < len(self.level.grid) and 
                    self.level.grid[next_pos[1]][next_pos[0]] == 1 and 
                    self.has_floor_support(next_pos[0], next_pos[1]) and 
                    next_pos[1] == self.base_y and 
                    next_pos not in visited):
                    stack.append((next_pos, path + [next_pos]))
        return []

    def a_star(self, start, target):
        open_set = [(0, start)]
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, target)}
        directions = [(1, 0), (-1, 0)]  # Chỉ di chuyển ngang

        while open_set:
            current = heapq.heappop(open_set)[1]
            if current == target and current[1] == self.base_y:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]

            for dx, dy in directions:
                neighbor = (current[0] + dx, current[1] + dy)
                if (0 <= neighbor[0] < len(self.level.grid[0]) and 
                    0 <= neighbor[1] < len(self.level.grid) and 
                    self.level.grid[neighbor[1]][neighbor[0]] == 1 and 
                    self.has_floor_support(neighbor[0], neighbor[1]) and 
                    neighbor[1] == self.base_y):
                    tentative_g_score = g_score[current] + 1
                    if tentative_g_score < g_score.get(neighbor, float('inf')):
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g_score
                        f_score[neighbor] = g_score[neighbor] + self.heuristic(neighbor, target)
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
        return []

    def steepest_ascent(self, start, target):
        directions = [(1, 0), (-1, 0)]
        current = start
        visited = set()
        path = [current]

        while True:
            best = None
            best_score = self.heuristic(current, target)

            for dx, dy in directions:
                next_node = (current[0] + dx, current[1] + dy)
                if (0 <= next_node[0] < len(self.level.grid[0]) and
                    0 <= next_node[1] < len(self.level.grid) and
                    self.level.grid[next_node[1]][next_node[0]] == 1 and
                    self.has_floor_support(next_node[0], next_node[1]) and
                    next_node[1] == self.base_y and
                    next_node not in visited):

                    score = self.heuristic(next_node, target)
                    if score < best_score:
                        best = next_node
                        best_score = score

            if best is None:
                break

            visited.add(best)
            current = best
            path.append(current)

            if current == target:
                break

        return path
    
    def simulated_annealing(self, start, target):
        import math
        from random import choice, random

        T = 10.0              # Nhiệt độ ban đầu
        T_min = 0.1           # Nhiệt độ thấp nhất
        alpha = 0.95          # Tỷ lệ giảm nhiệt độ

        current = start
        path = [current]

        def neighbors(pos):
            x, y = pos
            result = []
            for dx in [-1, 1]:  # chỉ đi ngang
                nx = x + dx
                ny = y
                if (0 <= nx < len(self.level.grid[0]) and
                    0 <= ny < len(self.level.grid) and
                    self.level.grid[ny][nx] == 1 and
                    self.has_floor_support(nx, ny)):
                    result.append((nx, ny))
            return result

        def cost(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])  # Manhattan

        current_cost = cost(current, target)

        while T > T_min:
            ns = neighbors(current)
            if not ns:
                break
            next_node = choice(ns)
            next_cost = cost(next_node, target)

            delta = next_cost - current_cost
            if delta < 0 or random() < math.exp(-delta / T):
                current = next_node
                path.append(current)
                current_cost = next_cost
                if current == target:
                    break

            T *= alpha  # làm nguội

        return path

    def backtracking(self, start, target, visited=None, path=None, max_length=50):
        if not hasattr(self.level, 'grid'):
            print("Lỗi: self.level.grid chưa được khởi tạo trong backtracking")
            return []
        if visited is None:
            visited = set()
        if path is None:
            path = [start]
        if len(path) > max_length:
            return []
        if start == target and start[1] == self.base_y:
            return path

        directions = [(1, 0), (-1, 0)]
        for dx, dy in directions:
            next_pos = (start[0] + dx, start[1] + dy)
            if (0 <= next_pos[0] < len(self.level.grid[0]) and 
                0 <= next_pos[1] < len(self.level.grid) and 
                self.level.grid[next_pos[1]][next_pos[0]] == 1 and 
                self.has_floor_support(next_pos[0], next_pos[1]) and 
                next_pos[1] == self.base_y and 
                next_pos not in visited):
                visited.add(next_pos)
                new_path = self.backtracking(next_pos, target, visited, path + [next_pos], max_length)
                if new_path:
                    return new_path
                visited.remove(next_pos)
        return []

    def initialize_q_table(self, x):
        if x not in self.q_table:
            self.q_table[x] = [0.0, 0.0]  # [Q_trái, Q_phải]

    def q_learning(self, start, target):
        if not hasattr(self.level, 'grid'):
            print("Lỗi: self.level.grid chưa được khởi tạo trong q_learning")
            return []
        if self.grid_width is None:
            self.grid_width = len(self.level.grid[0])

        current_x = start[0]  # Trạng thái bắt đầu (tọa độ x)
        path = [start]
        steps = 0

        while steps < self.max_steps:
            self.initialize_q_table(current_x)
            # Chọn hành động (ε-greedy)
            if random.random() < self.epsilon:
                action_idx = choice([0, 1])  # Ngẫu nhiên: 0=trái, 1=phải
            else:
                action_idx = np.argmax(self.q_table[current_x])  # Hành động tốt nhất

            # Thực hiện hành động
            dx = -1 if action_idx == 0 else 1
            next_x = current_x + dx
            next_pos = (next_x, self.base_y)

            # Kiểm tra tính hợp lệ và tính phần thưởng
            if (0 <= next_x < self.grid_width and 
                self.level.grid[self.base_y][next_x] == 1 and 
                self.has_floor_support(next_x, self.base_y)):
                if next_pos == target:
                    reward = 100  # Đến được người chơi
                    path.append(next_pos)
                    # Cập nhật giá trị Q
                    self.q_table[current_x][action_idx] += self.alpha * (
                        reward - self.q_table[current_x][action_idx]
                    )
                    break
                else:
                    reward = -1  # Phạt mỗi bước
                    path.append(next_pos)
            else:
                reward = -100  # Di chuyển không hợp lệ
                # Cập nhật giá trị Q và dừng
                self.q_table[current_x][action_idx] += self.alpha * (
                    reward - self.q_table[current_x][action_idx]
                )
                break

            # Cập nhật giá trị Q cho di chuyển hợp lệ
            self.initialize_q_table(next_x)
            max_future_q = max(self.q_table[next_x])
            self.q_table[current_x][action_idx] += self.alpha * (
                reward + self.gamma * max_future_q - self.q_table[current_x][action_idx]
            )

            # Chuyển sang trạng thái tiếp theo
            current_x = next_x
            steps += 1

        return path

    def no_observation_search(self, start):
        if not hasattr(self.level, 'grid'):
            print("Lỗi: self.level.grid chưa được khởi tạo trong no_observation_search")
            return []
        path = []
        x, y = start

        for _ in range(10):  # Tối đa 10 bước dò
            next_x = x + self.search_direction
            if (0 <= next_x < len(self.level.grid[0]) and
                0 <= y < len(self.level.grid) and
                self.level.grid[y][next_x] == 1 and
                self.has_floor_support(next_x, y)):
                x = next_x
                path.append((x, y))
            else:
                self.search_direction *= -1  # quay đầu nếu không đi được
                
        return path

    def patrol_randomly(self):
        start = self.get_grid_pos(self.rect.topleft)
        self.path = self.no_observation_search(start)

    def update_path(self):
        if not hasattr(self.level, 'grid'):
            print("Lỗi: self.level.grid chưa được khởi tạo trong update_path")
            return
        start = self.get_grid_pos(self.rect.topleft)
        target = self.get_grid_pos(self.player.hitbox_rect.topleft)
        if target[1] == self.base_y:
            if self.algorithm == 'DFS':
                self.path = self.dfs(start, target)
            elif self.algorithm == 'A*':
                self.path = self.a_star(start, target)
            elif self.algorithm == 'STEEPEST':
                self.path = self.steepest_ascent(start, target)
            elif self.algorithm == 'ANNEALING':
                self.path = self.simulated_annealing(start, target)
            elif self.algorithm == 'BACKTRACKING':
                self.path = self.backtracking(start, target)
            elif self.algorithm == 'Q_LEARNING':
                self.path = self.q_learning(start, target)
            elif self.algorithm == 'NO_OBSERVATION':
                self.path = self.no_observation_search(start)
        else:
            self.path = []

    def move_along_path(self, dt):
        if vector(self.rect.center).distance_to(self.player.hitbox_rect.center) < self.vision_radius:
            if not self.path:
                self.update_path()
                if not self.path:
                    return
            next_tile = self.path[0]
            next_pos = (next_tile[0] * TILE_SIZE + TILE_SIZE / 2, next_tile[1] * TILE_SIZE + TILE_SIZE)  # Điểm giữa đáy

            direction_vector = vector(next_pos) - vector(self.rect.midbottom)
            if direction_vector.length() < self.speed * dt:
                self.rect.midbottom = next_pos
                self.path.pop(0)
            else:
                direction_vector = direction_vector.normalize()
                move = direction_vector * self.speed * dt
                self.rect.midbottom += move
                self.direction = 1 if direction_vector.x > 0 else -1

    def update(self, dt):
        self.hit_timer.update()

        # Kiểm tra khoảng cách để xác định trạng thái
        distance = vector(self.rect.center).distance_to(self.player.hitbox_rect.center)
        if distance <= self.vision_radius:
            self.state = 'chase'
        else:
            self.state = 'patrol'

        # Animation
        self.frame_index += ANIMATION_SPEED * dt
        self.image = self.frames[int(self.frame_index % len(self.frames))]
        self.image = pygame.transform.flip(self.image, True, False) if self.direction < 0 else self.image

        # Logic chính
        if self.state == 'chase':
            if not self.path:
                self.update_path()
        else:
            self.patrol_randomly()

        # Di chuyển theo đường đã có
        self.move_along_path(dt)

    def reverse(self):
        if not self.hit_timer.active:
            self.direction *= -1
            self.hit_timer.activate()

import pygame
import heapq
import random
import math

class Timer:
    def __init__(self, duration):
        self.duration = duration
        self.time_left = duration
        self.active = False

    def activate(self):
        self.active = True
        self.time_left = self.duration

    def update(self):
        if self.active:
            self.time_left -= 1
            if self.time_left <= 0:
                self.active = False
                self.time_left = 0

class Shell(pygame.sprite.Sprite):
    def __init__(self, pos, frames, groups, reverse, player, create_pearl):
        super().__init__(groups)

        if reverse:
            self.frames = {}
            for key, surfs in frames.items():
                self.frames[key] = [pygame.transform.flip(surf, True, False) for surf in surfs]
            self.bullet_direction = -1
        else:
            self.frames = frames 
            self.bullet_direction = 1
        
        self.frame_index = 0
        self.state = 'idle'
        self.image = self.frames[self.state][self.frame_index]
        self.rect = self.image.get_frect(topleft = pos)
        self.old_rect = self.rect.copy()
        self.z = Z_LAYERS['main']
        self.player = player
        self.shoot_timer = Timer(1000)
        self.has_fired = False
        self.create_pearl = create_pearl

    def state_management(self):
        player_pos, shell_pos = vector(self.player.hitbox_rect.center), vector(self.rect.center)
        player_near = shell_pos.distance_to(player_pos) < 500
        player_front = shell_pos.x < player_pos.x if self.bullet_direction > 0 else shell_pos.x > player_pos.x
        player_level = abs(shell_pos.y - player_pos.y) < 100
        if player_near and player_level and not self.shoot_timer.active:
            self.state = 'fire'
            self.frame_index = 0
            self.shoot_timer.activate()

    def update(self, dt):
        self.shoot_timer.update()
        self.state_management()

        # animation / attack 
        self.frame_index += ANIMATION_SPEED * dt
        if self.frame_index < len(self.frames[self.state]):
            self.image = self.frames[self.state][int(self.frame_index)]

            # fire 
            if self.state == 'fire' and int(self.frame_index) == 3 and not self.has_fired:
                self.create_pearl(self.rect.center, self.bullet_direction)
                self.has_fired = True 
        else:
            self.frame_index = 0
            if self.state == 'fire':
                self.state = 'idle'
                self.has_fired = False
        facing_right = self.player.rect.centerx > self.rect.centerx
        self.image = self.frames[self.state][int(self.frame_index)]
        if not facing_right:
                self.image = pygame.transform.flip(self.image, True, False)	

import pygame
import random
import heapq
from math import dist, exp
from settings import *

class Pearl(pygame.sprite.Sprite):
    def __init__(self, pos, groups, surf, player, grid, max_distance=150, algorithm='DFS'):
        super().__init__(groups)
        self.image = surf
        self.rect = self.image.get_rect(center=pos)
        self.hitbox_rect = self.rect.copy()

        self.player = player
        self.grid = grid
        self.max_distance = max_distance
        self.algorithm = algorithm

        self.speed = 200
        self.path = []
        self.path_index = 0
        self.pos = pygame.Vector2(self.rect.center)
        self.timer = 0
        self.duration = 3  # seconds before self-destruct

        self.find_path()

    def grid_pos(self, pixel_pos):
        return int(pixel_pos[1] // TILE_SIZE), int(pixel_pos[0] // TILE_SIZE)

    def pixel_pos(self, grid_pos):
        return pygame.Vector2(grid_pos[1] * TILE_SIZE + TILE_SIZE / 2,
                              grid_pos[0] * TILE_SIZE + TILE_SIZE / 2)

    def get_neighbors(self, node):
        neighbors = []
        y, x = node
        for dy, dx in [(-1,0), (1,0), (0,-1), (0,1)]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < len(self.grid) and 0 <= nx < len(self.grid[0]) and self.grid[ny][nx] != '1':
                neighbors.append((ny, nx))
        return neighbors

    def dfs(self, start, goal):
        stack = [(start, [start])]
        visited = set()
        while stack:
            current, path = stack.pop()
            if current == goal:
                return path
            if current in visited:
                continue
            visited.add(current)
            for neighbor in self.get_neighbors(current):
                if neighbor not in visited:
                    stack.append((neighbor, path + [neighbor]))
        return []

    def a_star(self, start, goal):
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}

        def heuristic(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        while open_set:
            _, current = heapq.heappop(open_set)
            if current == goal:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                return path[::-1]

            for neighbor in self.get_neighbors(current):
                tentative_g = g_score[current] + 1
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score, neighbor))
        return []

    def hill_climbing(self, start, goal):
        path = [start]
        current = start
        visited = {start}
        while current != goal:
            neighbors = [n for n in self.get_neighbors(current) if n not in visited]
            if not neighbors:
                break
            current = min(neighbors, key=lambda n: dist(n, goal))
            visited.add(current)
            path.append(current)
        return path if current == goal else []

    def beam_search(self, start, goal, width=3):
        frontier = [(start, [start])]
        while frontier:
            new_frontier = []
            for state, path in frontier:
                if state == goal:
                    return path
                for neighbor in self.get_neighbors(state):
                    if neighbor not in path:
                        new_path = path + [neighbor]
                        new_frontier.append((neighbor, new_path))
            frontier = sorted(new_frontier, key=lambda x: dist(x[0], goal))[:width]
        return []

    def simulated_annealing(self, start, goal):
        current = start
        path = [start]
        T = 1.0
        T_min = 0.01
        alpha = 0.9

        def cost(n):
            return dist(n, goal)

        while T > T_min:
            neighbors = self.get_neighbors(current)
            if not neighbors:
                break
            next_node = random.choice(neighbors)
            delta = cost(current) - cost(next_node)
            if delta > 0 or exp(delta / T) > random.random():
                current = next_node
                path.append(current)
                if current == goal:
                    return path
            T *= alpha
        return []

    def backtracking(self, start, goal):
        path = []
        visited = set()

        def backtrack(current):
            if current == goal:
                path.append(current)
                return True
            visited.add(current)
            for neighbor in self.get_neighbors(current):
                if neighbor not in visited:
                    if backtrack(neighbor):
                        path.append(current)
                        return True
            return False

        if backtrack(start):
            return path[::-1]
        else:
            return []

    def q_learning(self, start, goal, episodes=50, alpha=0.1, gamma=0.9, epsilon=0.2):
        Q = {}
        for y in range(len(self.grid)):
            for x in range(len(self.grid[0])):
                Q[(y, x)] = {n: 0 for n in self.get_neighbors((y, x))}

        for _ in range(episodes):
            state = start
            visited = set()
            while state != goal:
                if not Q[state]:
                    break
                if random.random() < epsilon:
                    next_state = random.choice(list(Q[state].keys()))
                else:
                    next_state = max(Q[state], key=Q[state].get)

                reward = 100 if next_state == goal else -1
                max_future_q = max(Q[next_state].values(), default=0)
                Q[state][next_state] += alpha * (reward + gamma * max_future_q - Q[state][next_state])

                state = next_state
                visited.add(state)

        state = start
        path = [state]
        visited = set()
        while state != goal and Q[state]:
            next_state = max(Q[state], key=Q[state].get)
            if next_state in visited:
                break
            visited.add(next_state)
            state = next_state
            path.append(state)
        return path

    def find_path(self):
        start = self.grid_pos(self.rect.center)
        goal = self.grid_pos(self.player.rect.center)

        if self.algorithm == 'DFS':
            self.path = self.dfs(start, goal)
        elif self.algorithm == 'A*':
            self.path = self.a_star(start, goal)
        elif self.algorithm == 'HillClimbing':
            self.path = self.hill_climbing(start, goal)
        elif self.algorithm == 'Beam':
            self.path = self.beam_search(start, goal, width=3)
        elif self.algorithm == 'SimulatedAnnealing':
            self.path = self.simulated_annealing(start, goal)
        elif self.algorithm == 'Backtracking':
            self.path = self.backtracking(start, goal)
        elif self.algorithm == 'Q_Learning':
            self.path = self.q_learning(start, goal)
        else:
            self.path = self.dfs(start, goal)

        if not self.path:
            if self.grid[goal[0]][goal[1]] == '1':
                goal = start
            self.path = [start, goal]

        self.path_index = 0

    def update(self, dt):
        self.timer += dt
        if self.timer >= self.duration:
            self.kill()
            return

        if self.path and self.path_index < len(self.path):
            target_pos = self.pixel_pos(self.path[self.path_index])
            direction = (target_pos - self.pos)
            if direction.length() > 0:
                direction = direction.normalize()
                self.pos += direction * self.speed * dt
                self.rect.center = self.pos
                self.hitbox_rect.center = self.rect.center

                if dist(self.pos, target_pos) < 5:
                    self.path_index += 1
        else:
            self.kill()
