import time
import random
import queue
import matplotlib.pyplot as plt

class FlowNetwork:
    def __init__(self, capacity, cost=None):
        self.n = len(capacity)
        self.capacity = capacity
        self.cost = cost
        self.flow = [[0] * self.n for _ in range(self.n)]

    def bfs(self, s, t, parent):
        visited = [False] * self.n
        q = queue.Queue()
        q.put(s)
        visited[s] = True

        while not q.empty():
            u = q.get()
            for v in range(self.n):
                if not visited[v] and self.capacity[u][v] - self.flow[u][v] > 0:
                    q.put(v)
                    visited[v] = True
                    parent[v] = u
        return visited[t]

    def ford_fulkerson(self, s, t):
        parent = [-1] * self.n
        max_flow = 0

        while self.bfs(s, t, parent):
            path_flow = float('inf')
            v = t
            while v != s:
                u = parent[v]
                path_flow = min(path_flow, self.capacity[u][v] - self.flow[u][v])
                v = parent[v]

            v = t
            while v != s:
                u = parent[v]
                self.flow[u][v] += path_flow
                self.flow[v][u] -= path_flow
                v = parent[v]

            max_flow += path_flow
        return max_flow

    def push_relabel(self, source, sink):
        n = self.n
        height = [0] * n
        excess = [0] * n
        flow = [[0] * n for _ in range(n)]

        height[source] = n
        for v in range(n):
            flow[source][v] = self.capacity[source][v]
            flow[v][source] = -self.capacity[source][v]
            excess[v] = self.capacity[source][v]

        def push(u, v):
            send = min(excess[u], self.capacity[u][v] - flow[u][v])
            flow[u][v] += send
            flow[v][u] -= send
            excess[u] -= send
            excess[v] += send

        def relabel(u):
            min_height = float('inf')
            for v in range(n):
                if self.capacity[u][v] - flow[u][v] > 0:
                    min_height = min(min_height, height[v])
            height[u] = min_height + 1

        def discharge(u):
            for v in range(n):
                while excess[u] > 0 and self.capacity[u][v] - flow[u][v] > 0:
                    if height[u] == height[v] + 1:
                        push(u, v)
                    else:
                        relabel(u)

        active = [i for i in range(n) if i != source and i != sink and excess[i] > 0]

        while active:
            u = active.pop(0)
            old_height = height[u]
            discharge(u)
            if height[u] > old_height:
                active.insert(0, u)

        return sum(flow[source][i] for i in range(n))

    def bellman_ford(self, s):
        distance = [float('inf')] * self.n
        distance[s] = 0
        for _ in range(self.n - 1):
            for u in range(self.n):
                for v in range(self.n):
                    if self.capacity[u][v] > 0 and distance[v] > distance[u] + self.cost[u][v]:
                        distance[v] = distance[u] + self.cost[u][v]
        return distance

    def min_cost_flow(self, s, t, required_flow):
        flow = 0
        cost = 0
        residual = [row[:] for row in self.capacity]

        while flow < required_flow:
            dist = self.bellman_ford(s)
            if dist[t] == float('inf'):
                break

            path_flow = required_flow - flow
            v = t
            while v != s:
                u = self.prev[v]
                path_flow = min(path_flow, residual[u][v])
                v = u

            v = t
            while v != s:
                u = self.prev[v]
                residual[u][v] -= path_flow
                residual[v][u] += path_flow
                cost += path_flow * self.cost[u][v]
                v = u

            flow += path_flow

        return flow, cost

def read_file(filename, with_cost=False):
    with open(filename, 'r') as file:
        n = int(file.readline())
        cap = [list(map(int, file.readline().split())) for _ in range(n)]
        if with_cost:
            file.readline()
            cost = [list(map(int, file.readline().split())) for _ in range(n)]
            return cap, cost
        return cap, None

def generate_random_flow_problem(n):
    cap = [[0 for _ in range(n)] for _ in range(n)]
    cost = [[0 for _ in range(n)] for _ in range(n)]

    edges = (n * n) // 2

    for _ in range(edges):
        i, j = random.randint(0, n-1), random.randint(0, n-1)
        while i == j:
            i, j = random.randint(0, n-1), random.randint(0, n-1)
        cap[i][j] = random.randint(1, 100)
        cost[i][j] = random.randint(1, 100)

    return cap, cost

def benchmark():
    ns = [10, 20, 40, 100, 400, 1000, 4000]
    FF_times, PR_times, MIN_times = [], [], []

    for n in ns:
        ff, pr, mn = [], [], []
        for _ in range(100):
            cap, cost = generate_random_flow_problem(n)
            s, t = 0, n-1
            
            # Ford-Fulkerson
            net = FlowNetwork([row[:] for row in cap])
            start = time.time()
            net.ford_fulkerson(s, t)
            ff.append(time.time() - start)

            # Push-Relabel
            net = FlowNetwork([row[:] for row in cap])
            start = time.time()
            net.push_relabel(s, t)
            pr.append(time.time() - start)

            # Min-Cost
            net = FlowNetwork([row[:] for row in cap], cost)
            start = time.time()
            net.min_cost_flow(s, t, sum(cap[0]) // 2)
            mn.append(time.time() - start)

        FF_times.append(ff)
        PR_times.append(pr)
        MIN_times.append(mn)

    # Plot
    for times, label in zip([FF_times, PR_times, MIN_times], ["Ford-Fulkerson", "Push-Relabel", "Min-Cost"]):
        plt.figure()
        for i, n in enumerate(ns):
            plt.scatter([n]*100, times[i], s=5)
        plt.title(f"{label} Execution Time")
        plt.xlabel("n")
        plt.ylabel("Time (s)")
        plt.show()
