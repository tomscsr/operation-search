import time
import random
import queue
import matplotlib.pyplot as plt
from tabulate import tabulate

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

            # Afficher la trace de l'itération
            self.print_residual_graph()

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
        prev = [-1] * self.n
        distance[s] = 0
        
        for _ in range(self.n - 1):
            for u in range(self.n):
                for v in range(self.n):
                    if self.capacity[u][v] > 0 and distance[v] > distance[u] + self.cost[u][v]:
                        distance[v] = distance[u] + self.cost[u][v]
                        prev[v] = u
        
        return distance, prev

    def min_cost_flow(self, s, t, required_flow):
        flow = 0
        cost = 0
        residual = [row[:] for row in self.capacity]

        while flow < required_flow:
            dist, prev = self.bellman_ford(s)
            if dist[t] == float('inf'):
                break

            path_flow = required_flow - flow
            v = t
            while v != s:
                u = prev[v]
                path_flow = min(path_flow, residual[u][v])
                v = u

            v = t
            while v != s:
                u = prev[v]
                residual[u][v] -= path_flow
                residual[v][u] += path_flow
                cost += path_flow * self.cost[u][v]
                v = u

            flow += path_flow

        return flow, cost

    def print_residual_graph(self):
        print("\nCurrent Residual Graph:")
        table = []
        for i in range(self.n):
            table.append(self.flow[i])
        print(tabulate(table, headers=[f"v{i}" for i in range(self.n)], tablefmt="grid"))


def read_file(filename, with_cost=False):
    # Ajouter le chemin du dossier input graphs
    filepath = f"input graphs/{filename}"
    with open(filepath, 'r') as file:
        n = int(file.readline())
        cap = [list(map(int, file.readline().split())) for _ in range(n)]
        if with_cost:
            file.readline()  # Ligne vide entre capacité et coût
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


def main():
    while True:
        try:
            print("\nFlow Network Algorithms")
            print("1. Solve Max Flow Problem (Ford-Fulkerson / Push-Relabel)")
            print("2. Solve Min-Cost Flow Problem")
            print("3. Exit")
            choice = input("Enter choice: ")

            if choice == '1':
                problem = input("Enter the problem number (1-10): ")
                print(f"Reading file proposal{problem}.txt...")
                capacity, cost = read_file(f"proposal{problem}.txt", with_cost=True)
                print("File read successfully")
                net = FlowNetwork(capacity, cost)
                algorithm_choice = input("Select algorithm (1. Ford-Fulkerson, 2. Push-Relabel): ")
                if algorithm_choice == '1':
                    print("Running Ford-Fulkerson...")
                    max_flow = net.ford_fulkerson(0, len(capacity)-1)
                    print(f"Max Flow (Ford-Fulkerson): {max_flow}")
                elif algorithm_choice == '2':
                    print("Running Push-Relabel...")
                    max_flow = net.push_relabel(0, len(capacity)-1)
                    print(f"Max Flow (Push-Relabel): {max_flow}")
            elif choice == '2':
                problem = input("Enter the problem number (1-10): ")
                capacity, cost = read_file(f"proposal{problem}.txt", with_cost=True)
                net = FlowNetwork(capacity, cost)
                required_flow = int(input("Enter required flow: "))
                flow, cost = net.min_cost_flow(0, len(capacity)-1, required_flow)
                print(f"Min-Cost Flow: {flow}, Cost: {cost}")
            elif choice == '3':
                break
            else:
                print("Invalid choice. Try again.")
        except Exception as e:
            print(f"Une erreur s'est produite : {str(e)}")

if __name__ == '__main__':
    main()
