import sys
import time
import random
from collections import deque
import heapq
from typing import List, Dict, Tuple, Set
import matplotlib.pyplot as plt
import numpy as np

class FlowNetwork:
    def __init__(self, n: int):
        self.n = n  # Number of vertices
        self.capacity = [[0] * n for _ in range(n)]  # Capacity matrix
        self.cost = [[0] * n for _ in range(n)]  # Cost matrix (for min-cost flow)
        self.source = 0  # Source vertex (v1)
        self.sink = n - 1  # Sink vertex (vn)
    
    def add_edge(self, u: int, v: int, cap: int, cost: int = 0):
        """Add an edge from u to v with given capacity and cost."""
        self.capacity[u][v] = cap
        self.cost[u][v] = cost
    
    def read_from_file(self, filename: str, has_cost: bool = False):
        """Read flow network from a text file."""
        with open(filename, 'r') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
            
            # Read n
            self.n = int(lines[0])
            self.source = 0
            self.sink = self.n - 1
            self.capacity = [[0] * self.n for _ in range(self.n)]
            self.cost = [[0] * self.n for _ in range(self.n)]
            
            # Read capacity matrix
            for i in range(self.n):
                parts = list(map(int, lines[1+i].split()))
                for j in range(self.n):
                    self.capacity[i][j] = parts[j]
            
            # Read cost matrix if present
            if has_cost:
                for i in range(self.n):
                    parts = list(map(int, lines[1+self.n+i].split()))
                    for j in range(self.n):
                        self.cost[i][j] = parts[j]
    
    def display_capacity_matrix(self):
        """Display the capacity matrix in a formatted table."""
        print("\nCapacity Matrix:")
        print("   " + " ".join(f"{i:3}" for i in range(self.n)))
        for i in range(self.n):
            print(f"{i:2} " + " ".join(f"{self.capacity[i][j]:3}" for j in range(self.n)))
    
    def display_cost_matrix(self):
        """Display the cost matrix in a formatted table."""
        print("\nCost Matrix:")
        print("   " + " ".join(f"{i:3}" for i in range(self.n)))
        for i in range(self.n):
            print(f"{i:2} " + " ".join(f"{self.cost[i][j]:3}" for j in range(self.n)))

class MaxFlowSolver:
    @staticmethod
    def ford_fulkerson(network: FlowNetwork, verbose: bool = False) -> Tuple[int, List[List[int]]]:
        """Ford-Fulkerson algorithm using BFS (Edmonds-Karp implementation)."""
        n = network.n
        s = network.source
        t = network.sink
        capacity = [row[:] for row in network.capacity]
        flow = [[0] * n for _ in range(n)]
        max_flow = 0
        
        if verbose:
            print("\nInitial Residual Graph:")
            network.display_capacity_matrix()
        
        iteration = 1
        while True:
            # BFS to find augmenting path
            parent = [-1] * n
            queue = deque()
            queue.append(s)
            parent[s] = -2  # Mark source as visited
            
            if verbose:
                print(f"\n* Iteration {iteration}:")
                print("Breadth-first search:")
                print(f"v{s}")
            
            found = False
            while queue and not found:
                u = queue.popleft()
                if verbose:
                    neighbors = []
                
                for v in range(n):
                    if capacity[u][v] > 0 and parent[v] == -1:
                        parent[v] = u
                        if verbose:
                            neighbors.append(f"v{v}")
                        if v == t:
                            found = True
                            break
                        queue.append(v)
                
                if verbose and neighbors:
                    print(" ".join(neighbors) + f" ; " + 
                          " ; ".join(f"Π(v{neighbor}) = v{u}" for neighbor in neighbors))
            
            if not found:
                break
            
            # Find minimum residual capacity along the path
            path_flow = float('inf')
            v = t
            path = []
            while v != s:
                path.append(v)
                u = parent[v]
                path_flow = min(path_flow, capacity[u][v])
                v = u
            path.append(s)
            path.reverse()
            
            if verbose:
                print(f"\nDetection of an improving chain: " + 
                      "->".join(f"v{node}" for node in path) + 
                      f" with a flow {path_flow}")
            
            # Update residual capacities and flow
            v = t
            while v != s:
                u = parent[v]
                capacity[u][v] -= path_flow
                capacity[v][u] += path_flow
                flow[u][v] += path_flow
                flow[v][u] -= path_flow
                v = u
            
            max_flow += path_flow
            
            if verbose:
                print("\nModifications to the residual graph:")
                print("   " + " ".join(f"{i:3}" for i in range(n)))
                for i in range(n):
                    print(f"{i:2} " + " ".join(f"{capacity[i][j]:3}" for j in range(n)))
                print(f"\nCurrent max flow value: {max_flow}")
            
            iteration += 1
        
        if verbose:
            print("\n★ Max flow display:")
            print("   " + " ".join(f"{i:3}" for i in range(n)))
            for i in range(n):
                print(f"{i:2} " + " ".join(
                    f"{flow[i][j]}/{network.capacity[i][j] if network.capacity[i][j] > 0 else 0:3}" 
                    for j in range(n)))
            print(f"\nValue of the max flow = {max_flow}")
        
        return max_flow, flow
    
    @staticmethod
    def push_relabel(network: FlowNetwork, verbose: bool = False) -> Tuple[int, List[List[int]]]:
        """Push-Relabel algorithm for max flow."""
        n = network.n
        s = network.source
        t = network.sink
        capacity = [row[:] for row in network.capacity]
        
        # Initialize height and excess flow
        height = [0] * n
        excess = [0] * n
        flow = [[0] * n for _ in range(n)]
        
        # Preflow: saturate all edges from source
        height[s] = n
        for v in range(n):
            if capacity[s][v] > 0:
                flow[s][v] = capacity[s][v]
                flow[v][s] = -flow[s][v]
                excess[v] = flow[s][v]
                excess[s] -= flow[s][v]
                capacity[s][v] -= flow[s][v]
                capacity[v][s] += flow[s][v]
        
        if verbose:
            print("\nInitial Preflow:")
            print("   " + " ".join(f"{i:3}" for i in range(n)))
            for i in range(n):
                print(f"{i:2} " + " ".join(f"{flow[i][j]:3}" for j in range(n)))
            print("\nInitial Heights:", height)
            print("Initial Excess:", excess)
        
        # List of active nodes (excluding source and sink)
        active_nodes = [v for v in range(n) if v != s and v != t and excess[v] > 0]
        iteration = 1
        
        while active_nodes:
            if verbose:
                print(f"\n* Iteration {iteration}:")
                print("Active nodes:", [f"v{v}" for v in active_nodes])
            
            u = active_nodes[0]
            pushed = False
            
            # Try to push flow to admissible edges
            for v in range(n):
                if capacity[u][v] > 0 and height[u] == height[v] + 1:
                    # Push as much flow as possible
                    push_amount = min(excess[u], capacity[u][v])
                    
                    if verbose:
                        print(f"Push {push_amount} from v{u} to v{v}")
                    
                    flow[u][v] += push_amount
                    flow[v][u] -= push_amount
                    excess[u] -= push_amount
                    excess[v] += push_amount
                    capacity[u][v] -= push_amount
                    capacity[v][u] += push_amount
                    
                    pushed = True
                    
                    # Add v to active nodes if it's not s or t and now has excess
                    if v != s and v != t and excess[v] > 0 and v not in active_nodes:
                        active_nodes.append(v)
                    
                    if excess[u] == 0:
                        break
            
            if not pushed:
                # Relabel if we couldn't push
                min_height = float('inf')
                for v in range(n):
                    if capacity[u][v] > 0:
                        min_height = min(min_height, height[v])
                
                if min_height != float('inf'):
                    if verbose:
                        print(f"Relabel v{u} from {height[u]} to {min_height + 1}")
                    height[u] = min_height + 1
            
            # Remove u from active nodes if it has no more excess
            if excess[u] == 0:
                active_nodes.pop(0)
            
            iteration += 1
            
            if verbose:
                print("\nCurrent flow:")
                print("   " + " ".join(f"{i:3}" for i in range(n)))
                for i in range(n):
                    print(f"{i:2} " + " ".join(f"{flow[i][j]:3}" for j in range(n)))
                print("\nHeights:", height)
                print("Excess:", excess)
        
        max_flow = sum(flow[s][v] for v in range(n))
        
        if verbose:
            print("\n★ Max flow display:")
            print("   " + " ".join(f"{i:3}" for i in range(n)))
            for i in range(n):
                print(f"{i:2} " + " ".join(
                    f"{flow[i][j]}/{network.capacity[i][j] if network.capacity[i][j] > 0 else 0:3}" 
                    for j in range(n)))
            print(f"\nValue of the max flow = {max_flow}")
        
        return max_flow, flow

class MinCostFlowSolver:
    @staticmethod
    def min_cost_flow(network: FlowNetwork, desired_flow: int, verbose: bool = False) -> Tuple[int, List[List[int]]]:
        """Find minimum cost flow using successive shortest paths with Bellman-Ford."""
        n = network.n
        s = network.source
        t = network.sink
        capacity = [row[:] for row in network.capacity]
        cost = [row[:] for row in network.cost]
        
        # Initialize flow and residual network
        flow = [[0] * n for _ in range(n)]
        current_flow = 0
        total_cost = 0
        
        if verbose:
            print("\nInitial Residual Graph:")
            print("Capacity:")
            print("   " + " ".join(f"{i:3}" for i in range(n)))
            for i in range(n):
                print(f"{i:2} " + " ".join(f"{capacity[i][j]:3}" for j in range(n)))
            print("\nCost:")
            print("   " + " ".join(f"{i:3}" for i in range(n)))
            for i in range(n):
                print(f"{i:2} " + " ".join(f"{cost[i][j]:3}" for j in range(n)))
        
        iteration = 1
        while current_flow < desired_flow:
            # Run Bellman-Ford to find shortest path in residual network
            distance = [float('inf')] * n
            parent = [-1] * n
            distance[s] = 0
            
            if verbose:
                print(f"\n* Iteration {iteration}:")
                print("Bellman-Ford algorithm:")
                print(f"Initial distances: {distance}")
            
            # Relax all edges up to n-1 times
            for _ in range(n - 1):
                updated = False
                for u in range(n):
                    for v in range(n):
                        if capacity[u][v] > 0 and distance[v] > distance[u] + cost[u][v]:
                            distance[v] = distance[u] + cost[u][v]
                            parent[v] = u
                            updated = True
                
                if verbose:
                    print(f"After relaxation {_ + 1}: {distance}")
                
                if not updated:
                    break
            
            # Check for negative cycles (shouldn't happen with conservative costs)
            for u in range(n):
                for v in range(n):
                    if capacity[u][v] > 0 and distance[v] > distance[u] + cost[u][v]:
                        raise ValueError("Negative cost cycle detected")
            
            if distance[t] == float('inf'):
                break  # No more augmenting paths
            
            if verbose:
                print("\nBellman-Ford result:")
                print("Distances:", distance)
                print("Parents:", parent)
            
            # Find the path and minimum residual capacity
            path_flow = float('inf')
            v = t
            path = []
            while v != s:
                path.append(v)
                u = parent[v]
                path_flow = min(path_flow, capacity[u][v])
                v = u
            path.append(s)
            path.reverse()
            
            # Adjust the flow to not exceed desired_flow
            path_flow = min(path_flow, desired_flow - current_flow)
            
            if verbose:
                print(f"\nFound augmenting path: " + "->".join(f"v{node}" for node in path) + 
                      f" with residual capacity {path_flow}")
                print(f"Path cost: {distance[t]}")
            
            # Update flow and residual network
            v = t
            while v != s:
                u = parent[v]
                flow[u][v] += path_flow
                flow[v][u] -= path_flow
                capacity[u][v] -= path_flow
                capacity[v][u] += path_flow
                v = u
            
            current_flow += path_flow
            total_cost += path_flow * distance[t]
            
            if verbose:
                print("\nUpdated flow:")
                print("   " + " ".join(f"{i:3}" for i in range(n)))
                for i in range(n):
                    print(f"{i:2} " + " ".join(f"{flow[i][j]:3}" for j in range(n)))
                print("\nUpdated residual capacities:")
                print("   " + " ".join(f"{i:3}" for i in range(n)))
                for i in range(n):
                    print(f"{i:2} " + " ".join(f"{capacity[i][j]:3}" for j in range(n)))
                print(f"\nCurrent flow: {current_flow}/{desired_flow}")
                print(f"Current total cost: {total_cost}")
            
            iteration += 1
        
        if verbose:
            print("\n★ Final flow:")
            print("   " + " ".join(f"{i:3}" for i in range(n)))
            for i in range(n):
                print(f"{i:2} " + " ".join(
                    f"{flow[i][j]}/{network.capacity[i][j] if network.capacity[i][j] > 0 else 0:3}" 
                    for j in range(n)))
            print(f"\nTotal flow achieved: {current_flow}/{desired_flow}")
            print(f"Total minimum cost: {total_cost}")
        
        return total_cost, flow

class ComplexityAnalyzer:
    @staticmethod
    def generate_random_network(n: int) -> FlowNetwork:
        """Generate a random flow network with n vertices."""
        network = FlowNetwork(n)
        
        # Generate capacity matrix
        num_edges = (n * n) // 2  # About half of possible edges
        edges_added = 0
        
        while edges_added < num_edges:
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u != v and network.capacity[u][v] == 0:
                cap = random.randint(1, 100)
                cost = random.randint(1, 100)
                network.add_edge(u, v, cap, cost)
                edges_added += 1
        
        return network
    
    @staticmethod
    def measure_complexity(max_n: int = 100, samples: int = 100):
        """Measure execution times for different algorithms and network sizes."""
        sizes = [10, 20, 40, 100]
        results = {
            'ford_fulkerson': {n: [] for n in sizes},
            'push_relabel': {n: [] for n in sizes},
            'min_cost_flow': {n: [] for n in sizes}
        }
        
        for n in sizes:
            print(f"\nTesting networks of size {n}...")
            for _ in range(samples):
                network = ComplexityAnalyzer.generate_random_network(n)
                
                # Measure Ford-Fulkerson
                start = time.time()
                MaxFlowSolver.ford_fulkerson(network)
                end = time.time()
                results['ford_fulkerson'][n].append(end - start)
                
                # Measure Push-Relabel
                start = time.time()
                MaxFlowSolver.push_relabel(network)
                end = time.time()
                results['push_relabel'][n].append(end - start)
                
                # Measure Min-Cost Flow (with half of max flow as target)
                max_flow, _ = MaxFlowSolver.ford_fulkerson(network)
                desired_flow = max_flow // 2
                start = time.time()
                MinCostFlowSolver.min_cost_flow(network, desired_flow)
                end = time.time()
                results['min_cost_flow'][n].append(end - start)
        
        return results
    
    @staticmethod
    def plot_results(results: dict):
        """Plot the complexity analysis results."""
        sizes = sorted(results['ford_fulkerson'].keys())
        
        plt.figure(figsize=(15, 5))
        
        # Plot all samples
        plt.subplot(1, 3, 1)
        for n in sizes:
            plt.scatter([n] * len(results['ford_fulkerson'][n]), results['ford_fulkerson'][n], alpha=0.5)
        plt.title('Ford-Fulkerson')
        plt.xlabel('Network size (n)')
        plt.ylabel('Execution time (s)')
        plt.grid(True)
        
        plt.subplot(1, 3, 2)
        for n in sizes:
            plt.scatter([n] * len(results['push_relabel'][n]), results['push_relabel'][n], alpha=0.5)
        plt.title('Push-Relabel')
        plt.xlabel('Network size (n)')
        plt.ylabel('Execution time (s)')
        plt.grid(True)
        
        plt.subplot(1, 3, 3)
        for n in sizes:
            plt.scatter([n] * len(results['min_cost_flow'][n]), results['min_cost_flow'][n], alpha=0.5)
        plt.title('Min-Cost Flow')
        plt.xlabel('Network size (n)')
        plt.ylabel('Execution time (s)')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('complexity_analysis.png')
        plt.show()
        
        # Plot worst-case (max times) and theoretical complexities
        plt.figure(figsize=(15, 5))
        
        # Ford-Fulkerson: O(E*max_flow) ~ O(n^3) in practice
        max_ff = [max(results['ford_fulkerson'][n]) for n in sizes]
        plt.subplot(1, 3, 1)
        plt.plot(sizes, max_ff, 'o-', label='Measured')
        plt.plot(sizes, [n**3 * max_ff[0]/sizes[0]**3 for n in sizes], '--', label='O(n³)')
        plt.title('Ford-Fulkerson Worst Case')
        plt.xlabel('Network size (n)')
        plt.ylabel('Max execution time (s)')
        plt.legend()
        plt.grid(True)
        
        # Push-Relabel: O(n^3)
        max_pr = [max(results['push_relabel'][n]) for n in sizes]
        plt.subplot(1, 3, 2)
        plt.plot(sizes, max_pr, 'o-', label='Measured')
        plt.plot(sizes, [n**3 * max_pr[0]/sizes[0]**3 for n in sizes], '--', label='O(n³)')
        plt.title('Push-Relabel Worst Case')
        plt.xlabel('Network size (n)')
        plt.ylabel('Max execution time (s)')
        plt.legend()
        plt.grid(True)
        
        # Min-Cost Flow: O(n^4) (Bellman-Ford for each unit of flow)
        max_mcf = [max(results['min_cost_flow'][n]) for n in sizes]
        plt.subplot(1, 3, 3)
        plt.plot(sizes, max_mcf, 'o-', label='Measured')
        plt.plot(sizes, [n**4 * max_mcf[0]/sizes[0]**4 for n in sizes], '--', label='O(n⁴)')
        plt.title('Min-Cost Flow Worst Case')
        plt.xlabel('Network size (n)')
        plt.ylabel('Max execution time (s)')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('worst_case_complexity.png')
        plt.show()

def main():
    print("Flow Problem Solver")
    print("===================")
    
    while True:
        print("\nMenu:")
        print("1. Solve max flow problem")
        print("2. Solve min-cost flow problem")
        print("3. Run complexity analysis")
        print("4. Exit")
        
        choice = input("Enter your choice (1-4): ")
        
        if choice == '1':
            filename = input("Enter the filename for the flow network: ")
            has_cost = input("Does the file include cost matrix? (y/n): ").lower() == 'y'
            
            network = FlowNetwork(0)  # Temporary, will be initialized by read_from_file
            try:
                network.read_from_file(filename, has_cost)
                print("\nNetwork loaded successfully:")
                network.display_capacity_matrix()
                if has_cost:
                    network.display_cost_matrix()
                
                print("\nSelect algorithm:")
                print("1. Ford-Fulkerson (with BFS)")
                print("2. Push-Relabel")
                algo_choice = input("Enter your choice (1-2): ")
                
                verbose = input("Show detailed execution trace? (y/n): ").lower() == 'y'
                
                if algo_choice == '1':
                    print("\nRunning Ford-Fulkerson algorithm...")
                    max_flow, flow = MaxFlowSolver.ford_fulkerson(network, verbose)
                    print(f"\nFinal max flow value: {max_flow}")
                elif algo_choice == '2':
                    print("\nRunning Push-Relabel algorithm...")
                    max_flow, flow = MaxFlowSolver.push_relabel(network, verbose)
                    print(f"\nFinal max flow value: {max_flow}")
                else:
                    print("Invalid choice")
                
                # Save execution trace to file
                trace_filename = input("Enter filename to save execution trace (leave empty to skip): ")
                if trace_filename:
                    original_stdout = sys.stdout
                    with open(trace_filename, 'w') as f:
                        sys.stdout = f
                        if algo_choice == '1':
                            MaxFlowSolver.ford_fulkerson(network, True)
                        elif algo_choice == '2':
                            MaxFlowSolver.push_relabel(network, True)
                        sys.stdout = original_stdout
                    print(f"Execution trace saved to {trace_filename}")
            
            except FileNotFoundError:
                print(f"Error: File '{filename}' not found.")
            except Exception as e:
                print(f"Error: {str(e)}")
        
        elif choice == '2':
            filename = input("Enter the filename for the flow network (with costs): ")
            
            network = FlowNetwork(0)
            try:
                network.read_from_file(filename, True)
                print("\nNetwork loaded successfully:")
                network.display_capacity_matrix()
                network.display_cost_matrix()
                
                desired_flow = int(input("Enter the desired flow value: "))
                verbose = input("Show detailed execution trace? (y/n): ").lower() == 'y'
                
                print("\nRunning Min-Cost Flow algorithm...")
                total_cost, flow = MinCostFlowSolver.min_cost_flow(network, desired_flow, verbose)
                print(f"\nTotal cost for flow {desired_flow}: {total_cost}")
                
                # Save execution trace to file
                trace_filename = input("Enter filename to save execution trace (leave empty to skip): ")
                if trace_filename:
                    original_stdout = sys.stdout
                    with open(trace_filename, 'w') as f:
                        sys.stdout = f
                        MinCostFlowSolver.min_cost_flow(network, desired_flow, True)
                        sys.stdout = original_stdout
                    print(f"Execution trace saved to {trace_filename}")
            
            except FileNotFoundError:
                print(f"Error: File '{filename}' not found.")
            except ValueError as e:
                print(f"Error: {str(e)}")
            except Exception as e:
                print(f"Error: {str(e)}")
        
        elif choice == '3':
            print("\nRunning complexity analysis...")
            print("This may take several minutes...")
            results = ComplexityAnalyzer.measure_complexity(samples=10)  # Reduced samples for demo
            ComplexityAnalyzer.plot_results(results)
            print("Complexity analysis completed. Plots saved as:")
            print("- complexity_analysis.png")
            print("- worst_case_complexity.png")
        
        elif choice == '4':
            print("Exiting...")
            break
        
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()