"""
PSO-CPP Solver
==============

Particle Swarm Optimization (PSO) approach to the Chinese Postman Problem (CPP),
benchmarked against an exact recursive (brute-force) solution.

Pipeline
--------
1. Generate a random connected weighted graph with a target number of odd-degree
   nodes (the "Odd Nodes" that must be optimally matched to make the graph
   Eulerian).
2. Compute the exact optimal matching cost with a recursive algorithm based on
   Dijkstra shortest paths (ground truth / benchmark).
3. Run a PSO metaheuristic to find a (near-)optimal matching of the odd nodes,
   tracking convergence over iterations.
4. Plot the PSO convergence curve against the recursive optimum.

Author: Giacinto Angelo Sgarro
License: MIT (see LICENSE file). Note: the recursive odd-node pairing
algorithm (Section 3 below) is adapted from third-party work — see
attribution notice in that section.
"""

import random

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# ===========================================================================================
# 1. GRAPH GENERATION
# ===========================================================================================


def generate_graph(num_nodes, max_weight, target_odd_nodes, max_edges):
    """
    Generate a random connected, weighted graph with a target number of
    odd-degree nodes.

    Parameters
    ----------
    num_nodes : int
        Total number of nodes in the graph.
    max_weight : int
        Maximum edge weight (weights are drawn uniformly from [1, max_weight]).
    target_odd_nodes : int
        Desired number of odd-degree nodes.
    max_edges : int
        Upper bound on the number of edges to add.

    Returns
    -------
    edge_list : list of [int, int, int]
        Edge list as [source, target, weight] triples (1-indexed nodes).
    adjacency_matrix : list of list of int
        Weighted adjacency matrix (1-indexed nodes, 0-indexed matrix).
    num_odd_nodes : int
        Number of odd-degree nodes actually obtained.
    """
    if target_odd_nodes > num_nodes:
        raise ValueError("The number of odd nodes cannot exceed the total number of nodes")

    if max_edges <= (2 * num_nodes - 2):
        raise ValueError(f"N_Edges must be greater than {2 * num_nodes - 2}!")

    # --- Ensure the graph is connected -----------------------------------------------------
    # Build a Hamiltonian cycle so the graph is guaranteed to be connected from the start.
    G = nx.Graph()
    G.add_nodes_from(list(range(1, num_nodes + 1)))
    for i in range(1, num_nodes):
        G.add_edge(i, i + 1, weight=random.randint(1, max_weight))
    G.add_edge(num_nodes, 1, weight=random.randint(1, max_weight))

    # --- Add random edges until the target number of odd nodes (or edge cap) is reached ----
    num_edges = G.number_of_edges()
    odd_nodes = [n for n in G.nodes if G.degree(n) % 2 != 0]
    num_odd_nodes = len(odd_nodes)

    while num_odd_nodes < target_odd_nodes and num_edges < max_edges:
        i, j = random.sample(list(range(1, num_nodes + 1)), 2)
        if i != j and not G.has_edge(i, j) and not G.has_edge(j, i):
            G.add_edge(i, j, weight=random.randint(1, max_weight))
            num_edges = G.number_of_edges()
            odd_nodes = [n for n in range(1, num_nodes + 1) if G.degree(n) % 2 != 0]
            num_odd_nodes = len(odd_nodes)

    # --- Final check -------------------------------------------------------------------------
    odd_nodes = [n for n in range(1, num_nodes + 1) if G.degree(n) % 2 != 0]
    num_odd_nodes = len(odd_nodes)

    # --- Convert to numpy adjacency matrix / edge list ---------------------------------------
    adj_matrix = nx.to_numpy_array(G, weight='weight')
    rows, cols = np.where(adj_matrix > 0)
    edge_array = np.array(list(zip(rows + 1, cols + 1, adj_matrix[rows, cols])))

    adjacency_matrix = [[int(v) for v in row] for row in adj_matrix.tolist()]
    edge_list = [[int(v) for v in edge] for edge in edge_array.tolist()]

    return edge_list, adjacency_matrix, num_odd_nodes


def get_odd_nodes(graph):
    """
    Return the indices of odd-degree nodes in a weighted adjacency matrix.

    Parameters
    ----------
    graph : list of list of int/float
        Weighted adjacency matrix.

    Returns
    -------
    list of int
        Indices (0-based) of the odd-degree nodes.
    """
    degrees = [0 for _ in range(len(graph))]
    for i in range(len(graph)):
        for j in range(len(graph)):
            if graph[i][j] != 0:
                degrees[i] += 1

    return [i for i in range(len(degrees)) if degrees[i] % 2 != 0]


def roulette_wheel_selection(p):
    """
    Perform roulette-wheel selection on a probability matrix, returning the
    (row, column) indices of the selected cell.

    Parameters
    ----------
    p : numpy.ndarray
        A (normalized) probability matrix.

    Returns
    -------
    (int, int)
        Row and column indices of the selected entry.
    """
    p_vector = p.flatten(order='C')
    cumulative_probabilities = np.cumsum(p_vector)

    random_value = random.random()  # uniform in [0, 1)
    while random_value == 0:
        random_value = random.uniform(0, 1)  # avoid selecting (start == end) = 0

    smaller_indices = list(np.where(cumulative_probabilities < random_value)[0])
    if not smaller_indices:
        flat_index = 0
    else:
        flat_index = smaller_indices[-1] + 1

    i = int(flat_index / len(p))
    j = flat_index % len(p)

    return i, j


# ===========================================================================================
# 2. SHORTEST PATHS (DIJKSTRA)
# ===========================================================================================


def dijkstra(graph, source, dest):
    """
    Compute the shortest path distance between two nodes using Dijkstra's algorithm.

    Parameters
    ----------
    graph : list of list of int/float
        Weighted adjacency matrix (0 indicates no edge).
    source : int
        Source node index.
    dest : int
        Destination node index.

    Returns
    -------
    int/float
        Shortest path distance from source to dest.
    """
    n = len(graph)
    shortest = [0 for _ in range(n)]
    selected = [source]
    infinity = 10_000_000
    min_selected = infinity
    closest_node = source

    if source == dest:
        return 0

    for i in range(n):
        if i == source:
            shortest[source] = 0
        else:
            if graph[source][i] == 0:
                shortest[i] = infinity
            else:
                shortest[i] = graph[source][i]
                if shortest[i] < min_selected:
                    min_selected = shortest[i]
                    closest_node = i

    selected.append(closest_node)
    while closest_node != dest:
        for i in range(n):
            if i not in selected:
                if graph[closest_node][i] != 0:
                    if (graph[closest_node][i] + min_selected) < shortest[i]:
                        shortest[i] = graph[closest_node][i] + min_selected

        temp_min = 1_000_000
        for j in range(n):
            if j not in selected:
                if shortest[j] < temp_min:
                    temp_min = shortest[j]
                    closest_node = j

        min_selected = temp_min
        selected.append(closest_node)

    return shortest[dest]


# ===========================================================================================
# 3. EXACT RECURSIVE ALGORITHM (BRUTE-FORCE OPTIMAL MATCHING)
# ===========================================================================================
#
# The recursive odd-node pairing algorithm below (generate_pairs, get_pairings,
# shortest_paths_sum, recursive_algorithm) is adapted from:
#
#   Araz Sharma, "Chinese Postman in Python", Towards Data Science, Nov 9 2020.
#   https://towardsdatascience.com/chinese-postman-in-python-45e9987e1b7d
#
# All credit for the original recursive pairing-generation approach goes to the
# author above; this implementation follows the same logic with adapted
# naming and integration into the PSO pipeline.


def generate_pairs(odd_nodes):
    """
    Build all candidate (i, j) pairs of odd nodes, organized by "column"
    (i.e. grouped by the first node of the pair).

    Parameters
    ----------
    odd_nodes : list of int
        Indices of the odd-degree nodes.

    Returns
    -------
    list of list of [int, int]
        Candidate pairs grouped by first element.
    """
    pairs = []
    for i in range(len(odd_nodes) - 1):
        pairs.append([])
        for j in range(i + 1, len(odd_nodes)):
            pairs[i].append([odd_nodes[i], odd_nodes[j]])

    return pairs


def get_pairings(pairs, pairing_length, used_nodes=None, current_pairing=None, all_pairings=None):
    """
    Recursively enumerate all valid perfect matchings of the odd nodes.

    Parameters
    ----------
    pairs : list of list of [int, int]
        Candidate pairs, as returned by `generate_pairs`.
    pairing_length : int
        Number of pairs required for a complete matching.
    used_nodes : list of int, optional
        Nodes already used in the current partial matching (internal, recursive use).
    current_pairing : list of [int, int], optional
        Pairs selected so far (internal, recursive use).
    all_pairings : list of list of [int, int], optional
        Accumulator for all complete matchings found (internal, recursive use).

    Returns
    -------
    list of list of [int, int]
        All complete perfect matchings of the odd nodes.
    """
    if used_nodes is None:
        used_nodes = []
    if current_pairing is None:
        current_pairing = []
    if all_pairings is None:
        all_pairings = []

    if pairs[0][0][0] not in used_nodes:
        used_nodes.append(pairs[0][0][0])

        for candidate in pairs[0]:
            new_pairing = current_pairing[:]
            new_used_nodes = used_nodes[:]

            if candidate[1] not in new_used_nodes:
                new_pairing.append(candidate)
            else:
                continue

            if len(new_pairing) == pairing_length:
                all_pairings.append(new_pairing)
                return all_pairings
            else:
                new_used_nodes.append(candidate[1])
                get_pairings(pairs[1:], pairing_length, new_used_nodes, new_pairing, all_pairings)
    else:
        get_pairings(pairs[1:], pairing_length, used_nodes, current_pairing, all_pairings)

    return all_pairings


def shortest_paths_sum(graph, pairings, sums=None):
    """
    Compute the total shortest-path cost for each candidate matching.

    Parameters
    ----------
    graph : list of list of int/float
        Weighted adjacency matrix.
    pairings : list of list of [int, int]
        Candidate perfect matchings.
    sums : list of float, optional
        Accumulator for the resulting costs (internal, recursive use).

    Returns
    -------
    list of float
        Total cost of each candidate matching.
    """
    if sums is None:
        sums = []

    for pairing in pairings:
        total = 0
        for pair in pairing:
            total += dijkstra(graph, pair[0], pair[1])
        sums.append(total)

    return sums


def recursive_algorithm(odd_nodes, graph):
    """
    Compute the exact minimum-cost perfect matching of the odd-degree nodes
    via exhaustive enumeration (brute force).

    Parameters
    ----------
    odd_nodes : list of int
        Indices of the odd-degree nodes.
    graph : list of list of int/float
        Weighted adjacency matrix.

    Returns
    -------
    best_cost : float
        Minimum total matching cost.
    best_tour : list of [int, int]
        Optimal pairing of odd nodes (1-indexed to match the original graph).
    """
    pairs = generate_pairs(odd_nodes)

    pairing_length = (len(pairs) + 1) // 2
    all_pairings = get_pairings(pairs, pairing_length)

    all_costs = shortest_paths_sum(graph, all_pairings, sums=[])
    best_cost = np.min(all_costs)

    best_index = all_costs.index(best_cost)
    best_tour = all_pairings[best_index]

    # Shift indices by 1 to realign with the original (1-indexed) graph nodes
    for pair in best_tour:
        for i in range(len(pair)):
            pair[i] += 1

    return best_cost, best_tour


# ===========================================================================================
# 4. PSO HELPER FUNCTIONS
# ===========================================================================================


def build_distance_matrix(odd_nodes, graph):
    """
    Build the pairwise shortest-path distance matrix between all odd nodes.

    Parameters
    ----------
    odd_nodes : list of int
        Indices of the odd-degree nodes.
    graph : list of list of int/float
        Weighted adjacency matrix.

    Returns
    -------
    numpy.ndarray
        Distance matrix of shape (len(odd_nodes), len(odd_nodes)).
    """
    D = np.zeros((len(odd_nodes), len(odd_nodes)))

    for i in range(D.shape[0]):
        for j in range(D.shape[0]):
            if i != j:
                D[i][j] = dijkstra(graph, odd_nodes[i], odd_nodes[j])

    return D


def generate_random_matching(n_var):
    """
    Generate a random permutation used as an initial (perfect) matching.

    Parameters
    ----------
    n_var : int
        Number of odd nodes (must be even).

    Returns
    -------
    list of int
        Random permutation of node indices [0, n_var).
    """
    matching = list(range(n_var))
    random.shuffle(matching)
    return matching


def compute_matching_cost(tour, D):
    """
    Compute the total cost of a matching, given as a flat list of paired indices.

    Parameters
    ----------
    tour : list of int
        Flat matching representation: [i1, j1, i2, j2, ...].
    D : numpy.ndarray
        Pairwise distance matrix.

    Returns
    -------
    float
        Total matching cost.
    """
    cost = 0
    for t in range(0, len(tour), 2):
        i, j = tour[t], tour[t + 1]
        cost += D[i][j]
    return cost


def get_value(mode, matrix_val=None):
    """
    Sample or select a velocity/weight contribution value depending on the
    chosen mode.

    Parameters
    ----------
    mode : str
        One of 'none', 'cell', or 'matrix'.
    matrix_val : float, optional
        Precomputed value to use when mode == 'matrix'.

    Returns
    -------
    float
        The resulting value (0 if mode == 'none').
    """
    if mode == 'none':
        return 0
    elif mode == 'cell':
        return random.uniform(0.5, 1.5)
    elif mode == 'matrix':
        return matrix_val if matrix_val is not None else random.uniform(0.5, 1.5)
    return 1.0


class Particle:
    """A single particle in the swarm, representing a candidate matching."""

    def __init__(self, velocity_matrix, tour, cost):
        self.velocity_matrix = velocity_matrix
        self.tour = tour
        self.cost = cost
        self.pbest_tour = tour
        self.pbest_cost = cost


# ===========================================================================================
# 5. MAIN EXECUTION
# ===========================================================================================


def main():
    # -----------------------------------------------------------------------------------------
    # CHOICE 1: GRAPH SIZE
    # -----------------------------------------------------------------------------------------
    target_odd_nodes = 8  # 8, 10, 12

    # Generation parameters corresponding to each size (do not modify)
    graph_size_params = {
        8: {'num_nodes': 18, 'max_edges': 60},
        10: {'num_nodes': 20, 'max_edges': 60},
        12: {'num_nodes': 24, 'max_edges': 70},
    }
    num_nodes = graph_size_params[target_odd_nodes]['num_nodes']
    max_edges = graph_size_params[target_odd_nodes]['max_edges']

    max_weight = 1000

    # -----------------------------------------------------------------------------------------
    # CHOICE 2: r0 / r1 / r2 / EXTRA TERM MODES
    # -----------------------------------------------------------------------------------------
    r0_mode = 'none'       # 'none', 'cell', 'matrix'
    r1_mode = 'cell'       # 'cell', 'matrix'
    r2_mode = 'matrix'     # 'cell', 'matrix'
    extra_mode = 'random'  # 'eta', 'uniform', 'random'

    # -----------------------------------------------------------------------------------------
    # CHOICE 3: PSO COEFFICIENTS
    # -----------------------------------------------------------------------------------------
    w = 0.7                # 0.5, 0.7, 0.9
    c1 = 1.5                # 1.0, 1.5, 2.0
    c2 = 1.5                # 1.0, 1.5, 2.0
    uniform_coeff = 1        # used only if extra_mode == 'uniform'
    eta_coeff = 0.3           # used only if extra_mode == 'eta'
    random_coeff = 0.1        # used only if extra_mode == 'random'

    # -----------------------------------------------------------------------------------------
    # CHOICE 4: SWARM SIZE / ITERATIONS
    # -----------------------------------------------------------------------------------------
    n_particles = 10  # 10, 20, 30, 40
    max_iter = 10      # 10, 20, 30, 40

    # -----------------------------------------------------------------------------------------
    # GRAPH GENERATION AND EXACT OPTIMUM (computed once)
    # -----------------------------------------------------------------------------------------
    print(f"Generating graph with {target_odd_nodes} odd nodes...")

    _, graph, num_odds = generate_graph(num_nodes, max_weight, target_odd_nodes, max_edges)

    while num_odds != target_odd_nodes:
        print(f"  num_odds={num_odds} != target. Regenerating graph...")
        _, graph, num_odds = generate_graph(num_nodes, max_weight, target_odd_nodes, max_edges)

    odd_nodes = get_odd_nodes(graph)

    print("Computing exact optimum (recursive algorithm)...")
    optimal_cost, _ = recursive_algorithm(odd_nodes, graph)
    print(f"Global optimum: {optimal_cost}")

    # -----------------------------------------------------------------------------------------
    # PSO EXECUTION WITH CONVERGENCE TRACKING (single run)
    # -----------------------------------------------------------------------------------------
    n_var = len(odd_nodes)
    D = build_distance_matrix(odd_nodes, graph)

    eta_matrix = 1 / (D + 0.00001)
    np.fill_diagonal(eta_matrix, 0)

    # --- Swarm initialization ------------------------------------------------------------------
    swarm = []
    for _ in range(n_particles):
        initial_tour = generate_random_matching(n_var)
        initial_cost = compute_matching_cost(initial_tour, D)

        velocity_matrix = np.zeros((n_var, n_var))
        r_vel = random.uniform(0.5, 1.5) if r0_mode == 'matrix' else None
        for t in range(0, len(initial_tour), 2):
            i, j = initial_tour[t], initial_tour[t + 1]
            val = get_value(r0_mode, r_vel)
            if val != 0:
                velocity_matrix[i][j] = val
                velocity_matrix[j][i] = val

        swarm.append(Particle(velocity_matrix, initial_tour, initial_cost))

    gbest_particle = min(swarm, key=lambda particle: particle.cost)
    gbest_tour = gbest_particle.tour
    gbest_cost = gbest_particle.cost

    cost_history = np.zeros((max_iter, 2))  # column 0 = MeanCost, column 1 = BestCost

    # --- Main PSO loop ---------------------------------------------------------------------------
    for it in range(max_iter):
        current_costs = []

        for particle in swarm:
            # Personal-best attraction matrix
            m_pbest = np.zeros((n_var, n_var))
            r_pbest = random.uniform(0.5, 1.5) if r1_mode == 'matrix' else None
            for t in range(0, len(particle.pbest_tour), 2):
                i, j = particle.pbest_tour[t], particle.pbest_tour[t + 1]
                val = get_value(r1_mode, r_pbest)
                if val != 0:
                    m_pbest[i][j] = val
                    m_pbest[j][i] = val

            # Global-best attraction matrix
            m_gbest = np.zeros((n_var, n_var))
            r_gbest = random.uniform(0.5, 1.5) if r2_mode == 'matrix' else None
            for t in range(0, len(gbest_tour), 2):
                i, j = gbest_tour[t], gbest_tour[t + 1]
                val = get_value(r2_mode, r_gbest)
                if val != 0:
                    m_gbest[i][j] = val
                    m_gbest[j][i] = val

            # Extra exploration/exploitation term
            uniform_weight = max(0.1, 1.0 - it / max_iter)
            m_uniform = uniform_weight * np.ones((n_var, n_var)) / n_var
            np.fill_diagonal(m_uniform, 0)

            m_random = np.random.uniform(0.5, 1.5, (n_var, n_var))
            np.fill_diagonal(m_random, 0)

            # Combined selection probability matrix
            P = (w * particle.velocity_matrix) + (c1 * m_pbest) + (c2 * m_gbest)

            if extra_mode == 'eta':
                P += eta_coeff * eta_matrix
            elif extra_mode == 'uniform':
                P += uniform_coeff * m_uniform
            elif extra_mode == 'random':
                P += random_coeff * m_random

            if np.sum(P) > 0:
                P = P / np.sum(P)

            # Build a new matching via roulette-wheel selection
            new_tour = []
            temp_P = P.copy()
            for _ in range(int(n_var / 2)):
                temp_P = temp_P / np.sum(temp_P)
                i, j = roulette_wheel_selection(temp_P)
                new_tour.append(i)
                new_tour.append(j)
                temp_P[i] = 0
                temp_P[:, i] = 0
                temp_P[j] = 0
                temp_P[:, j] = 0

            new_cost = compute_matching_cost(new_tour, D)
            current_costs.append(new_cost)

            new_velocity_matrix = np.zeros((n_var, n_var))
            r_new_vel = random.uniform(0.5, 1.5) if r0_mode == 'matrix' else None
            for t in range(0, len(new_tour), 2):
                i, j = new_tour[t], new_tour[t + 1]
                val = get_value(r0_mode, r_new_vel)
                if val != 0:
                    new_velocity_matrix[i][j] = val
                    new_velocity_matrix[j][i] = val

            particle.velocity_matrix = new_velocity_matrix
            particle.tour = new_tour
            particle.cost = new_cost

            if new_cost < particle.pbest_cost:
                particle.pbest_tour = new_tour.copy()
                particle.pbest_cost = new_cost

            if new_cost < gbest_cost:
                gbest_tour = new_tour.copy()
                gbest_cost = new_cost

        cost_history[it][0] = np.mean(current_costs)
        cost_history[it][1] = np.min(current_costs)

    print(f"Best cost found by PSO: {gbest_cost}")
    print(f"Success (PSO == optimum): {gbest_cost == optimal_cost}")

    # -----------------------------------------------------------------------------------------
    # CONVERGENCE PLOT WITH RECURSIVE OPTIMUM REFERENCE LINE
    # -----------------------------------------------------------------------------------------
    iterations = np.arange(1, max_iter + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(iterations, cost_history[:, 0], label='Mean Cost', color='tab:blue')
    plt.plot(iterations, cost_history[:, 1], label='Best Cost', color='tab:orange')
    plt.axhline(y=optimal_cost, color='green', linestyle='--', label='Recursive Optimum')

    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.title(
        f'PSO-CPP Convergence ({target_odd_nodes} Odd Nodes, '
        f'nParticles={n_particles}, MaxIt={max_iter})'
    )
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(
        f'convergence_{target_odd_nodes}nodes_{n_particles}p_{max_iter}it.png',
        dpi=300
    )
    plt.show()


if __name__ == "__main__":
    main()