# PSO_CPP

Particle Swarm Optimization approach to the **Chinese Postman Problem** (also known as the *Route Inspection Problem*), benchmarked against an exact recursive brute-force solution.

## Overview

Given a connected, undirected, weighted graph, the Chinese Postman Problem asks for the minimum-cost closed walk that traverses every edge at least once. The classical solution requires:

1. Identifying all odd-degree nodes in the graph.
2. Finding the minimum-cost perfect matching between those odd nodes (via shortest paths).
3. Adding the matched shortest paths as duplicate edges to make the graph Eulerian, then computing the total cost.

This project focuses on step 2 — the odd-node matching — comparing two approaches:

- **Exact recursive algorithm**: enumerates all possible perfect matchings and selects the minimum-cost one via Dijkstra's shortest paths. Guaranteed optimal, but computationally expensive as the number of odd nodes grows (the number of matchings grows factorially).
- **Particle Swarm Optimization (PSO)**: a metaheuristic that searches for a near-optimal matching much faster, at the cost of an optimality guarantee.

The script generates a random graph, computes the exact optimum for benchmarking, runs the PSO solver, and plots the PSO convergence curve against the recursive optimum.

## Features

- Random connected weighted graph generator with a configurable target number of odd-degree nodes.
- Exact recursive solver for the minimum-cost odd-node matching (brute-force, guaranteed optimal).
- Dijkstra's algorithm implementation for shortest-path distances.
- PSO metaheuristic for the odd-node matching problem, with configurable:
  - Swarm size and number of iterations
  - Inertia (`w`) and acceleration coefficients (`c1`, `c2`)
  - Velocity/attraction update modes (`r0`, `r1`, `r2`: `'none'`, `'cell'`, `'matrix'`)
  - Additional exploration term (`extra_mode`: `'eta'`, `'uniform'`, `'random'`)
- Convergence plot (mean cost, best cost, and exact optimum reference line) saved as PNG.

## Requirements

- Python 3.8+
- [NumPy](https://numpy.org/)
- [NetworkX](https://networkx.org/)
- [Matplotlib](https://matplotlib.org/)

Install dependencies with:

```bash
pip install numpy networkx matplotlib
```

## Usage

Run the script directly:

```bash
python "PSO CPP Solver.py"
```

All configuration options are grouped at the top of the `main()` function:

```python
target_odd_nodes = 8   # 8, 10, 12 — size of the odd-node matching problem

w = 0.7                 # inertia weight
c1 = 1.5                 # personal-best (cognitive) coefficient
c2 = 1.5                 # global-best (social) coefficient

n_particles = 10        # swarm size
max_iter = 10            # number of PSO iterations
```

On completion, the script prints the exact optimal cost, the best cost found by PSO, and whether the PSO run matched the exact optimum. A convergence plot is saved in the working directory as:

```
convergence_<N>nodes_<nParticles>p_<MaxIt>it.png
```

## Example Output

```
Generating graph with 8 odd nodes...
Computing exact optimum (recursive algorithm)...
Global optimum: 1773
Best cost found by PSO: 1773.0
Success (PSO == optimum): True
```

## Project Structure

```
PSO_CPP/
├── PSO CPP Solver.py   # main script (graph generation, recursive solver, PSO, plotting)
├── LICENSE
└── README.md
```

## Credits

The recursive odd-node pairing algorithm (`generate_pairs`, `get_pairings`, `shortest_paths_sum`, `recursive_algorithm`) is adapted from:

> Araz Sharma, ["Chinese Postman in Python"](https://towardsdatascience.com/chinese-postman-in-python-45e9987e1b7d), Towards Data Science, Nov 9, 2020.

All credit for the original recursive pairing-generation approach goes to the author above; this implementation follows the same logic, adapted and integrated into the PSO benchmarking pipeline.

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

Note: the recursive pairing algorithm described above is adapted from third-party work; see the [Credits](#credits) section and the attribution notice in the source code.
