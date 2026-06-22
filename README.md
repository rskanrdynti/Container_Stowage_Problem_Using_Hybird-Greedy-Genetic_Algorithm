# Container Stowage Optimization using Genetic Algorithm and Greedy Algorithm

## Project Overview
This repository contains implementations of container stowage optimization for vessel loading using multiple optimization approaches. The objective is to determine an efficient arrangement of containers within ship slots while satisfying operational constraints and improving loading quality.

The project compares different methods:

- Genetic Algorithm (GA)
- Greedy Algorithm
- Main integrated implementation

The optimization considers vessel geometry configuration and allocation strategies for container placement.

---

## Repository Structure

```text
├── GA aja.py
├── Greedy aja.py
├── Source_Code_KP_clean.py
├── README.md
```

### File Description

#### `GA aja.py`
Implementation of the container stowage optimization using Genetic Algorithm.

Features:
- Population initialization
- Fitness evaluation
- Selection mechanism
- Crossover operation
- Mutation operation
- Parallel processing support
- Optimization result generation

---

#### `Greedy aja.py`
Implementation of a Greedy-based container allocation approach.

Features:
- Sequential decision making
- Fast execution
- Lower computational cost
- Baseline comparison method

---

#### `Source_Code_KP_clean.py`
Main cleaned and integrated source code for the project.

Features:
- Combined optimization workflow
- Multiprocessing support
- Improved structure and readability
- Performance monitoring

---

## Methodology

### Genetic Algorithm Workflow

1. Generate initial population
2. Evaluate fitness value
3. Select parents
4. Perform crossover
5. Perform mutation
6. Generate new population
7. Repeat until stopping criteria is reached

### Greedy Algorithm Workflow

1. Sort available container positions
2. Select the best immediate placement
3. Allocate container
4. Repeat until all containers are assigned

---

## Requirements

Install required libraries before running the code:

```bash
pip install pandas numpy
```

Additional modules used:

```python
random
os
time
multiprocessing
datetime
sys
```

---

## How to Run

Run Genetic Algorithm:

```bash
python "GA aja.py"
```

Run Greedy Algorithm:

```bash
python "Greedy aja.py"
```

Run Main Program:

```bash
python "Source_Code_KP_clean.py"
```

---

## Expected Output

The program generates:

- Optimized container allocation results
- Performance comparison
- Execution time statistics
- Optimization metrics

---

## Project Goal

The main goals of this project are:

- Optimize container placement on vessels
- Compare optimization performance between algorithms
- Improve loading efficiency
- Reduce imbalance and operational constraints

---

## Author

**Name:** Ariska Nurdyanti
**Student ID:** 5002231137

---

## License

This project is developed for academic and research purposes.
