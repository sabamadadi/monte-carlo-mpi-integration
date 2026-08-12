# Distributed Monte Carlo Approximation of ODE Definite Integrals via MPI 🚀

A distributed implementation of Monte Carlo integration using C and MPI based on the **Master-Worker architecture** to evaluate definite integrals of ordinary differential equations (ODEs) with numerical precision and parallel profiling.

---

## 📌 Abstract

This project evaluates the distributed approximation of the definite integral representing the solution to the initial value problem:
$$u'(x) = \sin(x), \quad u(0) = 0 \implies u(x) = \int_{0}^{x} \sin(t) \, dt = 1 - \cos(x)$$

Using a Monte Carlo stochastic sampling approach, the computational workload is partitioned across multiple nodes in a distributed-memory system via MPI. Process 0 (Master) coordinates workload distribution and aggregates partial results, while processes $1 \dots P-1$ (Workers) perform independent sampling. Numerical precision is bolstered using **Kahan compensated summation** to minimize floating-point truncation errors. Empirical benchmarks validate the theoretical stochastic error convergence rate of $\mathcal{O}(N^{-1/2})$ and demonstrate effective strong scaling for large sample sizes ($N = 10^7$).

---

## 🧮 Method & Parallel Design

The Monte Carlo estimator transforms the integral over $R \sim \mathcal{U}(0, 1)$ into an expectation:
$$\hat{u}(x) = x \cdot \frac{1}{N} \sum_{i=1}^{N} \sin(x \cdot R_i)$$

### 🛠️ Key Implementation Highlights
* **Master-Worker Model 🤖:** Rank 0 evenly distributes $N$ samples across $P-1$ worker processes, properly handling remainder allocation $r = N \pmod{P-1}$.
* **Independent RNG 🎲:** Uses **SplitMix64** seeded uniquely per process rank to avoid correlated pseudo-random streams.
* **Kahan Summation ⚖️:** Prevents catastrophic cancellation during large floating-point additions ($N \ge 10^7$).
* **Sequential Fallback ⚡:** Runs an optimized single-process execution path when $P=1$ to evaluate true baseline speedup.

---

## 📂 Repository Structure

```text
.
├── src/
│   └── mc_integral_mpi.c       # C source code with MPI routines
├── scripts/
│   ├── run_experiments.sh      # Automated test execution script
│   └── plot_results.py         # Matplotlib visualization script
├── figures/                    # Benchmark plots displayed in README
│   ├── accuracy.png
│   ├── runtime.png
│   ├── speedup.png
│   └── efficiency.png
├── Makefile                    # Compilation build automation
└── README.md

```

---

## 🚀 Quick Start

### Prerequisites

* C Compiler (`gcc` or `clang`) with C11 support
* MPI implementation (OpenMPI or MPICH)

### Build & Run

```bash
# Compile
make

# Execute across 4 processes for x = pi/2 and N = 10,000,000
mpiexec -n 4 ./mc_integral_mpi 1.5707963267948966 10000000 12345

```

---

## 📊 Experimental Results & Performance Plots

All benchmarks were evaluated for $u(\pi/2) = 1.0$ across process counts $P \in \{1, 2, 4, 8\}$ and sample sizes $N \in \{10^4, 10^5, 10^6, 10^7\}$.

---

### 1. Numerical Accuracy Convergence 🎯

* **Analysis:** The integration error follows the theoretical Monte Carlo convergence bound of $\mathcal{O}(N^{-1/2})$. Increasing the total sample size $N$ from $10^4$ to $10^7$ reduces the absolute integration error by approximately two orders of magnitude (from $\sim 10^{-3}$ down to $\sim 3 \times 10^{-5}$).
* **Stochastic Behavior:** Minor fluctuations between different process counts ($P$) arise due to distinct, independent random number generation sequences per rank, rather than algorithmic variance.

---

### 2. Execution Time & Runtime Scaling ⏱️

* **Analysis:** For computationally intensive workloads ($N = 10^7$), parallel execution provides substantial time savings—dropping total runtime from **0.445s** on a single process ($P=1$) down to **0.049s** across 8 processes ($P=8$).
* **Low Workload Threshold:** For small sample sizes ($N \le 10^5$), the execution time flattens out rapidly because the overhead of initializing MPI and sending messages outweighs the compute time.

---

### 3. Speedup Analysis $S(N,P)$ 📈

* **Analysis:** Speedup performance improves significantly as the workload per worker increases. At $N = 10^7$, the implementation achieves strong scalability ($S \approx 9.0$ at $P=8$).
* **Cache Locality Influence:** The observed superlinear scaling behavior at high process counts ($P=8$) stems from reduced working set sizes per rank, allowing local arrays and RNG states to fit entirely within the fast L1/L2 data caches.

---

### 4. Parallel Efficiency $E(N,P)$ ⚡

* **Analysis:** Efficiency measures core resource utilization ($E = S/P$). For large sample sizes ($N = 10^7$), efficiency remains near or above **100%**, proving high hardware capability utilization.
* **Communication Latency Impact:** For small sample counts ($N = 10^4$), efficiency drops drastically to below **0.10** at $P=8$ due to communication overhead and Master-Worker synchronization latency dominating the minimal worker compute load.

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for details.
