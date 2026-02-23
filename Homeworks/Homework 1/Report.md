# Homework 1 Report

## Exercise 1 — Floating-Point Arithmetic

### Problem Statement

Given pairs $(a, b)$ where $a = 0.1 \cdot b$, compute the quantity

$$a - \sum_{i=1}^{b} 0.1$$

which is mathematically equal to zero. The goal is to measure the absolute error introduced by floating-point arithmetic across different summation strategies and increasing problem sizes.

The three test cases are:

| $a$     | $b$       | Exact result |
|---------|-----------|--------------|
| 1 000   | 10 000    | 0            |
| 10 000  | 100 000   | 0            |
| 100 000 | 1 000 000 | 0            |

### Methods

Four approaches were implemented:

1. **Brute-force loop** (`equationMethod`) — accumulates 0.1 in a `while` loop, then subtracts from `a`. Each addition carries a rounding error of roughly $\epsilon_{\text{machine}}/2 \approx 1.1 \times 10^{-17}$, and these errors accumulate linearly with $b$.
2. **`sum` built-in** (`sumMethod`) — uses Python's `sum()` over a generator of 0.1 values. Python's `sum` uses pairwise compensated summation internally, which substantially reduces round-off.
3. **`math.fsum`** (`fsumMethod`) — uses an exact rational accumulator that produces a correctly-rounded result. This is the most accurate option available in the standard library.
4. **NumPy `np.sum`** (`numpyMethod`) — operates on a fixed array of 0.1 values in 64-bit floating-point. NumPy uses pairwise summation, giving logarithmic error growth ($O(\log b)$ instead of $O(b)$).

### Results

Absolute errors (|result − 0|):

| Method        | $(1000,\ 10000)$          | $(10000,\ 100000)$        | $(100000,\ 1000000)$      |
|---------------|---------------------------|---------------------------|---------------------------|
| Brute-force   | $1.588 \times 10^{-10}$   | $1.885 \times 10^{-8}$    | $1.333 \times 10^{-6}$    |
| `sum`         | $0$                       | $0$                       | $0$                       |
| `math.fsum`   | $0$                       | $0$                       | $0$                       |
| NumPy         | $2.274 \times 10^{-13}$   | $1.819 \times 10^{-12}$   | $2.037 \times 10^{-10}$   |

### Discussion

The results clearly show the impact of summation algorithm design on floating-point accuracy.

The **brute-force loop** accumulates error linearly: going from $b = 10^4$ to $b = 10^6$ (a 100× increase) increases the error by roughly $10^4\times$, confirming $O(b)$ growth. This is the expected behaviour of naive sequential summation.

**`sum` and `math.fsum`** both return exactly zero for all test cases. `math.fsum` achieves this by maintaining an exact partial sum internally. Python's `sum` benefits from compensated arithmetic that, in practice, cancels the error for this specific input pattern (all summands are identical).

**NumPy** shows errors that grow much more slowly than the brute-force approach, consistent with pairwise summation's $O(\log b)$ error bound. Errors are still non-zero because 0.1 is not representable exactly in binary floating-point, but they remain well below the brute-force errors.

**Conclusion:** `math.fsum` is the most numerically reliable choice for summation tasks where accuracy matters. NumPy's pairwise summation is an excellent practical compromise for large arrays. The brute-force loop should be avoided when summing many terms of similar magnitude.

---

## Exercise 2 — Linear Algebra

### Problem Statement

Given the matrices and vectors:

$$
A = \begin{pmatrix} 1 & 2 \\ -1 & 1 \end{pmatrix}, \quad
B = \begin{pmatrix} 2 & 0 \\ 0 & 2 \end{pmatrix}, \quad
C = \begin{pmatrix} 2 & 0 & -3 \\ 0 & 0 & -1 \end{pmatrix}, \quad
D = \begin{pmatrix} 1 & 2 \\ 2 & 3 \\ -1 & 0 \end{pmatrix}
$$

$$
\mathbf{x} = \begin{pmatrix} 1 \\ 0 \end{pmatrix}, \quad
\mathbf{y} = \begin{pmatrix} 0 \\ 1 \end{pmatrix}, \quad
\mathbf{z} = \begin{pmatrix} 1 \\ 2 \\ -1 \end{pmatrix}
$$

Compute the following expressions using NumPy.

### Results

| Expression       | Result |
|-----------------|--------|
| $A + B$         | $\begin{pmatrix} 3 & 2 \\ -1 & 3 \end{pmatrix}$ |
| $3\mathbf{x} - 4\mathbf{y}$ | $\begin{pmatrix} 3 \\ -4 \end{pmatrix}$ |
| $A\mathbf{x}$   | $\begin{pmatrix} 1 \\ -1 \end{pmatrix}$ |
| $B(\mathbf{x} - \mathbf{y})$ | $\begin{pmatrix} 2 \\ -2 \end{pmatrix}$ |
| $D\mathbf{x}$   | $\begin{pmatrix} 1 \\ 2 \\ -1 \end{pmatrix}$ |
| $D\mathbf{y} + \mathbf{z}$ | $\begin{pmatrix} 3 \\ 5 \\ -1 \end{pmatrix}$ |
| $AB$            | $\begin{pmatrix} 2 & 4 \\ -2 & 2 \end{pmatrix}$ |
| $BC$            | $\begin{pmatrix} 4 & 0 & -6 \\ 0 & 0 & -2 \end{pmatrix}$ |
| $CD$            | $\begin{pmatrix} 5 & 4 \\ 1 & 0 \end{pmatrix}$ |

### Discussion

All operations were computed with `numpy.matmul` (or equivalent `np.array` arithmetic for element-wise operations). NumPy correctly enforces dimensionality: expressions such as $CB$ or $DC$ would raise a shape mismatch error, which is consistent with the rules of matrix multiplication.

Note that $B = 2I$ (a scalar multiple of the identity), so $B(\mathbf{x} - \mathbf{y}) = 2(\mathbf{x} - \mathbf{y})$, and $BC = 2C$, both of which are confirmed by the results above.

---

## Exercise 3 — The Logistic Map

### Problem Statement

The logistic map is the recurrence relation:

$$x_{n+1} = \rho \, x_n (1 - x_n)$$

Starting from $x_0 = 0.5$, compute 50 iterates ($x_0$ through $x_{49}$) for each of the following values of the parameter $\rho$:

$$\rho \in \{0.8,\ 1.5,\ 2.8,\ 3.2,\ 3.5,\ 3.65\}$$

### Results

The value reached after 50 iterations ($x_{49}$) and the qualitative long-term behaviour for each $\rho$:

| $\rho$ | $x_{49}$ | Behaviour |
|--------|----------|-----------|
| 0.8    | $2.12 \times 10^{-6}$ | Convergence to 0 |
| 1.5    | $0.3333\ldots$ | Convergence to fixed point $1 - 1/\rho = 1/3$ |
| 2.8    | $0.6429$ | Convergence to fixed point $1 - 1/\rho \approx 0.6429$ |
| 3.2    | $0.7995$ | Period-2 cycle (oscillates between $\approx 0.513$ and $\approx 0.799$) |
| 3.5    | $0.8750$ | Period-4 cycle |
| 3.65   | $0.7597$ | Chaotic (aperiodic) |

### Discussion

The logistic map is a classic example of how a simple one-dimensional nonlinear recurrence can produce a rich spectrum of dynamical behaviour as a single parameter is varied.

**$\rho = 0.8$:** When $\rho < 1$ the only stable fixed point is $x^* = 0$. All orbits starting in $(0, 1)$ converge exponentially to zero, as confirmed by $x_{49} \approx 2 \times 10^{-6}$.

**$\rho = 1.5$ and $\rho = 2.8$:** For $1 < \rho \leq 3$ there is a stable non-trivial fixed point at $x^* = 1 - 1/\rho$. The sequence converges monotonically (for $\rho \leq 2$) or with damped oscillations (for $2 < \rho \leq 3$) to this value.

- $\rho = 1.5$: $x^* = 1/3 \approx 0.3333$. The simulation matches to 10 decimal places after only 50 steps.
- $\rho = 2.8$: $x^* \approx 0.6429$. Convergence is slower (oscillatory) but still clearly reached by $n = 49$.

**$\rho = 3.2$:** A period-doubling bifurcation occurs at $\rho = 3$. Beyond this point the fixed point becomes unstable and the orbit settles into a **period-2 cycle**. The value $x_{49} \approx 0.7995$ is one of the two cycle values (the other is $\approx 0.5130$), and the index 49 lands on the larger one.

**$\rho = 3.5$:** A second period-doubling bifurcation near $\rho \approx 3.449$ produces a **period-4 cycle**. The four cycle values are approximately $\{0.3828, 0.8269, 0.5009, 0.8750\}$, and $x_{49} \approx 0.8750$ is one of them.

**$\rho = 3.65$:** The system has entered the **chaotic regime**. There is no stable cycle; instead, trajectories are aperiodic and sensitively dependent on initial conditions. The value $x_{49} \approx 0.7597$ is essentially unpredictable without running the full iteration — a signature of deterministic chaos.

**Summary:** As $\rho$ increases from 0 to 4 the logistic map undergoes a period-doubling cascade (fixed point → period-2 → period-4 → … → chaos). The six chosen values of $\rho$ sample four qualitatively distinct regimes — extinction, stable equilibrium, periodic cycling, and chaos — making them a concise illustration of this celebrated bifurcation structure.
