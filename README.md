# FCVI

This repository contains the code for the numerical experiments of our paper titled:

**"First-order Methods for Stochastic Variational Inequality Problems with Function Constraints (FCVI)"**

The numerical section is divided into three main parts:

## 1. Quadratic-Constrained Quadratic Saddle-Point Problem

We first consider quadratic saddle-point problems where the constraints may be either **coupling** or **disjoint**. Part 1 contains the problem setup and corresponding implementation. Please refer to the main file and functions in the folder **`part 1/`**.

### 1.1 QCQS with Coupling Deterministic Constraint

We use a quadratic coupling constraint of the form:

$$
\|x + y\|^2 \leq D_{x,y}
$$

This setting is referred to as **`set1`** in the main file under `part 1`.

### 1.2 QCQS with Disjoint Stochastic Constraint

We use a quadratic stochastic constraint of the form:

$$
\mathbb{E}[(x^\top s_j + \xi_j)^2] \leq \theta_j \quad \forall j \in [m]
$$

This setting is referred to as **`set2`** in the main file under `part 1`.

---

## 2. Robust Optimal Pricing

We implement our **OpConEx** method on a robust optimal pricing problem and compare its performance with a state-of-the-art algorithm, **RLSA**. For implementation details, see the files in the folder **`part 2`**.

---

## 3. AUC Maximization with Fairness Constraints

We reformulate the AUC maximization problem with fairness constraints as an FCVI, and implement the **OpConEx** method using the [Adult Income dataset](https://archive.ics.uci.edu/dataset/2/adult).

The dataset and implementation files are located in the folder **`part 3`**.



