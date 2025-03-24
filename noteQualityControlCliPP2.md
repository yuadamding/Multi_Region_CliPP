# Piecewise-Constant Post-Processing of ADMM Output (Multidimensional Case)

This document generalizes the **piecewise-constant post-processing** method to the case where each mutation has a **vector** of raw ADMM solutions. Instead of one scalar value per mutation, we may have multiple *regions* or *dimensions* per mutation.

---

## 1. Notation

- **N**: Total number of mutations (indexed by i = 1, 2, …, N).
- **M**: Total number of regions (or dimensions).
- **phi_hat_i**: The *vector* of raw ADMM solutions for mutation i, i.e.  
  phi_hat_i = (phi_hat_i1, phi_hat_i2, …, phi_hat_iM).  
  Each phi_hat_i is thus an M-dimensional vector.

- **n_i**: A weight associated with mutation i (e.g., total coverage across all regions, or some other quantity).
- **post_th**: Threshold on the (vector) distance below which two mutations are deemed “no difference.”
- **least_mut**: Minimum allowed size for any group of mutations.
- **least_diff**: Minimum allowed gap (in the chosen vector norm) between the mean vectors of any two groups.

The *goal* is to partition the set {1, …, N} (the mutations) into groups G1, G2, … so that:

1. If the vector difference between two mutations i and j is below **post_th** (in some norm), they end up in the same group.  
2. Each group has at least **least_mut** members.  
3. Any two distinct group mean vectors are separated by at least **least_diff** in that same norm (unless there is only one group).

At the end, we produce a final piecewise-constant solution **phi_res** = {phi_res_i} (i = 1 … N), where each phi_res_i is an M-dimensional vector (one for each region).

---

## 2. Choosing a Distance (or Norm)

To handle multidimensional data, we must pick a norm or distance metric **‖·‖** in ℝ^M. A common choice is the Euclidean (L2) norm:

‖ x − y ‖₂ = sqrt( Σ (x_j − y_j)² ),

but you may also use L1, L∞, or any suitable metric. For clarity below, assume we use L2:

Δ(i, j) = ‖ phi_hat_i − phi_hat_j ‖₂.

---

## 3. Thresholding Small Pairwise Differences

1. **Construct the difference matrix**: For each pair (i, j), compute:

   Δ(i, j) = ‖ phi_hat_i − phi_hat_j ‖  (e.g., Euclidean distance).

2. **Apply threshold post_th**:  
   Define a modified distance Δ'(i, j):

   - If Δ(i, j) ≤ post_th, set Δ'(i, j) = 0.  
   - Otherwise, Δ'(i, j) = Δ(i, j).  

A value of 0 in Δ'(i, j) means “mutation i and j are effectively the same” for clustering.

---

## 4. Form Initial Groups

Treat “Δ'(i, j) = 0” as an edge in a graph that links mutations i and j. Finding the connected components of this graph yields initial groups. One incremental approach:

1. Set class_label(1) = 0; for each i > 1, set class_label(i) = −1.  
2. For i = 2 to N:
   - If there exists j < i such that Δ'(j, i) = 0 and class_label(j) = c, assign class_label(i) = c.
   - Otherwise, create a new group label.

Result: a preliminary partition G0, G1, …, GK−1 (each Gi is a set of mutation indices).

---

## 5. Refine Small Groups

If any group Gs has size < least_mut:

- For each mutation i in Gs:
  - Compute the distance from i to every other group Gc (c ≠ s). This can be done by  
    min { Δ'(i, j) | j in Gc }.  
  - Reassign i to whichever group yields the smallest distance.  

Repeat until no group is smaller than least_mut.

---

## 6. Compute Group Mean Vectors

Suppose the refined groups are G1, G2, …, GL. Each group Gℓ has members {i1, i2, …}. We define the group’s mean vector phi_out,ℓ (in ℝ^M) by a weighted average of the raw ADMM vectors:

phi_out,ℓ =  
   ( Σ over i in Gℓ of [ n_i * phi_hat_i ] )  
   /  
   ( Σ over i in Gℓ of n_i ).

Concretely, for each region component m = 1, 2, …, M:

phi_out,ℓ(m) =  
   ( Σ over i in Gℓ of [ n_i * phi_hat_i(m) ] )  
   /  
   ( Σ over i in Gℓ of n_i ).

---

## 7. Merge Groups with Insufficient Separation

We now check if any pair of group mean vectors ( phi_out,p, phi_out,q ) is too close:

1. Sort the group means in any convenient manner (sorting by L2 norm is one option, or simply check all pairs).  
2. For each adjacent pair (or for every pair p, q), compute  
   ‖ phi_out,p − phi_out,q ‖₂  
   (or the chosen norm).  
3. If the smallest such distance is < least_diff, merge those two groups into a single group and recalculate its mean (as in Step 6).  
4. Repeat until all group-pair distances ≥ least_diff or only one group remains.

---

## 8. Final Piecewise-Constant Assignment

After merging, assume we have a final set of groups G1, …, G⎿L⏌, each with a mean vector phi_out,ℓ. For each mutation i in group Gℓ, set:

phi_res_i = phi_out,ℓ.

Hence, the final **phi_res** is an N×M matrix (or an array of N vectors in ℝ^M) where:

- Mutations in the same group share the same M-dimensional mean.
- No group is smaller than least_mut.
- Group means differ by at least least_diff in the chosen norm.

---

## 9. Summary of the Multidimensional Extension

1. **Distance Metric**: Decide on an appropriate norm (often L2) for comparing the M-dimensional ADMM solutions.  
2. **Threshold**: Pairwise differences below post_th → force these mutations into the same group.  
3. **Minimum Group Size**: If a group is too small (< least_mut), redistribute those mutations to the closest larger group.  
4. **Compute Group Means**: Each group’s mean is a vector in ℝ^M, computed by weighted averaging of phi_hat_i.  
5. **Enforce Separation**: If two group mean vectors are closer than least_diff, merge them.  
6. **Final Output**: A piecewise-constant vector solution phi_res ∈ (ℝ^M)^N (or N×M), where each mutation’s M-dimensional value is the mean of its final group.

Through these steps, you obtain a *piecewise-constant solution* that is more interpretable and meets practical constraints:

- **Post-Thresholding** ensures that mutations with nearly identical M-dimensional ADMM estimates collapse into the same group.  
- **Minimum Group Size** eliminates spurious tiny clusters.  
- **Minimum Mean Separation** ensures distinctly labeled groups remain sufficiently different in all M regions.
