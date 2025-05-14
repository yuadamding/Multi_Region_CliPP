Below is a step-by-step mathematical description of the core ideas in the provided code. The code essentially performs a **clustering** of mutations (rows indexed by $i=1,\dots,N$) and then **merges** or **refines** clusters based on (1) minimal cluster size constraints and (2) proximity of cluster means in the space of $\phi$-vectors.

---

## 1. Notation and Setup

1. **Number of mutations**: $N = \text{No\_mutation}$.

2. **Observed (or estimated) matrices**:

   * $w_{\text{new}} \in \mathbb{R}^d$ (dimension depends on the model),
   * $\eta_{\text{new}} \in \mathbb{R}^N$ (or $\mathbb{R}^{N \times k}$, depending on your code),
   * $\phi_{\text{hat}} \in \mathbb{R}^{N \times M}$.
     Here:
   * Each row $\phi_{\text{hat}}[i,:]$ corresponds to mutation $i$'s $\phi$-vector across $M$ “regions” (or samples).

3. **Coverage**: $n \in \mathbb{R}^{N \times M}$. Element $n[i,j]$ is coverage for mutation $i$ in region $j$.

4. **Hyperparameters**:

   * $\text{post\_th}$ — threshold below which elements of $\eta_{\text{new}}$ are set to 0,
   * $\text{least\_mut} = \lceil 0.05 \times N\rceil$ — minimum cluster size (5% rule in the code),
   * $\text{least\_diff}$ — distance threshold below which two cluster centers are merged,
   * $\text{purity}$ — used in the final reassign step (details inside `reassign_labels_by_distance`).

5. **Distance matrix**:

   $$
     \text{diff} \;=\; \text{diff\_mat}(w_{\text{new}}) \;\in \mathbb{R}^{N \times N}.
   $$

   This is a pairwise difference or distance matrix among the $N$ mutations (the exact definition depends on how `diff_mat` is implemented).

6. **Cluster labels**: $\text{class\_label} \in \{-1, 0, 1, 2, \dots\}^{N}$.

   * Initially $\text{class\_label}[0] = 0$ (the first mutation starts cluster 0) and $\text{class\_label}[i] = -1$ for $i>0$.
   * $\text{group\_size}[k]$ holds the size of cluster $k$.

---

## 2. Preprocessing Steps

**(a) Thresholding $\eta_{\text{new}}$:**
For each component of $\eta_{\text{new}}$,

$$
  \eta_{\text{new}}[i] \;\leftarrow\; 
  \begin{cases}
    0, & \text{if } |\eta_{\text{new}}[i]| \le \text{post\_th},\\
    \eta_{\text{new}}[i], & \text{otherwise}.
  \end{cases}
$$

**(b) Update upper-triangular part of $\text{diff}$ using $\eta_{\text{new}}$:**
In code, `diff[ids] = np.linalg.norm(eta_new, axis=1)`, meaning for each upper-triangular pair $(j,i)$, the distance $\text{diff}[j,i]$ is set to $\|\eta_{\text{new}}[\dots]\|$. (The exact indexing logic depends on how `ids` was constructed; the idea is that each pair gets a new distance from $\eta_{\text{new}}$.)

---

## 3. Initial Clustering

We iterate $i=1,\dots,N-1$ to assign each mutation $i$ to a cluster:

1. For each $j\in \{0,\dots,i-1\}$:

   * If $\text{diff}[j,i] = 0$, then **merge** $i$ into the cluster of $j$:

     $$
       \text{class\_label}[i] = \text{class\_label}[j], 
       \quad
       \text{group\_size}\bigl(\text{class\_label}[j]\bigr) \;+=\; 1.
     $$

     Break the loop.

2. If after scanning all $j$'s you still have $\text{class\_label}[i] = -1$, this means no distance was 0. Then:

   $$
     \text{class\_label}[i] = \text{labl}, \quad \text{labl} \;+=\; 1,
     \quad \text{group\_size}(\text{labl}) \leftarrow 1.
   $$

   In other words, mutation $i$ starts a **new** cluster.

---

## 4. Quality Control and Refinement

We impose a minimum cluster size $\text{least\_mut} = \lceil 0.05\,N\rceil$.

1. Let

   $$
     \text{tmp\_size} \;=\; \min_{k : \text{group\_size}[k] > 0} \text{group\_size}[k].
   $$

   If $\text{tmp\_size} < \text{least\_mut}$, set $\text{refine} = \text{True}$; otherwise $\text{False}$.

2. **While** $\text{refine} = \text{True}$:

   * Find the cluster $g$ with size $\text{tmp\_size}$. Let $\text{tmp\_col} = \{i : \text{class\_label}[i] = g\}$.
   * For each $i\in \text{tmp\_col}$:

     1. Compute a “distance vector” $\text{tmp\_diff}$ from $i$ to all other mutations except set artificially large distances ($\gg 1$) for members of the same cluster $\text{tmp\_col}$.
     2. Let $ \ell = \arg\min(\text{tmp\_diff})$.
     3. Reassign $i$ to the cluster of $\ell$:

        $$
          \text{group\_size}\bigl(\text{class\_label}[i]\bigr) \;-\!=\; 1, 
          \quad
          \text{class\_label}[i] = \text{class\_label}[\ell],
          \quad
          \text{group\_size}\bigl(\text{class\_label}[i]\bigr) \;+\!=\; 1.
        $$
   * Recompute $\text{tmp\_size}$. If still $\text{tmp\_size} < \text{least\_mut}$, repeat.

This loop tries to **break up** tiny clusters by reassigning their points to the **nearest** cluster (nearest in $\text{diff}$-distance).

---

## 5. Cluster-Level $\phi$-Vectors

After refinement, let $\text{labels} = \{\text{unique cluster labels}\}$. Suppose there are $K = |\text{labels}|$ clusters.

Define the **cluster-level mean** of $\phi_{\text{hat}}$ (weighted by $n$):

$$
  \phi_{\text{out}}[k, :] 
  \;=\;
  \frac{\sum_{i : \text{class\_label}[i] = k} 
          \bigl(\phi_{\text{hat}}[i,:] \cdot n[i,:]\bigr)}
       {\sum_{i : \text{class\_label}[i] = k} n[i,:]}
  \quad
  \text{for }k = 1,\dots,K.
$$

This gives one $\phi$-vector per cluster. (All operations above are elementwise in the region dimension.)

---

## 6. Merging Clusters with Similar $\phi$-Vectors

1. If $K>1$, compute the Euclidean norm of each cluster’s mean $\phi_{\text{out}}[k,:]$. Sort them in ascending order of $\|\phi_{\text{out}}[k,:]\|$. Denote the sorted list by $\text{sort\_phi}$.

2. For each adjacent pair $\text{sort\_phi}[k+1] - \text{sort\_phi}[k]$, compute the Euclidean distance:

   $$
     \|\text{sort\_phi}[k+1, :] - \text{sort\_phi}[k, :]\|_2.
   $$

   Let $\text{min\_ind}$ be the index of the pair with **smallest** distance $\text{min\_val}$.

3. If $\|\text{min\_val}\| < \text{least\_diff}$, we **merge** those two clusters into one cluster, recompute $\phi_{\text{out}}$, and repeat until no pair of cluster centers is closer than $\text{least\_diff}$.

---

## 7. Final $\phi$-Assignments and Re-Labeling

1. After merging, define a final matrix $\phi_{\text{res}} \in \mathbb{R}^{N \times M}$ by assigning each mutation $i$ its cluster’s mean:

   $$
     \phi_{\text{res}}[i,:] \;=\; \phi_{\text{out}}\bigl[\text{class\_label}[i],:\bigr].
   $$

2. Lastly, a function `reassign_labels_by_distance(\phi_{\text{res}}, \text{class\_label}, \text{purity})` is called, which may do a final pass of label adjustments (the exact math depends on that function’s logic).

---

## 8. Summary of the Mathematical Logic

1. **Build** an initial distance matrix $\text{diff}$ from $w_{\text{new}}$ and $\eta_{\text{new}}$.
2. **Initialize** clusters by grouping any pairs $(j,i)$ for which $\text{diff}[j,i] = 0$.
3. **Refine** clusters if some are too small, redistributing their points to the nearest larger cluster.
4. **Compute** weighted means of $\phi_{\text{hat}}$ per cluster to get $\phi_{\text{out}}$.
5. **Merge** clusters whose $\phi_{\text{out}}$-means are closer than $\text{least\_diff}$ in Euclidean norm.
6. **Assign** each mutation the final cluster mean $\phi_{\text{res}}[i,:]$.
7. **Optionally** do a final “re-label by distance” step (`reassign_labels_by_distance`).

This complete set of steps is effectively a specialized **distance-based clustering** plus subsequent **centroid merging** in the $\phi$-space.
