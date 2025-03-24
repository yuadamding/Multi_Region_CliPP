```markdown
# Mathematical Formulation of the Post-Processing Code

Below is a **line-by-line** breakdown of the post-processing code that refines the raw ADMM solution \(\phi_{\hat i}\) into a **piecewise-constant** result \(\phi_{\text{res}}\). This step enforces:

1. A **threshold** on small pairwise differences, collapsing nearly identical mutations.
2. A **minimum group size**, eliminating tiny groups by reassigning their members.
3. A **minimum gap** between group means, merging any groups that are too close.

---

## Code Reference (For Context)

```python
# assign mutations based on the distance matrix
eta_new[np.where(np.abs(eta_new) <= post_th)] = 0
diff[ids] = eta_new
class_label = -np.ones(No_mutation)
class_label[0] = 0
group_size = [1]
labl = 1

for i in range(1, No_mutation):
    for j in range(i):
        if diff[j, i] == 0:
            class_label[i] = class_label[j]
            group_size[int(class_label[j])] += 1
            break
    if class_label[i] == -1:
        class_label[i] = labl
        labl += 1
        group_size.append(1)


# quality control
tmp_size = np.min(np.array(group_size)[np.array(group_size) > 0])
tmp_grp = np.where(group_size == tmp_size)
refine = False
if tmp_size < least_mut:
    refine = True
while refine:
    refine = False
    tmp_col = np.where(class_label == tmp_grp[0][0])[0]
    for i in range(len(tmp_col)):
        if tmp_col[i] != 0 and tmp_col[i] != No_mutation - 1:
            tmp_diff = np.abs(np.append(np.append(diff[0:tmp_col[i], tmp_col[i]].T.ravel(), 100),
                                        diff[tmp_col[i], (tmp_col[i] + 1):No_mutation].ravel()))
            tmp_diff[tmp_col] += 100
            diff[0:tmp_col[i], tmp_col[i]] = tmp_diff[0:tmp_col[i]]
            diff[tmp_col[i], (tmp_col[i] + 1):No_mutation] = tmp_diff[(tmp_col[i] + 1):No_mutation]
        elif tmp_col[i] == 0:
            tmp_diff = np.append(100, diff[0, 1:No_mutation])
            tmp_diff[tmp_col] += 100
            diff[0, 1:No_mutation] = tmp_diff[1:No_mutation]
        else:
            tmp_diff = np.append(diff[0:(No_mutation - 1), No_mutation - 1], 100)
            tmp_diff[tmp_col] += 100
            diff[0:(No_mutation - 1), No_mutation - 1] = tmp_diff[0:(No_mutation - 1)]
        ind = tmp_diff.argmin()
        group_size[class_label.astype(np.int64, copy=False)[tmp_col[i]]] -= 1
        class_label[tmp_col[i]] = class_label[ind]
        group_size[class_label.astype(np.int64, copy=False)[tmp_col[i]]] += 1
    tmp_size = np.min(np.array(group_size)[np.array(group_size) > 0])
    tmp_grp = np.where(group_size == tmp_size)
    refine = False
    if tmp_size < least_mut:
        refine = True

labels = np.unique(class_label)
phi_out = np.zeros(len(labels))
for i in range(len(labels)):
    ind = np.where(class_label == labels[i])[0]
    class_label[ind] = i
    phi_out[i] = np.sum(phi_hat[ind] * n[ind]) / np.sum(n[ind])

if len(labels) > 1:
    sort_phi = np.sort(phi_out)
    phi_diff = sort_phi[1:] - sort_phi[:-1]
    min_val = phi_diff.min()
    min_ind = phi_diff.argmin()
    while min_val < least_diff:
        combine_ind = np.where(phi_out == sort_phi[min_ind])[0]
        combine_to_ind = np.where(phi_out == sort_phi[min_ind + 1])[0]
        class_label[class_label == combine_ind] = combine_to_ind
        labels = np.unique(class_label)
        phi_out = np.zeros(len(labels))
        for i in range(len(labels)):
            ind = np.where(class_label == labels[i])[0]
            class_label[ind] = i
            phi_out[i] = np.sum(phi_hat[ind] * n[ind]) / np.sum(n[ind])
        if len(labels) == 1:
            break
        else:
            sort_phi = np.sort(phi_out)
            phi_diff = sort_phi[1:] - sort_phi[:-1]
            min_val = phi_diff.min()
            min_ind = phi_diff.argmin()

phi_res = np.zeros(No_mutation)
for lab in range(len(phi_out)):
    phi_res[class_label == lab] = phi_out[lab]
```

Below, we dissect this code **in six steps**, connecting each segment to its mathematical formulation.

---

## 1. Threshold Small Pairwise Differences

```python
eta_new[np.where(np.abs(eta_new) <= post_th)] = 0
diff[ids] = eta_new
```

### Explanation

- **`eta_new`** holds the pairwise logit-differences \(w_i - w_j\) from the ADMM step.  
- Any difference with absolute value \(\le \text{post_th}\) is set to 0, meaning “treat these two mutations as effectively the same.”  
- **`diff[ids] = eta_new`** updates the upper-triangular portion of the full `diff` matrix with the thresholded values.

### Math Formulation

1. Let \(\Delta_{i,j} = | \phi_{\hat i} - \phi_{\hat j}|\) (or an equivalent difference in logit space).
2. Define:
   \[
   \Delta'_{i,j} = 
       \begin{cases}
         0, & \text{if } \Delta_{i,j} \le \text{post\_th},\\
         \Delta_{i,j}, & \text{otherwise}.
       \end{cases}
   \]
3. Store these zeroed differences into the full matrix `diff`.

---

## 2. Form Initial Groups

```python
class_label = -np.ones(No_mutation)
class_label[0] = 0
group_size = [1]
labl = 1

for i in range(1, No_mutation):
    for j in range(i):
        if diff[j, i] == 0:
            class_label[i] = class_label[j]
            group_size[int(class_label[j])] += 1
            break
    if class_label[i] == -1:
        class_label[i] = labl
        labl += 1
        group_size.append(1)
```

### Explanation

- We store the group **label** of each mutation in `class_label`. Initially, everything is \(-1\) (unassigned) except the first mutation (label 0).  
- We iterate from `i=1` to `No_mutation - 1`. If `diff[j,i] = 0` for some `j < i`, then `i` joins the same group as `j`. Otherwise, we create a new group label.  
- We track **`group_size`** in parallel.

### Math Formulation

- Let `class_label(i)` = \(-1\) mean mutation \(i\) is unassigned.  
- For each new mutation \(i\), if there exists a \(j < i\) with \(\Delta'_{j,i} = 0\), assign `class_label(i)` = `class_label(j)`.  
- If no such \(j\) exists, define a **new** label `labl`, increment it, and set `class_label(i) = labl`.  
- Hence, we form an **initial clustering** of the mutations into groups \(\{G_0, G_1, \dots\}\).

---

## 3. Quality Control (Refine Small Groups)

### Identify the Smallest Group

```python
tmp_size = np.min(np.array(group_size)[np.array(group_size) > 0])
tmp_grp = np.where(group_size == tmp_size)
refine = False
if tmp_size < least_mut:
    refine = True
```

- We look at the **nonempty** groups and find the group size \(\text{tmp\_size}\) that is minimal.  
- If that size is below **least_mut**, we set `refine = True` and enter the loop.

### Redistribute the Smallest Group’s Members

```python
while refine:
    refine = False
    tmp_col = np.where(class_label == tmp_grp[0][0])[0]
    for i in range(len(tmp_col)):
        # compute tmp_diff to each other group
        # pick the group with the smallest difference
        # reassign
    ...
    tmp_size = ...
    tmp_grp = ...
    refine = (tmp_size < least_mut)
```

- We find the group index `tmp_grp[0][0]` that has the smallest size.  
- `tmp_col` are the **indices** of the mutations in this smallest group.  
- For each such mutation, we compute a distance vector `tmp_diff` to other possible groups, artificially large near the same group.  
- We pick the **argmin** from `tmp_diff` → reassign the mutation to that group.  
- Update group sizes accordingly.  
- Re-check the smallest group size; if still below **least_mut**, repeat.

#### Math Formulation

1. Let \(s\) be the index of the smallest group:
   \[
     s = \arg\min_{k} \lvert G_k \rvert,
   \]
   and \(\lvert G_s\rvert < \text{least\_mut}\).
2. For each \(i\in G_s\), define a distance measure to each group \(k \neq s\). In the code, it looks like:
   \[
     \delta(i,k) = \min_{j \in G_k} \bigl| \phi_{\hat i} - \phi_{\hat j}\bigr|.
   \]
   (The code uses `diff`, plus the “+100” trick to exclude the same group or self.)
3. Assign:
   \[
     \text{class\_label}(i) \leftarrow \arg\min_{k\neq s}\;\delta(i,k).
   \]
4. Update group sizes. Repeat until no group is below **least\_mut**.

---

## 4. Compute Group Means

```python
labels = np.unique(class_label)
phi_out = np.zeros(len(labels))
for i in range(len(labels)):
    ind = np.where(class_label == labels[i])[0]
    class_label[ind] = i
    phi_out[i] = np.sum(phi_hat[ind] * n[ind]) / np.sum(n[ind])
```

### Explanation

- Once the group assignments stabilize (none below `least_mut`), gather each group’s **indices** (`ind`).  
- Compute a **weighted mean** \(\phi_{\text{out}, i}\) for group \(i\), using \(\phi_{\hat{}}\) and the weights `n`.

### Math Formulation

If \(G_\ell\) is the set of mutations in group \(\ell\), then:
\[
\phi_{\text{out},\ell}
\;=\;
\frac{\sum_{i \in G_\ell} \bigl(n_i \,\phi_{\hat i}\bigr)}{\sum_{i \in G_\ell} n_i}.
\]
Each group label is re-indexed from \(\{ \text{some set}\}\) to \(\{0,1,\dots\}\) for consistency.

---

## 5. Merge Groups That Are Too Close

```python
if len(labels) > 1:
    sort_phi = np.sort(phi_out)
    phi_diff = sort_phi[1:] - sort_phi[:-1]
    min_val = phi_diff.min()
    ...
    while min_val < least_diff:
        combine_ind = ...
        combine_to_ind = ...
        class_label[class_label == combine_ind] = combine_to_ind
        ...
        # recompute phi_out, etc.
```

### Explanation

- We sort the current group means in ascending order, then find consecutive gaps.  
- If the minimum gap among them is < **least_diff**, we **merge** those two groups (the smaller mean merges into the larger, or vice versa).  
- Recompute the merged group’s mean and repeat until no gap is below **least_diff** or only one group remains.

### Math Formulation

1. Sort \(\phi_{\text{out}}\) into \((\phi_{(1)}, \phi_{(2)}, \dots)\).  
2. For each adjacent pair \((\phi_{(\ell)}, \phi_{(\ell+1)})\), define:
   \[
     d_\ell = \phi_{(\ell+1)} - \phi_{(\ell)}.
   \]
3. If \(\min_\ell d_\ell < \text{least\_diff}\), merge those groups. Let’s say we merge group \(p\) and group \(q\); the new group’s mean is again the weighted average of all members.  
4. Repeat until all group means differ by at least \(\text{least\_diff}\), or only one group remains.

---

## 6. Final Piecewise-Constant Assignment

```python
phi_res = np.zeros(No_mutation)
for lab in range(len(phi_out)):
    phi_res[class_label == lab] = phi_out[lab]
```

### Explanation

- After merging, each group \(\ell\) has a final mean \(\phi_{\text{out},\ell}\).  
- We construct `phi_res`, an array of size `No_mutation`, where each mutation in group \(\ell\) is assigned the mean \(\phi_{\text{out},\ell}\).  

### Math Formulation

\[
\phi_{\text{res}, i} \;=\; \phi_{\text{out},\,\text{class\_label}(i)}.
\]
Hence, \(\phi_{\text{res}}\) is **piecewise-constant** over the groups formed by the post-processing steps.

---

## Summary

This code transforms the raw ADMM solutions \(\phi_{\hat i}\) into a **segmented**, more interpretable vector \(\phi_{\text{res}}\). The final grouping satisfies:

1. **Thresholding**: Differences \(\le \text{post_th}\) are forced to 0 → those mutations share a group.  
2. **Minimum Group Size**: No group remains smaller than **least_mut** → members of undersized groups get reassigned to their nearest (in difference) larger group.  
3. **Minimum Mean Separation**: Groups whose means differ by < **least_diff** are merged into a single group.  

The final **piecewise-constant** solution \(\phi_{\text{res}}\) ensures that:

- Close mutations (below `post_th`) collapse into the same cluster.
- Clusters do not remain too small.
- Cluster means remain sufficiently spaced (≥ `least_diff`), unless all are merged into one.