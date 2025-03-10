import numpy as np
import math
import os

# Soft-threshold operator equivalent to ST(double x, double lam) in C++
def ST(x, lam):
    val = abs(x) - lam
    if x == 0 or val <= 0:
        return 0.0
    return val if x > 0 else -val

def expit(a):
    return 1.0 / (1.0 + np.exp(-a))

def logit(a):
    return np.log(a / (1.0 - a))

def CliPPIndividual(
    No_mutation,
    r,
    n,
    minor_,
    total,
    ploidy,
    Lambda,
    alpha_init,
    rho,
    gamma,
    Run_limit,
    precision,
    control_large,
    least_mut,
    post_th,
    least_diff,
    coef_1d,
    wcut_1d,
    purity,
    preliminary_folder,
    theta_hat,
    phi_hat,
    ids,
    DELTA,
    scale_parameter
):
    """
    Python translation of CliPPIndividual(...).
    This function performs the iterative optimization, clustering, and output of results.
    """

    # Initialize
    alpha = alpha_init
    w_new = np.zeros((No_mutation, 1))
    # Initialize w using logit of scaled phi_hat, with bounds
    for i in range(No_mutation):
        temp = phi_hat[i, 0] / scale_parameter

        high = expit(control_large)   # upper bound on expit
        low  = expit(-control_large)  # lower bound on expit

        temp = max(min(temp, high), low)
        temp = logit(temp)
        temp = max(min(temp, control_large), -control_large)
        w_new[i, 0] = temp

    # Initialize eta
    eta_new = np.zeros((No_mutation * (No_mutation - 1) // 2, 1))
    for i in range(No_mutation * (No_mutation - 1) // 2):
        i1, i2 = ids[i, 0], ids[i, 1]
        eta_new[i, 0] = w_new[i1, 0] - w_new[i2, 0]

    tau_new = np.ones((No_mutation * (No_mutation - 1) // 2, 1))

    # Prepare arrays for iteration
    A = np.zeros((No_mutation, 1))
    B = np.zeros((No_mutation, 1))
    theta = np.zeros((No_mutation, 1))

    # Iterative updates
    residual = 100.0
    k = 0

    # Keep track of SNVs that we clamp (theta >= 1.0)
    problematic_snvs = []

    while residual > precision and k < Run_limit:
        k += 1
        print(f"\rLambda: {Lambda}\titeration: {k}\tresidual: {residual}", end="")

        w_old = w_new.copy()
        eta_old = eta_new.copy()
        tau_old = tau_new.copy()

        # Update A, B, and theta
        for i in range(No_mutation):
            w_val = w_old[i, 0]
            # Compute new theta
            #    theta = exp(w)/(2 + exp(w)*total) * minor
            # This corresponds to "theta(i, 0) = exp(w_old(i, 0)) * minor_(i, 0) / (2 + ... )"
            theta[i, 0] = math.exp(w_val) * minor_[i, 0] / (2.0 + math.exp(w_val) * total[i, 0])

            # Identify which side of cuts (wcut_1d)
            w_low_cut  = wcut_1d[2*i + 0, 0]
            w_high_cut = wcut_1d[2*i + 1, 0]

            if w_val <= w_low_cut:
                tag1, tag3 = 1.0, 0.0
            else:
                tag1, tag3 = 0.0, 1.0

            if w_val >= w_high_cut:
                tag2, tag4 = 1.0, 0.0
            else:
                tag2, tag4 = 0.0, 1.0

            # Check clamp for large or invalid theta
            if theta[i, 0] >= 1.0:
                theta[i, 0] = 0.99
                problematic_snvs.append(i)

            # sqrt(n(i,0)) * [ (tag1*coef[i*6+1] + tag2*coef[i*6+5] + tag3*tag4*coef[i*6+3]) - theta_hat(i,0) ] / sqrt(theta(1-theta))
            denom = math.sqrt(theta[i, 0] * (1.0 - theta[i, 0])) if theta[i, 0] < 1.0 else 1e-8
            A[i, 0] = (
                math.sqrt(n[i, 0])
                * (
                    tag1 * coef_1d[i*6 + 1, 0]
                    + tag2 * coef_1d[i*6 + 5, 0]
                    + tag3 * tag4 * coef_1d[i*6 + 3, 0]
                    - theta_hat[i, 0]
                )
                / denom
            )
            B[i, 0] = (
                math.sqrt(n[i, 0])
                * (
                    tag1 * coef_1d[i*6 + 0, 0]
                    + tag2 * coef_1d[i*6 + 4, 0]
                    + tag3 * tag4 * coef_1d[i*6 + 2, 0]
                )
                / denom
            )

        # linear = DELTA * (alpha * eta_old + tau_new) - B .* A
        # We can do piecewise: first compute alpha*eta_old + tau_new, then multiply by DELTA
        # DELTA is shape (No_mutation, No_mutation*(No_mutation-1)//2)
        # so DELTA.dot(...) => shape (No_mutation, 1)

        # step1: cvec = alpha * eta_old + tau_new
        cvec = alpha * eta_old + tau_new
        # linear = DELTA dot cvec - B*C   (where C = A elementwise)
        # We'll build that product manually:

        # Build the vector DELTA * cvec
        # Each row i in DELTA has exactly two nonzero entries: +1 at col p, -1 at col q
        # because of the way DELTA was constructed.
        linear = np.zeros((No_mutation, 1))
        for idx in range(DELTA.shape[1]):
            i1, i2 = ids[idx, 0], ids[idx, 1]  # which rows get +1 / -1
            val = cvec[idx, 0]
            linear[i1, 0] += val
            linear[i2, 0] -= val

        BC = B * A  # elementwise
        linear -= BC

        # We want to invert diagonal M = B^2 + (No_mutation)*alpha (elementwise)
        Mvals = B[:, 0]*B[:, 0] + (No_mutation * alpha)
        Minv = 1.0 / Mvals  # diagonal inverse
        # We want [Minv - (alpha/(1+trace_g))*Minv_outer] * linear, 
        # but let's compute the trace factor from the code:
        # trace_g = -alpha * sum(Minv), so:
        trace_g = -alpha * np.sum(Minv)
        if np.isnan(trace_g):
            print(f"Lambda: {Lambda}\titeration: {k}\tEncountered NaN.")
            return -1

        # The update formula:
        # w_new = (Minv_diag - [1/(1+trace_g)] * (-alpha)*Minv_outer ) * linear
        # We can replicate the "woodbury" style update:
        # If we let D = diag(Minv),
        # then the matrix is D - (1/(1+trace_g)) * (-alpha) * (Minv * Minv^T).
        # We'll do this by splitting it up carefully:
        #   part1 = D*linear
        #   part2 = ...
        # Because D is diagonal, D*linear => elementwise multiply
        part1 = Minv[:, np.newaxis] * linear
        # Minv_outer = Minv * Minv^T => shape (No_mutation, No_mutation), but that might be large.
        # We'll do an equivalent vector-based approach:
        # correction = ( (-alpha)/(1+trace_g) ) * (Minv * (Minv dot linear))  (outer product)
        # The scalar "Minv dot linear" is sum(Minv[i]*linear[i]).
        dot_val = np.sum(Minv * linear[:, 0])
        correction_factor = (-alpha / (1.0 + trace_g)) * dot_val
        # Then final w_new = part1 - correction_factor * Minv
        #   (since outer product with Minv means we multiply each row by Minv[i], but we sum only once)
        w_new = part1 - correction_factor * Minv[:, np.newaxis]

        # Bound w_new
        w_new = np.clip(w_new, -control_large, control_large)

        # Update eta, tau
        # delt(i) = w_new(i1) - w_new(i2) - 1/alpha * tau_old(i)
        delt = np.zeros((No_mutation*(No_mutation-1)//2, 1))
        for i in range(delt.shape[0]):
            i1, i2 = ids[i, 0], ids[i, 1]
            delt[i, 0] = w_new[i1, 0] - w_new[i2, 0] - (1.0/alpha)*tau_old[i, 0]

        for i in range(delt.shape[0]):
            temp = delt[i, 0]
            # replicate the piecewise logic for the fused-lasso style penalty
            if abs(temp) > gamma * Lambda:
                tag1, tag3 = 1.0, 0.0
            else:
                tag1, tag3 = 0.0, 1.0

            if abs(temp) < (Lambda + Lambda/alpha):
                tag2, tag4 = 1.0, 0.0
            else:
                tag2, tag4 = 0.0, 1.0

            # eta_new formula is a combination of ST and these tags
            val_st1 = ST(temp, Lambda/alpha)
            val_st2 = ST(temp, gamma*Lambda/((gamma - 1.0)*alpha)) / (1.0 - 1.0/((gamma-1.0)*alpha)) if (gamma != 1.0) else 0.0

            eta_new[i, 0] = temp*tag1 + val_st1*tag2 + val_st2*tag3*tag4
            tau_new[i, 0] = tau_old[i, 0] - alpha*( (w_new[i1, 0] - w_new[i2, 0]) - eta_new[i, 0] )

        # Decay alpha
        alpha *= rho

        # Compute residual = max(| w_new(i1)-w_new(i2) - eta_new |)
        max_val = 0.0
        for i in range(eta_new.shape[0]):
            i1, i2 = ids[i, 0], ids[i, 1]
            diff_val = abs((w_new[i1, 0] - w_new[i2, 0]) - eta_new[i, 0])
            if diff_val > max_val:
                max_val = diff_val
        residual = max_val

    print("")

    # Remove duplicates in problematic_snvs
    problematic_snvs = list(set(problematic_snvs))
    problematic_snvs.sort()

    # Post-processing: threshold small |eta| to 0
    for i in range(No_mutation*(No_mutation-1)//2):
        if abs(eta_new[i, 0]) <= post_th:
            eta_new[i, 0] = 0.0

    # Build a matrix diff(i,j) = eta_new(...) for j>i
    diff_mat = np.zeros((No_mutation, No_mutation))
    for i in range(eta_new.shape[0]):
        i1, i2 = ids[i, 0], ids[i, 1]
        diff_mat[i1, i2] = eta_new[i, 0]

    # Cluster SNVs based on zero-differences
    class_label = np.full(No_mutation, -1, dtype=np.int32)
    class_label[0] = 0
    group_size = [1]
    label_next = 1

    for i in range(1, No_mutation):
        assigned = False
        for j in range(i):
            # if diff_mat(j,i)==0 => same cluster
            if diff_mat[j, i] == 0.0:
                class_label[i] = class_label[j]
                group_size[class_label[j]] += 1
                assigned = True
                break
        if not assigned:
            class_label[i] = label_next
            group_size.append(1)
            label_next += 1

    # If the smallest group is < least_mut => refine
    temp_size = No_mutation
    for sz in group_size:
        if sz > 0 and sz < temp_size:
            temp_size = sz

    # We only refine if temp_size < least_mut
    refine = 1 if (temp_size < least_mut) else 0

    while refine == 1:
        refine = 0
        # find group(s) that match temp_size
        tmp_grp = [i for i, sz in enumerate(group_size) if sz == temp_size]
        if len(tmp_grp) == 0:
            break

        # Take the first such group, rearrange membership
        group_to_refine = tmp_grp[0]
        # all SNVs in that group
        members = [i for i in range(No_mutation) if class_label[i] == group_to_refine]

        tmp_diff = np.zeros((No_mutation, 1))
        for snv_index in members:
            if snv_index != 0 and snv_index != (No_mutation - 1):
                # Build tmp_diff
                for jj in range(snv_index):
                    tmp_diff[jj, 0] = abs(diff_mat[jj, snv_index])
                tmp_diff[snv_index, 0] = 1e8
                for jj in range(snv_index+1, No_mutation):
                    tmp_diff[jj, 0] = abs(diff_mat[snv_index, jj])
                # exclude the same cluster members
                for m2 in members:
                    tmp_diff[m2, 0] += 1e8
            else:
                # edge cases snv_index=0 or snv_index=No_mutation-1
                if snv_index == 0:
                    tmp_diff[0, 0] = 1e8
                    for jj in range(1, No_mutation):
                        tmp_diff[jj, 0] = abs(diff_mat[0, jj])
                    for m2 in members:
                        tmp_diff[m2, 0] += 1e8
                else:
                    for jj in range(No_mutation-1):
                        tmp_diff[jj, 0] = abs(diff_mat[jj, No_mutation-1])
                    tmp_diff[No_mutation-1, 0] = 1e8
                    for m2 in members:
                        tmp_diff[m2, 0] += 1e8

            # find min
            new_label_idx = np.argmin(tmp_diff[:, 0])
            # reassign
            group_size[class_label[snv_index]] -= 1
            class_label[snv_index] = class_label[new_label_idx]
            group_size[class_label[snv_index]] += 1

        # re-check
        temp_size = No_mutation
        for sz in group_size:
            if sz > 0 and sz < temp_size:
                temp_size = sz
        tmp_grp = [i for i, sz in enumerate(group_size) if sz == temp_size]
        refine = 1 if (temp_size < least_mut) else 0

    # Recompute unique labels and sort
    unique_labels = np.unique(class_label)
    phi_out = np.zeros(len(unique_labels))

    # For each label, compute average (weighted by n) of phi_hat
    for idx, lab in enumerate(unique_labels):
        indices = np.where(class_label == lab)[0]
        tmp_sum = 0.0
        tmp_n   = 0.0
        for i in indices:
            tmp_sum += phi_hat[i, 0]*n[i, 0]
            tmp_n   += n[i, 0]
        phi_val = tmp_sum / tmp_n if tmp_n > 0 else 0.0
        phi_out[idx] = phi_val
        # re-map those class_label to the "compressed" idx
        class_label[indices] = idx

    # Merge extremely close clusters if needed
    if len(phi_out) > 1:
        sort_phi = np.sort(phi_out)
        phi_diff = np.diff(sort_phi)
        min_ind = np.argmin(phi_diff)
        min_val = phi_diff[min_ind]

        # Keep track of repeated merges
        merges_history = [min_val]

        while min_val < least_diff:
            # find cluster indices in phi_out
            # merges the cluster with value sort_phi[min_ind] into the cluster with value sort_phi[min_ind+1]
            from_val = sort_phi[min_ind]
            to_val   = sort_phi[min_ind+1]
            from_idx = np.where(phi_out == from_val)[0]
            to_idx   = np.where(phi_out == to_val)[0]
            if len(from_idx) == 0 or len(to_idx) == 0:
                break

            from_idx = from_idx[0]
            to_idx   = to_idx[0]

            # Reassign all from_idx labels to to_idx
            class_label[class_label == from_idx] = to_idx

            # Recompute unique labels & phi_out
            unique_labels = np.unique(class_label)
            new_phi_out = np.zeros(len(unique_labels))
            for i2, labval in enumerate(unique_labels):
                indices = np.where(class_label == labval)[0]
                tmp_sum = 0.0
                tmp_n   = 0.0
                for m2 in indices:
                    tmp_sum += phi_hat[m2, 0]*n[m2, 0]
                    tmp_n   += n[m2, 0]
                phi_val = tmp_sum/tmp_n if tmp_n > 0 else 0.0
                new_phi_out[i2] = phi_val
                class_label[indices] = i2

            phi_out = new_phi_out

            if len(phi_out) == 1:
                break

            sort_phi = np.sort(phi_out)
            phi_diff = np.diff(sort_phi)
            if len(phi_diff) == 0:
                break
            min_ind = np.argmin(phi_diff)
            min_val = phi_diff[min_ind]
            merges_history.append(min_val)

            # If we've repeatedly found the same minimal difference, we can stop to avoid infinite loops
            if len(merges_history) > 4:
                # If all merges are the same min_val => might be stuck
                all_same = all(abs(x - merges_history[0]) < 1e-12 for x in merges_history)
                if all_same:
                    print(f"\nLambda: {Lambda}\titeration: {k}\tfailed merges due to repeating minimal differences.")
                    return -1

    # Finally, write the outputs
    # Convert float Lambda to a string without trailing zeros:
    lambda_str = f"{Lambda}".rstrip('0').rstrip('.')

    phi_file_path = os.path.join(preliminary_folder, f"lam{lambda_str}_phi.txt")
    label_file_path = os.path.join(preliminary_folder, f"lam{lambda_str}_label.txt")
    summary_file_path = os.path.join(preliminary_folder, f"lam{lambda_str}_summary_table.txt")

    try:
        with open(phi_file_path, 'w') as phi_file, \
             open(label_file_path, 'w') as label_file, \
             open(summary_file_path, 'w') as summary_file:

            # For each mutation, write the final phi and label
            for i in range(len(class_label)):
                group_idx = class_label[i]
                res = phi_out[group_idx]
                phi_file.write(f"{round(res*1000.0)/1000.0}\n")
                label_file.write(f"{group_idx}\n")

            # Summary: group index, size, cluster mean
            for i, val in enumerate(phi_out):
                count_grp = np.sum(class_label == i)
                summary_file.write(f"{i}\t{count_grp}\t{round(val*1000.0)/1000.0}\n")
    except IOError:
        print(f"Cannot open output files for Lambda = {Lambda}.")
        return 1

    # If we have problematic SNVs, write them out
    if len(problematic_snvs) > 0:
        problematic_snvs_file_path = os.path.join(preliminary_folder, f"lam{lambda_str}_problematic_snvs.txt")
        try:
            with open(problematic_snvs_file_path, 'w') as f:
                for snv_idx in problematic_snvs:
                    f.write(f"{snv_idx}\n")
        except IOError:
            print(f"Cannot open file {problematic_snvs_file_path}")
            return 1

    return 0


def CliPPCPP(
    No_mutation, 
    c_r, 
    c_n, 
    c_minor, 
    c_total, 
    ploidy, 
    Lambda_list, 
    Lambda_num, 
    alpha, 
    rho, 
    gamma, 
    Run_limit, 
    precision, 
    control_large, 
    least_mut, 
    post_th, 
    least_diff, 
    c_coef_1d, 
    c_wcut_1d, 
    purity, 
    preliminary_folder
):
    """
    Python translation of the main CliPPCPP(...) logic.
    Prepares input arrays, constructs the DELTA (difference) matrix,
    then loops over each Lambda and calls CliPPIndividual.
    """

    # Convert input lists to numpy arrays
    r       = np.array(c_r,      dtype=float).reshape((-1,1))
    n       = np.array(c_n,      dtype=float).reshape((-1,1))
    minor_  = np.array(c_minor,  dtype=float).reshape((-1,1))
    total   = np.array(c_total,  dtype=float).reshape((-1,1))
    wcut_1d = np.array(c_wcut_1d,dtype=float).reshape((-1,1))
    coef_1d = np.array(c_coef_1d,dtype=float).reshape((-1,1))

    # Precompute theta_hat, phi_hat
    theta_hat = r / n  # shape (No_mutation,1)
    # phi_hat = theta_hat*( (ploidy - purity*ploidy + purity*total)/minor )
    phi_hat = theta_hat * ((ploidy - purity*ploidy + purity*total)/minor_)

    # Scale parameter
    phi_hat_max = np.max(phi_hat)
    scale_parameter = max(1.0, phi_hat_max)

    # Build IDs list (for i<j)
    # Also row1, row2, but in Python we can keep them in ids
    pairs = []
    for i in range(No_mutation):
        for j in range(i+1, No_mutation):
            pairs.append((i, j))
    ids = np.array(pairs, dtype=int).reshape((-1,2))

    # Build the DELTA matrix in a (row, col) format
    # shape: (No_mutation, #pairs)
    # each column has +1 in row i, -1 in row j
    DELTA = np.zeros((No_mutation, ids.shape[0]))
    for col_idx in range(ids.shape[0]):
        i1, i2 = ids[col_idx]
        DELTA[i1, col_idx] =  1.0
        DELTA[i2, col_idx] = -1.0

    # Loop over each Lambda
    for lambda_index in range(Lambda_num):
        Lambda = Lambda_list[lambda_index]
        # Call CliPPIndividual
        ret = CliPPIndividual(
            No_mutation,
            r,
            n,
            minor_,
            total,
            ploidy,
            Lambda,
            alpha,
            rho,
            gamma,
            Run_limit,
            precision,
            control_large,
            least_mut,
            post_th,
            least_diff,
            coef_1d,
            wcut_1d,
            purity,
            preliminary_folder,
            theta_hat,
            phi_hat,
            ids,
            DELTA,
            scale_parameter
        )
        if ret != 0:
            print(f"CliPPIndividual returned error code {ret} for Lambda={Lambda}.")

    return 0

def CliPP(
    No_mutation,
    r,
    n,
    minor_,
    total_,
    ploidy,
    Lambda_list,
    Lambda_num,
    alpha,
    rho,
    gamma,
    Run_limit,
    precision,
    control_large,
    least_mut,
    post_th,
    least_diff,
    coef_1d,
    wcut_1d,
    purity,
    preliminary_folder
):
    """
    This is the Python equivalent of the extern "C" function interface
    that calls CliPPCPP(...).
    """

    return CliPPCPP(
        No_mutation,
        r,
        n,
        minor_,
        total_,
        ploidy,
        Lambda_list,
        Lambda_num,
        alpha,
        rho,
        gamma,
        Run_limit,
        precision,
        control_large,
        least_mut,
        post_th,
        least_diff,
        coef_1d,
        wcut_1d,
        purity,
        preliminary_folder
    )