import torch
import argparse
import numpy as np
import pandas as pd
import os
import sys
from sklearn.cluster import KMeans

from clipp2.core_coreset import ADMMConfig, solve_coreset_path, reconstruct_full_solution_admm, post_process_solution
from clipp2.preprocess import combine_patient_samples, add_to_df, build_tensor_arrays

def run(args):
    # --- 1. Configuration Setup ---
    print("[1/7] Setting up configuration...")
    if args.lambda_sequence is None:
        # Default lambda sequence if not provided
        lambda_sequence = np.logspace(-2, 0, 20).tolist()
        
    config = ADMMConfig(
        device='cuda' if torch.cuda.is_available() else 'cpu',
        precision=args.precision,
        Run_limit=args.run_limit
    )
    print(f"      -> Using device: {config.device}")
    print(f"      -> Lambda path: {lambda_sequence}")
    
    # --- 2. Data Loading and Preparation ---
    if not os.path.isdir(args.input_dir):
        print(f"Error: input directory not found: {args.input_dir!r}")
        sys.exit(1)
    print(f"[2/7] Loading and combining patient samples from: {args.input_dir}")
    df = combine_patient_samples(args.input_dir)
    print("[3/7] Adding computed columns to DataFrame")
    df = add_to_df(df)
    print("[4/7] Building initial NumPy arrays...")
    dic_np = build_tensor_arrays(df)
    r_full, n_full, minor_full, total_full = dic_np['r'], dic_np['n'], dic_np['minor'], dic_np['total']
    pur_arr_full, coef_list_full = dic_np['pur_arr'], dic_np['coef_list']
    pairs, samples = dic_np['pairs'], dic_np['samples']
    M = len(samples)
    S = r_full.shape[0]
    coreset_size = max(100, np.sqrt(S).astype(int) * 10)

    print(f"      -> Coreset size K: {coreset_size}")

    with np.errstate(divide='ignore', invalid='ignore'):
        frac = r_full / n_full.clip(min=1)
        phi_hat_orig = frac * ((2.0 - 2.0*pur_arr_full) + (pur_arr_full * total_full) / minor_full.clip(min=1e-9))
        p_hat_mle_np = np.log(phi_hat_orig / (1 - phi_hat_orig).clip(min=1e-9)).clip(-10, 10)
        
    # --- 5. Prepare Data for Two-Phase Framework with Explicit Type Casting ---
    print("[5/7] Preparing data for two-phase solver...")

    r_full = r_full.astype(np.float32)
    n_full = n_full.astype(np.float32)
    minor_full = minor_full.astype(np.float32)
    total_full = total_full.astype(np.float32)
    pur_arr_full = pur_arr_full.astype(np.float32)
    p_hat_mle_np = np.log(phi_hat_orig / (1 - phi_hat_orig).clip(min=1e-9)).clip(-10, 10).astype(np.float32)

    kmeans = KMeans(n_clusters=coreset_size, random_state=42, n_init='auto')
    cluster_assign_np = kmeans.fit_predict(p_hat_mle_np).astype(np.int64)
    w_init_np = kmeans.cluster_centers_.astype(np.float32) 

    _, counts = np.unique(cluster_assign_np, return_counts=True)
    cluster_size_np = np.zeros(coreset_size, dtype=np.float32)
    unique_labels = np.unique(cluster_assign_np)
    cluster_size_np[unique_labels] = counts

    coef_list_full_32 = [c.astype(np.float32) for c in coef_list_full]

    init_data = {
        'r_t': torch.from_numpy(r_full).to(config.device),
        'n_t': torch.from_numpy(n_full).to(config.device),
        'purity_t': torch.from_numpy(pur_arr_full).to(config.device),
        'total_t': torch.from_numpy(total_full).to(config.device),
        'minor_t': torch.from_numpy(minor_full).to(config.device),
        'p_hat_mle': torch.from_numpy(p_hat_mle_np).to(config.device),
        'c_all_t': torch.stack([torch.from_numpy(c).to(config.device) for c in coef_list_full_32], dim=1),
        'w_init_t': torch.from_numpy(w_init_np).to(config.device),
    }
    init_data['purity_t'] = init_data['purity_t'].unsqueeze(0).expand(S, -1)
    cluster_data = {
        'cluster_assign': torch.from_numpy(cluster_assign_np).to(config.device),
        'cluster_size': torch.from_numpy(cluster_size_np).to(config.device)
    }

    raw_data_np_dict = {key: val.cpu().numpy() for key, val in init_data.items()}
    
    # --- 6. Run Two-Phase Pipeline ---
    print(f"[6/7] Running two-phase pipeline across {len(lambda_sequence)} lambda values...")
    # Phase 1: Run coreset path optimization
    coreset_path_results = solve_coreset_path(init_data, cluster_data['cluster_assign'], cluster_data['cluster_size'], lambda_sequence, config)

    all_results_rows = []
    # Phase 2 + Post-processing for each lambda
    for lam, coreset_res in coreset_path_results.items():
        print(f"  -> Processing lambda = {lam:.4f}")
        recon_config = ADMMConfig(Lambda=lam, precision=config.precision, Run_limit=config.Run_limit)
        
        # Phase 2
        recon_results = reconstruct_full_solution_admm(coreset_res, init_data, recon_config, cluster_data['cluster_size'])
        p_recon_np = recon_results['p_recon_t'].cpu().numpy()
        
        # Run post-processing
        post_processed_res = post_process_solution(
            p_final=p_recon_np,
            eta_final = None,
            raw_data_np=raw_data_np_dict
        )
        
        # Append rows for this lambda's result
        for mut_idx in range(r_full.shape[0]):
            chrom, pos = pairs.iloc[mut_idx]
            for j, region in enumerate(samples):
                all_results_rows.append({
                    'lambda': lam,
                    'chromosome_index': chrom,
                    'position': pos,
                    'region': region,
                    'label': post_processed_res['labels'][mut_idx],
                    'phi_final': post_processed_res['phi_final'][mut_idx, j],
                    'phi_hat_initial': phi_hat_orig[mut_idx, j],
                    'aic': post_processed_res['aic'],
                    'bic': post_processed_res['bic'],
                    'ebic': post_processed_res['ebic']
                })
                
    # --- 7. Final Output ---
    print("[7/7] Assembling final report and writing to file...")
    final_output_df = pd.DataFrame(all_results_rows)

    output_path = os.path.join(args.output_dir, os.path.basename(os.path.normpath(args.input_dir)))
    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, 'two_phase_framework_results.tsv')

    final_output_df.to_csv(output_file, sep='\t', index=False, float_format='%.6f')
    print(f"Done. Full results for all lambdas written to {output_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Scalable Two-Phase Multi-Sample Subclone Reconstruction Pipeline')
    parser.add_argument('--input_dir', default='input', help="Root directory of processed samples (e.g., 'patient_1/')")
    parser.add_argument('--output_dir', default='output', help="Root directory for output files")
    parser.add_argument('--lambda_sequence', type=str, default=None, help="Default is None, else Comma-separated list of Lambda values for the regularization path")
    parser.add_argument('--precision', type=float, default=1e-4, help="ADMM convergence precision")
    parser.add_argument('--run_limit', type=int, default=200, help="ADMM max iterations")
    parser.add_argument('--post_th', type=float, default=0.01, help="Threshold for post-processing eta norm")
    parser.add_argument('--least_diff', type=float, default=0.1, help="Distance threshold for final cluster merging")

    args = parser.parse_args()
    run(args)