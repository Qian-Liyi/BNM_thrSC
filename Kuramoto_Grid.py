import os

import bct
import numpy as np
import pandas as pd


def get_sim_options():
    return dict(
        duration=450,
        bold_remove_s=30,
        TR=0.6,
        sc_dist=None,
        dt="0.1",
        bw_dt="1.0",
        states_ts=False,
        states_sampling=None,
        noise_out=False,
        sim_seed=0,
        noise_segment_length=30,
        gof_terms=["+fc_corr", "-fcd_ks"],
        do_fc=True,
        do_fcd=True,
        window_size=30,
        window_step=5,
        fcd_drop_edges=True,
        exc_interhemispheric=True,
        bw_params="heinzle2016-3T",
        sim_verbose=False,
    )


def process_sc_matrix(sc_raw, threshold):
    sc = (sc_raw + sc_raw.T) / 2
    if threshold < 1.0:
        sc = bct.threshold_proportional(sc, threshold)
    sc = sc / (np.mean(sc) * 100)
    return sc


def load_empirical_data(sub, atlas, data_root="../Data/MICA/derivatives/connectome"):
    from cubnm import utils

    emp_bold = pd.read_csv(f"{data_root}/{atlas}/{sub}_ts.csv", index_col=0).values[
        :, :695
    ]
    if emp_bold.shape[0] > emp_bold.shape[1]:
        emp_bold = emp_bold.T

    emp_fc_tril = utils.calculate_fc(
        emp_bold, exc_interhemispheric=True, return_tril=True
    )
    emp_fcd_tril = utils.calculate_fcd(
        emp_bold,
        window_size=30,
        window_step=5,
        exc_interhemispheric=True,
        return_tril=True,
    )
    return emp_fc_tril, emp_fcd_tril


def run_optimization_core(
    gpu_id, subject_chunk, atlas, threshold, sc_mode, out_dir_root, sc_dir=None
):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    from cubnm import optimize

    sim_options = get_sim_options()

    print(f"[GPU {gpu_id}] Starting GridSearch for {len(subject_chunk)} subjects...")

    for sub in subject_chunk:
        threshold_str = f"{threshold:.2f}".replace(".", "_")
        work_dir = f"{out_dir_root}/{atlas}/p{threshold_str}/{sub}"
        os.makedirs(work_dir, exist_ok=True)

        if sc_mode == "individual":
            sc_raw = pd.read_csv(f"{sc_dir}/{atlas}/{sub}_sc.csv", index_col=0).values
            sc = process_sc_matrix(sc_raw, threshold)
        else:
            raise ValueError(f"Unknown sc_mode: {sc_mode}")

        emp_fc_tril, emp_fcd_tril = load_empirical_data(sub, atlas)

        problem = optimize.BNMProblem(
            model="Kuramoto",
            params={"G": (0.001, 10.0)},
            emp_fc_tril=emp_fc_tril,
            emp_fcd_tril=emp_fcd_tril,
            sc=sc,
            out_dir=work_dir,
            **sim_options,
        )
        grid_opt = optimize.GridOptimizer()

        try:
            grid_opt.optimize(problem, grid_shape={"G": 100})
            grid_opt.save()

            best_g = grid_opt.opt["G"]
            print(f"  > Subject {sub} done. Best G: {best_g:.4f}")

        except Exception as e:
            print(f"  > [GPU {gpu_id}] Failed on subject {sub}: {str(e)}")

    print(f"[GPU {gpu_id}] All tasks completed.")
