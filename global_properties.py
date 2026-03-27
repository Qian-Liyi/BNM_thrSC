import os
from pathlib import Path

import bct
import numpy as np
import pandas as pd
import small_world_propensity
from joblib import Parallel, delayed
from tqdm import tqdm


def calculate_sc_global(W):
    W_copy = W.copy()
    np.fill_diagonal(W_copy, 0)

    L = bct.weight_conversion(W_copy, "lengths")

    _, q = bct.community_louvain(W_copy)

    D, _ = bct.distance_wei(L)

    _, ge, _, _, _ = bct.charpath(D)

    density, _, _ = bct.density_und(W_copy)

    return {
        "Density": float(density),
        "Modularity": float(q),
        "Global_Efficiency": float(ge),
    }


def load_sc(sc_path):
    sub = sc_path.parent.parent.name
    density = float(
        sc_path.parent.parent.parent.name.replace("p", "")
        .replace("w", "")
        .replace("_", ".")
    )
    sc_matrix = np.loadtxt(sc_path)
    return sub, density, sc_matrix


def process(sc_path):
    sub, density, sc_matrix = load_sc(sc_path)
    results = calculate_sc_global(sc_matrix)
    results["Density"] = density
    results["Subject"] = sub
    return results


for atlas in [
    "aparc",
    "schaefer100",
    "schaefer100_subcort",
    "schaefer200",
    "schaefer300",
]:
    print(f"Processing atlas: {atlas}")
    sc_dir = Path(f"../output/rWW/IndividualSC_Density/{atlas}/")
    sc_files = list(sc_dir.rglob("*//sc.txt"))

    results_list = Parallel(n_jobs=30)(delayed(process)(p) for p in tqdm(sc_files))
    results_df = pd.DataFrame(results_list)

    # Calculate Small-World Propensity (SWP) for each SC matrix
    SWP = small_world_propensity.small_world_propensity(
        [load_sc(p)[2] for p in sc_files], bin=[False] * len(sc_files)
    )

    # Combine results with SWP and save to CSV
    results_df = pd.concat([results_df, SWP], axis=1)
    os.makedirs("../output/graph", exist_ok=True)
    results_df.to_csv(
        f"../output/graph/global_properties_results_{atlas}.csv",
        index=False,
    )
