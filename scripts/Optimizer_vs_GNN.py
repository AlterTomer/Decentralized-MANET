import torch
from pathlib import Path
from configparser import ConfigParser
from models.models import ChainedGNN
from utils.DataUtils import generate_graph_data
from utils.ComparisonUtils import evaluate_across_snr
from utils.EstimationUtils import masked_band_variance_from_dataset, precompute_csi_estimates
from utils.ConfigUtils import parse_args, load_ini_config
from utils.ParseUtils import parse_tx_rx_data
from visualization.GraphingAux import plot_mean_rate_vs_snr
import pickle
import csv
from MANET_FFN.MANET_FFN_Dataset import FFNDataset
from MANET_FFN.model import FFNPowerAllocator

# ====== config ======
# args = parse_args()
# cfg_path = args.config.resolve()
# parser = load_ini_config(cfg_path)
# print(f"Loaded config from CLI: {cfg_path}")

cfg_arr = [r"C:\Users\alter\OneDrive\Desktop\PhD\Decentralized MANET\Config Files\Single Tx-Rx\comp_unicast.ini",
           r"C:\Users\alter\OneDrive\Desktop\PhD\Decentralized MANET\Config Files\Multicommodity\comp_multicommodity.ini",
           r"C:\Users\alter\OneDrive\Desktop\PhD\Decentralized MANET\Config Files\Multiunicast\comp_multiunicast.ini",
           r"C:\Users\alter\OneDrive\Desktop\PhD\Decentralized MANET\Config Files\Multicast\comp_multicast.ini"]

for path in cfg_arr:
    # cfg_path = r"C:\Users\alter\OneDrive\Desktop\PhD\Decentralized MANET\Config Files\Multiunicast\comp_multiunicast.ini"
    cfg_path = path
    parser = ConfigParser()
    parser.read_file(open(cfg_path))
    print(f"Loaded default config: {cfg_path}")

    USE_AMP = torch.cuda.is_available()
    # Training Parameters
    train_params = parser["Train Parameters"]
    SEED = int(train_params["SEED"])
    MODE = train_params["mode"]  # "single" | "multicast" | "multi" | "converge" | "multiunicast"
    B = int(train_params["B"])
    L = int(train_params["L"])
    n = int(train_params["n"])
    sigma = float(train_params["sigma"])
    DROPOUT = float(train_params["dropout"])
    num_samples = int(train_params["num samples"])
    est_csi = True if int(train_params["LMMSE estimation"]) == 1 else False
    # Multi-message (K>1) message reduction for ALL methods at eval: "fair" = eq (10)
    # max-min over messages (the paper); "sum"/"mean" = throughput. Must be identical
    # across every method for a fair comparison. Ignored for single/multicast.
    REDUCE = train_params.get("reduce", "sum").strip().lower()
    Z_MODE = train_params.get("z mode", "edge").strip().lower()
    # Must match how the checkpoint was trained (changes node input dim). Loading is strict=False,
    # so a mismatch would SILENTLY drop layer-0 weights and give a broken model — keep in sync
    # with the training config's "noise conditioning".
    NOISE_CONDITIONING = int(train_params.get("noise conditioning", 0)) == 1

    # ---- Diagnostics / decentralized GNN inference controls ----
    COLLECT_DIAGNOSTICS = int(train_params.get("collect diagnostics", 1)) == 1
    _diag_snr_raw = train_params.get("diagnostics snr", "").strip()
    DIAGNOSTICS_SNR = None if _diag_snr_raw == "" else [int(x) for x in _diag_snr_raw.replace(" ", "").split(",") if x]

    multi_like = MODE in {"multi", "converge", "multiunicast"}
    # Default: if this is a multi-message candidate-Z model, evaluate the GNN using
    # decentralized candidate inference unless the config explicitly disables it.
    _default_dec = 1 if (multi_like and Z_MODE == "candidate") else 0
    DECENTRALIZED_GNN_INFERENCE = int(train_params.get("decentralized gnn inference", _default_dec)) == 1
    MAX_ROUTE_HOPS = int(train_params.get("max route hops", L))
    HARD_CANDIDATE_GNN_INFERENCE = int(train_params.get("hard candidate gnn inference", 1)) == 1

    if DECENTRALIZED_GNN_INFERENCE and not (multi_like and Z_MODE == "candidate"):
        print(
            f"[warn] decentralized gnn inference requested, but MODE={MODE} and z_mode={Z_MODE}. "
            "It will have no effect unless MODE is multi-like and z_mode='candidate'."
        )

    print(
        "[benchmark config] "
        f"mode={MODE} | B={B} | L={L} | n={n} | K_cfg will be inferred | reduce={REDUCE} | "
        f"z_mode={Z_MODE} | decentralized_gnn_inference={DECENTRALIZED_GNN_INFERENCE} | "
        f"max_route_hops={MAX_ROUTE_HOPS} | hard_candidate_gnn_inference={HARD_CANDIDATE_GNN_INFERENCE} | "
        f"collect_diagnostics={COLLECT_DIAGNOSTICS} | "
        f"diagnostics_snr={'all' if DIAGNOSTICS_SNR is None else DIAGNOSTICS_SNR}"
    )
    # Files Parameters
    files_params = parser["Files"]
    try:
        channel_path = files_params["channel path"]
    except KeyError:
        channel_path = None
    fig_path = files_params["fig path"]
    ffn_path = files_params.get("ffn path", None)  # unused (FFN disabled); .get avoids KeyError
    model_path = files_params["model path"]
    fig_data_path = files_params["fig data path"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_list = [n] * num_samples
    # tx, rx can be int OR a list in the ini; handle both
    raw_tx = train_params["tx"].strip()
    raw_rx = train_params["rx"].strip()
    tx, rx = parse_tx_rx_data(raw_tx, raw_rx)

    # replicate tx, rx per sampl
    tx_list = [tx] * num_samples  # each sample may have a list of receivers
    rx_list = [rx] * num_samples  # each sample may have a list of receivers

    if MODE == "single":
        K_cfg = 1
    elif MODE in {"multicast", "multi"}:
        K_cfg = len(rx)
    elif MODE == "converge":
        K_cfg = len(tx)
    else:  # "multiunicast"
        if len(tx) != len(rx):
            raise ValueError("tx and rx must have the same length for multiunicast.")
        K_cfg = len(tx)

    sigma_list = [sigma] * num_samples

    dataset = generate_graph_data(
        n_list=n_list,
        tx_list=tx_list,
        rx_list=rx_list,
        sigma_list=sigma_list,
        B=B,
        K=K_cfg,
        problem=MODE,
        seed=SEED,
        channel_path=channel_path,
        device='cpu'
    )

    # The FFN benchmark dataset is built later, AFTER the est-CSI swap below, so it sees the
    # same CSI (true or estimated) as the GNN. See the "FFN benchmark" block further down.

    if est_csi:
        print("Using estimated CSI")
        prior_var = masked_band_variance_from_dataset(dataset)
        dataset = precompute_csi_estimates(
            dataset,
            pilots_M=4,
            pilot_power=1,
            prior_var=prior_var,
            est_noise_std=None,
            seed=SEED,
            device=torch.device('cpu'),
        )
    else:
        print("Using True CSI")

    # Choose K for the model:
    # - single: K_model = 1
    # - multicast: K_model = K_cfg (to enable per-receiver role channels; still one shared message)
    # - multi, converge, multiunicast: K_model = K_cfg (distinct messages, produces [B,K,n,n] + Z)
    if MODE == "single":
        K_model = 1
    elif MODE in {"multicast", "multi", "converge", "multiunicast"}:
        K_model = K_cfg
    else:
        raise ValueError("MODE must be 'multicast', 'multi', 'converge', or 'multiunicast'.")

    # MANET-GNN Model
    model = ChainedGNN(
        num_layers=L,
        B=B,
        K=K_model,
        problem=MODE,
        dropout=DROPOUT,
        use_jk=True,
        jk_mode="concat",
        noise_conditioning=NOISE_CONDITIONING,
        z_mode=Z_MODE,
    ).to(device).eval()
    ckpt = torch.load(model_path, map_location=device, weights_only=False)

    state_dict = ckpt["model_state_dict"]
    new_state_dict = {}

    for k, v in state_dict.items():
        # remap old names -> new names
        if k.startswith("head."):
            new_k = k.replace("head.", "p_head.")
            new_state_dict[new_k] = v
        else:
            new_state_dict[k] = v
    # Guard: strict=False silently DROPS shape-mismatched tensors, which would load a broken
    # (partly random) model. The usual cause is 'noise conditioning' disagreeing with how the
    # checkpoint was trained (it changes the node input dim -> layer-0 weight shapes). Fail loud.
    _msd = model.state_dict()
    _bad = [(k, tuple(v.shape), tuple(_msd[k].shape))
            for k, v in new_state_dict.items() if k in _msd and v.shape != _msd[k].shape]
    if _bad:
        raise RuntimeError(
            "Checkpoint/model shape mismatch (strict=False would silently drop these). "
            f"Check that 'noise conditioning' ({NOISE_CONDITIONING}) matches the trained checkpoint. "
            f"Mismatches: {_bad[:4]}"
        )
    # model.load_state_dict(ckpt["model_state_dict"])
    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    print(f"[checkpoint] loaded: {model_path}")
    print(f"[checkpoint] missing_keys={len(missing)} | unexpected_keys={len(unexpected)}")
    if missing:
        print(f"[checkpoint] first missing keys: {missing[:8]}")
    if unexpected:
        print(f"[checkpoint] first unexpected keys: {unexpected[:8]}")
    print(
        f"[GNN inference settings] z_mode={getattr(model, 'z_mode', 'edge')} | "
        f"decentralized_gnn_inference={DECENTRALIZED_GNN_INFERENCE} | "
        f"max_route_hops={MAX_ROUTE_HOPS} | hard_candidate_gnn_inference={HARD_CANDIDATE_GNN_INFERENCE}"
    )

    # ====== FFN benchmark (optional; requires an FFN checkpoint at `ffn path`) ======
    # Build the FFN dataset from the (possibly estimate-swapped) `dataset` so the FFN sees the
    # SAME CSI as the GNN: true CSI when est_csi is off, H_hat when on. Using the pre-swap
    # links_list here would silently feed the FFN true CSI while the GNN gets the estimate.
    ffn_model = None
    ffn_dataset = None
    if ffn_path is not None:
        try:
            ffn_adj_list = [d.adj_matrix.detach().cpu() for d in dataset]
            ffn_links_list = [d.links_matrix.detach().cpu() for d in dataset]
            ffn_dataset = FFNDataset(
                adj_list=ffn_adj_list,
                links_list=ffn_links_list,
                tx_list=tx_list,
                rx_list=rx_list,
                sigma_list=sigma_list,
                B=B,
                problem=MODE,
                K=K_cfg,
            )

            ffn_cfg = torch.load(Path(ffn_path) / "ffn_run_config.pt", weights_only=False)
            # noise_conditioning MUST match the checkpoint (it changes the FFN input dim).
            # Saved in ffn_run_config.pt by MANET_FFN/main.py.
            ffn_model = FFNPowerAllocator(
                n_nodes=ffn_cfg['n'],
                n_bands=ffn_cfg['B'],
                K=ffn_cfg['K_cfg'],
                problem=MODE,
                hidden_dim=ffn_cfg['hidden_dim'],
                num_layers=ffn_cfg['num_layers'],
                dropout=ffn_cfg['dropout'],
                use_layernorm=True,
                noise_conditioning=ffn_cfg.get('noise_conditioning', False),
            )
            ffn_state_dict = torch.load(Path(ffn_path) / "ffn_checkpoint.pt", weights_only=False)['model_state_dict']
            ffn_model.load_state_dict(ffn_state_dict)
            ffn_model.to(device).eval()
            print(f"[FFN] loaded: {ffn_path} | noise_conditioning={ffn_cfg.get('noise_conditioning', False)} | "
                  f"est_csi={est_csi} (FFN sees {'estimated' if est_csi else 'true'} CSI)")
        except Exception as e:
            print(f"[FFN] disabled — could not load FFN from '{ffn_path}': {e}")
            ffn_model = None
            ffn_dataset = None
    else:
        print("[FFN] disabled — no 'ffn path' in config.")


    g = torch.Generator().manual_seed(SEED)

    snr_db_list = list(range(0, 51, 2))
    results = evaluate_across_snr(
        dataset,
        model,
        B,
        snr_db_list,
        problem=MODE,
        reduce=REDUCE,
        collect_diagnostics=COLLECT_DIAGNOSTICS,
        diagnostics_snr=DIAGNOSTICS_SNR,
        decentralized_gnn_inference=DECENTRALIZED_GNN_INFERENCE,
        max_route_hops=MAX_ROUTE_HOPS,
        hard_candidate_gnn_inference=HARD_CANDIDATE_GNN_INFERENCE,
        ffn_model=ffn_model,
        ffn_dataset=ffn_dataset,
    )
    with open(fig_data_path, "wb") as file:
        pickle.dump(results, file)

    if COLLECT_DIAGNOSTICS and "diagnostics" in results:
        diag_path = Path(fig_data_path).with_name(Path(fig_data_path).stem + "_diagnostics.pkl")
        with open(diag_path, "wb") as file:
            pickle.dump(results["diagnostics"], file)
        summary_path = Path(fig_data_path).with_name(Path(fig_data_path).stem + "_diagnostics_summary.csv")
        with open(summary_path, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([
                "snr_db", "method", "mean_rate", "mean_fair_rate", "mean_sum_rate",
                "mean_served_commodities", "mean_total_power", "mean_off_route_fraction",
                "mean_selected_path_length", "frac_selected_paths_within_horizon",
                "mean_valid_lhop_candidates",
            ])
            for snr_key, methods in results.get("diagnostics_summary", {}).items():
                for method, vals in methods.items():
                    writer.writerow([
                        snr_key, method, vals.get("mean_rate", 0.0), vals.get("mean_fair_rate", 0.0),
                        vals.get("mean_sum_rate", 0.0), vals.get("mean_served_commodities", 0.0),
                        vals.get("mean_total_power", 0.0), vals.get("mean_off_route_fraction", 0.0),
                        vals.get("mean_selected_path_length", 0.0),
                        vals.get("frac_selected_paths_within_horizon", 0.0),
                        vals.get("mean_valid_lhop_candidates", 0.0),
                    ])
        print(f"Diagnostics saved at: {diag_path}")
        print(f"Diagnostics summary saved at: {summary_path}")

    plot_mean_rate_vs_snr(snr_db_list, results, save_path=fig_path)
    print(f'Fig saved at: {fig_path}')
    print('===================================================')
    print('DONE')
    print('===================================================')



