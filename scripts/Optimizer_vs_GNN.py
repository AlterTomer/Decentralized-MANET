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
from MANET_FFN.MANET_FFN_Dataset import FFNDataset
from MANET_FFN.model import FFNPowerAllocator

# ====== config ======
# args = parse_args()
# cfg_path = args.config.resolve()
# parser = load_ini_config(cfg_path)
# print(f"Loaded config from CLI: {cfg_path}")

cfg_arr = [# r"C:\Users\alter\OneDrive\Desktop\PhD\Decentralized MANET\Config Files\Single Tx-Rx\comp_unicast.ini",
           # r"C:\Users\alter\OneDrive\Desktop\PhD\Decentralized MANET\Config Files\Multicommodity\comp_multicommodity.ini",
           # r"C:\Users\alter\OneDrive\Desktop\PhD\Decentralized MANET\Config Files\Multiunicast\comp_multiunicast.ini",
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
    # Files Parameters
    files_params = parser["Files"]
    try:
        channel_path = files_params["channel path"]
    except KeyError:
        channel_path = None
    fig_path = files_params["fig path"]
    ffn_path = files_params["ffn path"]
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
    adj_list = []
    links_list = []
    for d in dataset:
        adj_list.append(d.adj_matrix)
        links_list.append(d.links_matrix)

    ffn_dataset = FFNDataset(
        adj_list=adj_list,
        links_list=links_list,
        tx_list=tx_list,
        rx_list=rx_list,
        sigma_list=sigma_list,
        B=B,
        problem=MODE,
        K=K_cfg,
    )

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
        jk_mode="concat"
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
    # model.load_state_dict(ckpt["model_state_dict"])
    model.load_state_dict(new_state_dict, strict=False)

    # FFN Model
    file_path = Path(ffn_path) / "ffn_run_config.pt"
    ffn_cfg = torch.load(file_path, weights_only=False)
    ffn_model = FFNPowerAllocator(n_nodes=ffn_cfg['n'],
                                  n_bands=ffn_cfg['B'],
                                  K=ffn_cfg['K_cfg'],
                                  problem=MODE,
                                  hidden_dim=ffn_cfg['hidden_dim'],
                                  num_layers=ffn_cfg['num_layers'],
                                  dropout=ffn_cfg['dropout'],
                                  use_layernorm=True)
    ffn_weights = Path(ffn_path) / "ffn_checkpoint.pt"
    ffn_state_dict = torch.load(ffn_weights, weights_only=False)['model_state_dict']
    ffn_model.load_state_dict(ffn_state_dict)
    ffn_model.to(device).eval()


    g = torch.Generator().manual_seed(SEED)

    snr_db_list = list(range(0, 51, 2))
    results = evaluate_across_snr(dataset, model, ffn_dataset, ffn_model, B, snr_db_list,problem=MODE)
    with open(fig_data_path, "wb") as file:
        pickle.dump(results, file)



    plot_mean_rate_vs_snr(snr_db_list, results, save_path=fig_path)
    print(f'Fig saved at: {fig_path}')
    print('===================================================')
    print('DONE')
    print('===================================================')

