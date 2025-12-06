from visualization.GraphingAux import plot_mean_rate_vs_snr, time_varying_model_compare_plot
import pickle

with open(r"C:\Users\alter\Desktop\PhD\Decentralized MANET\Figures Data\benchmark_ture_Rayleigh_n_10_B_6.pkl", 'rb') as f:
    results = pickle.load(f)
snr_db_list = list(range(-10, 12, 1))
plot_mean_rate_vs_snr(snr_db_list, results, save_path=r"C:\Users\alter\Desktop\PhD\Decentralized MANET\Figs\Benchmark_True_Rayleigh_n_10_B_6.png")
# time_varying_model_compare_plot(snr_db_list, results, 10, 8, save_path=r"C:\Users\alter\Desktop\PhD\Decentralized MANET\Figs\Scalability_Rayleigh_big_10 small_8.png")

