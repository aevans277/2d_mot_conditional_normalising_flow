import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2
import torch
import torch.nn as nn


##### this is used to test changes with the final plot of the CNF training process

# LaTeX style setup for matplotlib
plt.rc('font', family='serif', serif=['Computer Modern'])
plt.rc('text', usetex=True)

# Dummy model class with a sample method
class DummyModel(nn.Module):
    def __init__(self):
        super(DummyModel, self).__init__()

    def sample(self, n_samples, context):
        return torch.randn(n_samples, 5), None

# Dummy test data and condition
n_samples = 1000
test_data = torch.randn(n_samples, 5)
test_cond = torch.randn(n_samples, 5)
model = DummyModel()

def compute_chi_square(true_counts, gen_counts, errors_true, errors_gen):
    chi2_stat = np.sum((true_counts - gen_counts) ** 2 / (errors_true ** 2 + errors_gen ** 2))
    return chi2_stat

def plot_trained_distribution(test_data, test_cond, model):
    model.eval()
    with torch.no_grad():
        # Sample generated data from the model
        generated_test_data, _ = model.sample(test_data.shape[0], context=test_cond)
        generated_test_data = generated_test_data.cpu().numpy()

    assert generated_test_data.shape == test_data.shape, "Shapes of generated data and test data do not match!"

    fig, axs = plt.subplots(2, 5, figsize=(36, 7), sharex='col', gridspec_kw={'height_ratios': [3, 1]})

    def plot_ratio(ax, original_data, generated_data, bins, xlabel):
        epsilon = 1e-10  # Small value to avoid division by zero
        hist_original, _ = np.histogram(original_data, bins=bins, density=False)
        hist_generated, _ = np.histogram(generated_data, bins=bins, density=False)
        hist_original = hist_original + epsilon  # Add epsilon to avoid division by zero
        hist_generated = hist_generated + epsilon  # Add epsilon to avoid division by zero
        ratios = np.divide(hist_generated, hist_original, out=np.zeros_like(hist_generated, dtype=float),
                           where=hist_original != 0)
        bin_edges = bins

        # Calculate the propagated uncertainties for the ratios
        errors_true = np.sqrt(hist_original)  # Add epsilon to avoid division by zero
        errors_gen = np.sqrt(hist_generated)  # Add epsilon to avoid division by zero
        ratio_uncertainties = ratios * np.sqrt((errors_true / hist_original) ** 2 + (errors_gen / hist_generated) ** 2)

        ax.step(bin_edges[:-1], ratios, where='post', linestyle='--', color='black')
        ax.fill_between(bin_edges[:-1], ratios - ratio_uncertainties, ratios + ratio_uncertainties, step='post', color='gray', alpha=0.3)
        ax.axhline(1, color='grey', linewidth=0.5)  # Line for ratio=1
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Ratio')
        ax.set_ylim(0, 2)  # Set a reasonable y-limit for better visibility
        ax.invert_yaxis()  # Invert the y-axis
        ax.set_yticks(ax.get_yticks()[1:])  # Remove the 0 at the top
        ax.spines['top'].set_visible(False)  # Hide the top spine

        return hist_original, hist_generated

    chi2_stats = []

    for i, (label, column) in enumerate(zip(
            ['X', 'Y', 'Vx', 'Vy', 'log(1 + Vz)'],
            range(5)
    )):
        original_data = test_data[:, column].cpu().numpy()
        generated_data = generated_test_data[:, column]

        assert generated_data.shape == original_data.shape, "Plotting data doesn't match!"

        # Determine a common range for both datasets
        data_min = min(original_data.min(), generated_data.min())
        data_max = max(original_data.max(), generated_data.max())
        bins = np.linspace(data_min, data_max, 51)  # 50 bins

        axs[0, i].hist(original_data, bins=bins, color='blue', histtype='step', label='Ground Truth', density=False)
        axs[0, i].hist(generated_data, bins=bins, color='red', histtype='step', label='CNF Generated', density=False)
        axs[0, i].set_title(f'Histogram of {label} (normalised)')
        axs[0, i].set_ylabel('Count')
        axs[0, i].legend()

        hist_original, hist_generated = plot_ratio(axs[1, i], original_data, generated_data, bins,
                                                   f'{label} (normalised)')

        # Calculate the chi^2 statistic using the provided function
        errors_true = np.sqrt(hist_original)  # Add epsilon to avoid division by zero
        errors_gen = np.sqrt(hist_generated)  # Add epsilon to avoid division by zero
        chi2_stat = compute_chi_square(hist_original, hist_generated, errors_true, errors_gen)
        chi2_stats.append(chi2_stat)

        # Number of degrees of freedom
        ndf = len(bins) - 1

        # Calculate the p-value
        p_value = chi2.sf(chi2_stat, ndf)

        axs[0, i].set_title(
            f'Histogram of {label} (normalised)\n$\\chi^2 = {chi2_stat:.2f} / {ndf} = {chi2_stat / ndf:.2f} \\; (p = {p_value:.2e})$')

    plt.tight_layout(pad=1.5, h_pad=-0.44, w_pad=2.0)
    plt.savefig('./TEST_CNF_transformed_distribution.png', dpi=600)
    plt.show()

# Plot using the dummy data and model
plot_trained_distribution(test_data, test_cond, model)