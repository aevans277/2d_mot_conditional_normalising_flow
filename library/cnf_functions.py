import torch
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.stats import entropy
from sklearn.datasets import fetch_california_housing
from scipy.stats import chi2


def load_and_prepare_data(params_file, output_file):
    params_df = pd.read_csv(params_file)
    df = pd.read_csv(output_file)

    # filter data based on conditions
    df = df[(df['X'] <= 0.01) & (df['X'] >= -0.01) & (df['Y'] <= 0.01) & (df['Y'] >= -0.01)]
    df['X'] = df['X'] * 1000  # convert to mm
    df['Y'] = df['Y'] * 1000
    df['Vz'] = np.log1p(df['Vz'])  # log transformation using log1p to handle small values

    df = df.merge(params_df, on='simulation')

    # identify and remove rows with NaNs and infinite values
    columns = ['X', 'Y', 'Vx', 'Vy', 'Vz']
    conditioning_columns = ['cooling_beam_detuning', 'cooling_beam_radius', 'cooling_beam_power_mw',
                            'push_beam_detuning', 'push_beam_radius', 'push_beam_power',
                            'push_beam_offset', 'quadrupole_gradient', 'vertical_bias_field']

    initial_row_count = len(df)

    for col in columns + conditioning_columns:
        nan_count = df[col].isna().sum()
        inf_count = np.isinf(df[col]).sum()
        if nan_count > 0 or inf_count > 0:
            print(f"Column '{col}': {nan_count} NaNs, {inf_count} infinite values found removed.")
            df = df.dropna(subset=[col])
            df = df[~np.isinf(df[col])]

    removed_row_count = initial_row_count - len(df)
    print(f"Total rows removed: {removed_row_count}")

    normalisation_params = {}

    # normalise data columns
    for col in columns:
        mean, std = df[col].mean(), df[col].std()
        df[col] = (df[col] - mean) / std
        normalisation_params[col] = (mean, std)

    # normalise conditioning columns
    for col in conditioning_columns:
        mean, std = df[col].mean(), df[col].std()
        df[col] = (df[col] - mean) / std
        normalisation_params[col] = (mean, std)

    data_to_learn = df[columns].values
    conditioning_data = df[conditioning_columns].values

    return data_to_learn, conditioning_data, normalisation_params

def generate_toy_data(n=1e5):
    n = int(n)

    # fetch and select random data
    data = fetch_california_housing()
    X = data.data
    y = data.target

    indices = np.random.choice(np.arange(X.shape[0]), size=n, replace=True)
    sampled_X = X[indices]
    sampled_y = y[indices]

    # generate additional synthetic features to have 9 conditioning variables
    additional_features = np.random.uniform(0, 1, (n, 1))
    conditioning_data = np.hstack([sampled_X, additional_features])

    # generate 5 output variables based on the conditioning variables
    X_output = np.clip(conditioning_data[:, 0] * 0.00001 + np.random.normal(0, 0.00001, n), -0.005, 0.005)
    Y_output = np.clip(conditioning_data[:, 1] * 0.00001 + np.random.normal(0, 0.00001, n), -0.005, 0.005)
    V_output = np.abs(conditioning_data[:, 2] * 0.1 + np.random.normal(0, 0.01, n))
    Theta_output = np.arctan2(conditioning_data[:, 3], conditioning_data[:, 4]) + np.random.normal(0, 0.1, n)
    Azimuth_output = np.arctan2(conditioning_data[:, 5], conditioning_data[:, 6]) + np.random.normal(0, 0.1, n)

    # stack the output data to form the data_to_learn array
    data_to_learn = np.vstack([X_output, Y_output, V_output, Theta_output, Azimuth_output]).T

    return data_to_learn, conditioning_data

def split_data(data_to_learn, conditioning_data):
    train_data, test_data, train_cond, test_cond = train_test_split(
        data_to_learn, conditioning_data, test_size=0.3, random_state=42)
    valid_data, test_data, valid_cond, test_cond = train_test_split(
        test_data, test_cond, test_size=0.5, random_state=42)

    return train_data, valid_data, test_data, train_cond, valid_cond, test_cond

def convert_to_tensors(device, train_data, valid_data, test_data, train_cond, valid_cond, test_cond):
    train_data = torch.tensor(train_data, dtype=torch.float32).to(device)
    train_cond = torch.tensor(train_cond, dtype=torch.float32).to(device)
    valid_data = torch.tensor(valid_data, dtype=torch.float32).to(device)
    valid_cond = torch.tensor(valid_cond, dtype=torch.float32).to(device)
    test_data = torch.tensor(test_data, dtype=torch.float32).to(device)
    test_cond = torch.tensor(test_cond, dtype=torch.float32).to(device)

    return train_data, valid_data, test_data, train_cond, valid_cond, test_cond

def plot_initial_distribution(data_to_learn):
    fig, axs = plt.subplots(1, 5, figsize=(36, 6))

    axs[0].hist(data_to_learn[:, 0], bins=50, color='blue', histtype='step')
    axs[0].set_title('Histogram of X')
    axs[0].set_xlabel('X (normalised)')
    axs[0].set_ylabel('Count')

    axs[1].hist(data_to_learn[:, 1], bins=50, color='blue', histtype='step')
    axs[1].set_title('Histogram of Y')
    axs[1].set_xlabel('Y (normalised)')
    axs[1].set_ylabel('Count')

    axs[2].hist(data_to_learn[:, 2], bins=50, color='blue', histtype='step')
    axs[2].set_title('Histogram of Vx')
    axs[2].set_xlabel('Vx (normalised)')
    axs[2].set_ylabel('Count')

    axs[3].hist(data_to_learn[:, 3], bins=50, color='blue', histtype='step')
    axs[3].set_title('Histogram of Vy')
    axs[3].set_xlabel('Vy (normalised)')
    axs[3].set_ylabel('Count')

    axs[4].hist(data_to_learn[:, 4], bins=50, color='blue', histtype='step')
    axs[4].set_title('Histogram of log(1 + Vz)')
    axs[4].set_xlabel('log(1 + Vz) (normalised)')
    axs[4].set_ylabel('Count')

    plt.tight_layout(pad=3.0)
    plt.show()

def plot_training_and_validation_loss(train_loss_hist, val_loss_hist, epochs):
    plt.figure(figsize=(12, 10))
    plt.plot(train_loss_hist, label='Training Loss')
    plt.plot(np.linspace(0, epochs, len(val_loss_hist)), val_loss_hist, label='Validation Loss', linestyle='--')
    plt.yscale('log')
    plt.xscale('log')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('./plots/CNF_loss.png')
    plt.show()

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
    plt.savefig('./plots/CNF_transformed_distribution.png', dpi=600)
    plt.show()

def plot_trained_distribution_backup(test_data, test_cond, model):
    model.eval()
    with torch.no_grad():
        # Sample generated data from the model
        generated_test_data, _ = model.sample(test_data.shape[0], context=test_cond)
        generated_test_data = generated_test_data.cpu().numpy()

    assert generated_test_data.shape == test_data.shape, "Shapes of generated data and test data do not match!"

    fig, axs = plt.subplots(2, 5, figsize=(36, 12))

    def plot_ratio(ax, original_data, generated_data, bins, xlabel):
        hist_original, _ = np.histogram(original_data, bins=bins, density=False)
        hist_generated, _ = np.histogram(generated_data, bins=bins, density=False)
        ratios = np.divide(hist_generated, hist_original, out=np.zeros_like(hist_generated, dtype=float),
                           where=hist_original != 0)
        bin_centres = (bins[:-1] + bins[1:]) / 2

        ax.step(bin_centres, ratios, where='mid', linestyle='--', color='black')
        ax.axhline(1, color='grey', linewidth=0.5)  # Line for ratio=1
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Ratio (Generated/Ground Truth)')
        ax.set_ylim(0, 2)  # Set a reasonable y-limit for better visibility

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
        #axs[0, i].set_xlabel(f'{label} (normalised)')
        axs[0, i].set_xticks([])  # Remove x-axis ticks
        axs[0, i].set_ylabel('Count')
        axs[0, i].legend()

        hist_original, hist_generated = plot_ratio(axs[1, i], original_data, generated_data, bins, f'{label} (normalised)')

        # Calculate the chi^2 statistic using the provided function
        errors_true = np.sqrt(hist_original + 1e-10)  # Add small epsilon to avoid division by zero
        errors_gen = np.sqrt(hist_generated + 1e-10)  # Add small epsilon to avoid division by zero
        chi2_stat = compute_chi_square(hist_original, hist_generated, errors_true, errors_gen)
        chi2_stats.append(chi2_stat)

        # Number of degrees of freedom
        ndf = len(bins) - 1

        # Calculate the p-value
        p_value = chi2.sf(chi2_stat, ndf)

        # Add chi^2 statistic, ndf, and p-value below the title of the histogram
        axs[0, i].set_title(f'Histogram of {label} (normalised)\n$\\chi^2 = {chi2_stat:.2e} / {ndf} = {chi2_stat / ndf:.2e} \\; (p = {p_value:.2e})$')

    plt.tight_layout(pad=3.0)
    plt.savefig('./plots/CNF_transformed_distribution.png')
    plt.show()

def evaluate_model(model, test_data, test_cond):
    model.eval()
    with torch.no_grad():
        # Log-Likelihood and Perplexity
        log_prob = model.log_prob(test_data, test_cond)
        avg_log_likelihood = log_prob.mean().item()
        nll = -avg_log_likelihood
        perplexity = torch.exp(torch.tensor(nll)).item()

        # Generate samples for visual inspection and computing model density
        num_samples = 1000
        samples, _ = model.sample(num_samples, test_cond[:num_samples])
        samples = samples.cpu().numpy()

        # Compute true and model densities for KL divergence
        true_density, _ = np.histogramdd(test_data.cpu().numpy(), bins=50, density=True)
        model_density, _ = np.histogramdd(samples, bins=50, density=True)

        # Avoid division by zero
        true_density += 1e-10
        model_density += 1e-10

        forward_kl = entropy(true_density.flatten(), model_density.flatten())

    print(f"Average Log-Likelihood: {avg_log_likelihood}")
    print(f"Negative Log-Likelihood (NLL): {nll}")
    print(f"Perplexity: {perplexity}")
    print(f"Forward KL Divergence: {forward_kl}")