import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load the CSV file with assigned column names
original_file = input("Enter the file path for the original data set (ex: 'data_set_1.csv'): ")
data = pd.read_csv(original_file, header=None, names=['Index', 'X1_t', 'X2_t'])

# Grid-based sampling section
# Define the number of bins and samples per bin
num_bins = 50
samples_per_bin = 50  # We want exactly 50 samples per bin

# Create bins for X1_t and X2_t
x1_bins = np.linspace(data['X1_t'].min(), data['X1_t'].max(), num_bins + 1)
x2_bins = np.linspace(data['X2_t'].min(), data['X2_t'].max(), num_bins + 1)

# Assign each data point to a grid cell based on X1_t and X2_t
data['x1_bin'] = np.digitize(data['X1_t'], x1_bins) - 1
data['x2_bin'] = np.digitize(data['X2_t'], x2_bins) - 1

# Initialize an empty DataFrame to store the uniformly sampled data
grid_sampled_data = pd.DataFrame(columns=data.columns)

# Loop through each bin and sample exactly 50 points from each bin
for x1_bin in range(num_bins):
    for x2_bin in range(num_bins):
        # Filter data to get points within the current bin
        bin_data = data[(data['x1_bin'] == x1_bin) & (data['x2_bin'] == x2_bin)]
        
        # If the bin has 50 or more points, sample exactly 50
        if len(bin_data) >= samples_per_bin:
            sample = bin_data.sample(n=samples_per_bin, random_state=42)
            grid_sampled_data = pd.concat([grid_sampled_data, sample])
        else:
            # If the bin has fewer than 50 points, skip it
            continue

# Check if we have exactly 2,500 samples; if not, trim excess points or handle shortages
if len(grid_sampled_data) > 2500:
    grid_sampled_data = grid_sampled_data.sample(n=2500, random_state=42)
elif len(grid_sampled_data) < 2500:
    additional_needed = 2500 - len(grid_sampled_data)
    additional_samples = data.sample(n=additional_needed, random_state=42)
    grid_sampled_data = pd.concat([grid_sampled_data, additional_samples]).reset_index(drop=True)

# Save the grid-based sampled data to a new CSV file
grid_sampled_data.to_csv('grid_sampled_2500.csv', index=False)

print(f"Number of points in grid-based sample: {len(grid_sampled_data)}")




###########################################################################################################################

# Density-based sampling section
# Reload original data to avoid any modifications done during grid-based sampling
data = pd.read_csv(original_file, header=None, names=['Index', 'X1_t', 'X2_t'])

# Define acceptable ranges for filtering out outliers
x1_min, x1_max = data['X1_t'].quantile([0.001, 0.999])  # Retain the lower 0.1% and upper 0.1%
x2_min, x2_max = data['X2_t'].quantile([0.001, 0.999])  # Retain the lower 0.1% and upper 0.1%

# Filter data within the specified range
filtered_density_data = data[(data['X1_t'] >= x1_min) & (data['X1_t'] <= x1_max) &
                             (data['X2_t'] >= x2_min) & (data['X2_t'] <= x2_max)]
                    
# If filtered_data is still large, sample 10,000 points to prevent memory issues
if len(filtered_density_data) > 10000:
    filtered_density_data = filtered_density_data.sample(n=10000, random_state=42)

# Apply KMeans clustering with a larger number of clusters to ensure 2,500 samples
num_clusters = 2500  # Set to 2,500 to reach the desired sample size
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
filtered_density_data['cluster'] = kmeans.fit_predict(filtered_density_data[['X1_t', 'X2_t']])

# Sample one point from each cluster
density_sampled_data = filtered_density_data.groupby('cluster').apply(lambda x: x.sample(1)).reset_index(drop=True)

# If we still have fewer than 2,500 points, randomly sample additional points to reach 2,500
if len(density_sampled_data) < 2500:
    additional_density_samples = filtered_density_data.sample(n=2500 - len(density_sampled_data), random_state=42)
    density_sampled_data = pd.concat([density_sampled_data, additional_density_samples]).reset_index(drop=True)

# Save the density-based sampled data to a new CSV file
density_sampled_data.to_csv('density_sampled_2500.csv', index=False)


###########################################################################################################################

from scipy.interpolate import Rbf
import numpy as np
import pandas as pd

grid_file = 'grid_sampled_2500.csv'
density_file = 'density_sampled_2500.csv'

# Load grid-based and density-based samples
grid_data = pd.read_csv(grid_file)
density_data = pd.read_csv(density_file)

# Ensure the samples have the same length
assert len(grid_data) == len(density_data), "Grid and density datasets must have the same number of samples."

def rbf_interpolate_samples(grid_data, density_data, alpha):
    """
    Interpolates between grid-based and density-based samples using Radial Basis Function (RBF) based on alpha.
    
    Parameters:
    - grid_data: DataFrame with grid-based samples.
    - density_data: DataFrame with density-based samples.
    - alpha: float in [0, 1] where 0 is purely grid-based, 1 is purely density-based.
    
    Returns:
    - interpolated_data: DataFrame with interpolated samples.
    """
    # Ensure alpha is within [0, 1]
    alpha = np.clip(alpha, 0, 1)
    
    # Initialize the RBF interpolator for each dimension
    rbf_x1 = Rbf(grid_data['X1_t'], density_data['X1_t'], function='linear')
    rbf_x2 = Rbf(grid_data['X2_t'], density_data['X2_t'], function='linear')
    
    # Generate interpolated data based on alpha
    interpolated_data = grid_data.copy()
    interpolated_data['X1_t'] = (1 - alpha) * grid_data['X1_t'] + alpha * rbf_x1(grid_data['X1_t'])
    interpolated_data['X2_t'] = (1 - alpha) * grid_data['X2_t'] + alpha * rbf_x2(grid_data['X2_t'])
    
    return interpolated_data

# Prompt the user for the alpha value
try:
    alpha = float(input("Enter a value for alpha between 0 and 1 (0 = grid-based, 1 = density-based): "))
except ValueError:
    print("Invalid input. Setting alpha to 0.5 by default.")
    alpha = 0.5

# Generate the interpolated sample based on the user-specified alpha
interpolated_data = rbf_interpolate_samples(grid_data, density_data, alpha=alpha)

# Save the interpolated data to a new CSV file
output_file = f'interpolated_sample_alpha_{alpha}.csv'
interpolated_data.to_csv(output_file, index=False)
print(f"Interpolated sample saved to {output_file}")

###########################################################################################################################

import matplotlib.pyplot as plt

# Visualize four different datasets in one window (2x2 grid layout)
fig, axs = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Grid-Based Data
axs[0, 0].scatter(grid_data['X1_t'], grid_data['X2_t'], color='blue', alpha=0.6, s=10)
axs[0, 0].set_title('Grid-Based Data')
axs[0, 0].set_xlabel('X1_t (Frequency)')
axs[0, 0].set_ylabel('X2_t (Power)')
axs[0, 0].grid(True)

# Plot 2: Density-Based Data
axs[0, 1].scatter(density_data['X1_t'], density_data['X2_t'], color='green', alpha=0.6, s=10)
axs[0, 1].set_title('Density-Based Data')
axs[0, 1].set_xlabel('X1_t (Frequency)')
axs[0, 1].set_ylabel('X2_t (Power)')
axs[0, 1].grid(True)

# Plot 3: Interpolated Data
axs[1, 0].scatter(interpolated_data['X1_t'], interpolated_data['X2_t'], color='red', alpha=0.6, s=10)
axs[1, 0].set_title(f'Interpolated Data (alpha={alpha})')
axs[1, 0].set_xlabel('X1_t (Frequency)')
axs[1, 0].set_ylabel('X2_t (Power)')
axs[1, 0].grid(True)

# Plot 4: Original Data
axs[1, 1].scatter(data['X1_t'], data['X2_t'], color='purple', alpha=0.1, s=5)
axs[1, 1].set_title('Original Data')
axs[1, 1].set_xlabel('X1_t (Frequency)')
axs[1, 1].set_ylabel('X2_t (Power)')
axs[1, 1].grid(True)

# Adjust layout and display the plot
plt.tight_layout()
plt.show()

###########################################################################################################################


# Set the number of bins for the histograms
bins = 50

# Create a figure with a 2x2 layout for the four histograms
fig, axs = plt.subplots(2, 2, figsize=(14, 10))

# Plot histograms for X1_t

# Original Data - X1_t
axs[1, 1].hist(data['X1_t'], bins=bins, density=True, alpha=0.5, color='purple', label='Original Data (Normalized)')
axs[1, 1].set_title('Original Data (X1_t)')
axs[1, 1].set_xlabel('X1_t')
axs[1, 1].set_ylabel('Frequency Density')
axs[1, 1].legend()

# Grid-Based Data - X1_t
axs[0, 0].hist(grid_sampled_data['X1_t'], bins=bins, density=True, alpha=0.5, color='blue', label='Grid-Based Data')
axs[0, 0].set_title('Grid-Based Data (X1_t)')
axs[0, 0].set_xlabel('X1_t')
axs[0, 0].set_ylabel('Frequency Density')
axs[0, 0].legend()

# Density-Based Data - X1_t
axs[0, 1].hist(density_sampled_data['X1_t'], bins=bins, density=True, alpha=0.5, color='green', label='Density-Based Data')
axs[0, 1].set_title('Density-Based Data (X1_t)')
axs[0, 1].set_xlabel('X1_t')
axs[0, 1].set_ylabel('Frequency Density')
axs[0, 1].legend()

# Interpolated Data - X1_t
axs[1, 0].hist(interpolated_data['X1_t'], bins=bins, density=True, alpha=0.5, color='red', label=f'Interpolated Data (alpha={alpha})')
axs[1, 0].set_title(f'Interpolated Data (X1_t) - alpha={alpha}')
axs[1, 0].set_xlabel('X1_t')
axs[1, 0].set_ylabel('Frequency Density')
axs[1, 0].legend()

# Adjust layout and display the plot
plt.tight_layout()
plt.show()
