import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

#Grid-Based Sampling Section

#Open desired .csv file based on user input, and also change headers
original_file = input("Enter the file path for the original data set (ex: 'data_set_1.csv'): ")
data = pd.read_csv(original_file, header=None, names=['Index', 'X1_t', 'X2_t'])


num_bins = 50
samples_per_bin = 50
#limit to 50 samples per bin 

#create bins for X1_t and X2_t
x1_bins = np.linspace(data['X1_t'].min(), data['X1_t'].max(), num_bins + 1)
x2_bins = np.linspace(data['X2_t'].min(), data['X2_t'].max(), num_bins + 1)

#assign each data point in X1_t and X2_t to a certain grid-cell based on its value
data['x1_bin'] = np.digitize(data['X1_t'], x1_bins) - 1
data['x2_bin'] = np.digitize(data['X2_t'], x2_bins) - 1

#initialize empty dataframe to store grid data
grid_sampled_data = pd.DataFrame(columns=data.columns)

#loop that samples through each bin
for x1_bin in range(num_bins):
    for x2_bin in range(num_bins):
        #filter our data to get points from specific bin
        bin_data = data[(data['x1_bin'] == x1_bin) & (data['x2_bin'] == x2_bin)]
        
        #if the bin has 50 or more points, sample exactly 50
        if len(bin_data) >= samples_per_bin:
            sample = bin_data.sample(n=samples_per_bin, random_state=42)
            grid_sampled_data = pd.concat([grid_sampled_data, sample])
        else:
            #if the bin has fewer than 50 points, skip it
            continue

#incase we have more than 2500 samples, this trims our data
if len(grid_sampled_data) > 2500:
    grid_sampled_data = grid_sampled_data.sample(n=2500, random_state=42)
#incase we have less than 2500 samples, it adds more
elif len(grid_sampled_data) < 2500:
    additional_needed = 2500 - len(grid_sampled_data)
    additional_samples = data.sample(n=additional_needed, random_state=42)
    grid_sampled_data = pd.concat([grid_sampled_data, additional_samples]).reset_index(drop=True)

#saves our grid bases sampled data into a new .csv file
grid_sampled_data.to_csv('grid_sampled_2500.csv', index=False)

print(f"Number of points in grid-based sample: {len(grid_sampled_data)}")

###########################################################################################################################

#Density-Based Sampling Section

#launch our original data, to avoid grid based samples
data = pd.read_csv(original_file, header=None, names=['Index', 'X1_t', 'X2_t'])

#defines lower and upper 0.1% of values for each column
x1_min, x1_max = data['X1_t'].quantile([0.001, 0.999])  
x2_min, x2_max = data['X2_t'].quantile([0.001, 0.999]) 

#filters data to include only rows where X1_t and X2_t values fall within the range set by their corresponding x1_min, x1_max and x2_min, x2_man
filtered_density_data = data[(data['X1_t'] >= x1_min) & (data['X1_t'] <= x1_max) &
                             (data['X2_t'] >= x2_min) & (data['X2_t'] <= x2_max)]
                    
#incase the code has more than 10,000 rows, the code below will try to reduce the dataset down to 10,000 rows
if len(filtered_density_data) > 10000:
    filtered_density_data = filtered_density_data.sample(n=10000, random_state=42)

#groups the data into 2,500 clusters based on X1_t and X2_t values, adding each data point's cluster number as a new column
num_clusters = 2500
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
filtered_density_data['cluster'] = kmeans.fit_predict(filtered_density_data[['X1_t', 'X2_t']])

#samples one point from each cluster, to ensure a balanced dataset
density_sampled_data = filtered_density_data.groupby('cluster').apply(lambda x: x.sample(1)).reset_index(drop=True)

#incase we have less than 2500 samples, it adds more
if len(density_sampled_data) < 2500:
    additional_density_samples = filtered_density_data.sample(n=2500 - len(density_sampled_data), random_state=42)
    density_sampled_data = pd.concat([density_sampled_data, additional_density_samples]).reset_index(drop=True)

#saves our density based data to a new .csv file
density_sampled_data.to_csv('density_sampled_2500.csv', index=False)


###########################################################################################################################

#Interpolated Data Section

from scipy.interpolate import Rbf
import numpy as np
import pandas as pd

grid_file = 'grid_sampled_2500.csv'
density_file = 'density_sampled_2500.csv'

#load grid-based and density-based samples
grid_data = pd.read_csv(grid_file)
density_data = pd.read_csv(density_file)

#make sure grid data and density data have the same length, to avoid error or mismatches
assert len(grid_data) == len(density_data)

#blends grid_data and density beta based on the factor alpha, which allows for a gradual adjustment betwen both datasets
def rbf_interpolate_samples(grid_data, density_data, alpha):

    #ensures alpha is within [0, 1]
    alpha = np.clip(alpha, 0, 1)
    
    #initializes 2 linear interpolation functions that map values from our grid and density data
    rbf_x1 = Rbf(grid_data['X1_t'], density_data['X1_t'], function='linear')
    rbf_x2 = Rbf(grid_data['X2_t'], density_data['X2_t'], function='linear')
    
    #generates interpolated data based on alpha
    interpolated_data = grid_data.copy()
    interpolated_data['X1_t'] = (1 - alpha) * grid_data['X1_t'] + alpha * rbf_x1(grid_data['X1_t'])
    interpolated_data['X2_t'] = (1 - alpha) * grid_data['X2_t'] + alpha * rbf_x2(grid_data['X2_t'])
    
    return interpolated_data

#prompts a user input for the alpha value
try:
    alpha = float(input("Enter a value for alpha between 0 and 1 (0 = grid-based, 1 = density-based): "))
#incase value is invalid, set to 0.5 by default
except ValueError:
    print("Invalid input. Setting alpha to 0.5 by default.")
    alpha = 0.5

#generates our interpolated data based on alpha value
interpolated_data = rbf_interpolate_samples(grid_data, density_data, alpha=alpha)

#save the interpolated data to a new .csv file
output_file = f'interpolated_sample_alpha_{alpha}.csv'
interpolated_data.to_csv(output_file, index=False)
print(f"Interpolated sample saved to {output_file}")

###########################################################################################################################

import matplotlib.pyplot as plt

#visualize 2x2 figure with 4 different datasets (grid based, density based, interpolated, and original)
fig, axs = plt.subplots(2, 2, figsize=(14, 10))

#plot 1: Grid-Based Data
axs[0, 0].scatter(grid_data['X1_t'], grid_data['X2_t'], color='blue', alpha=0.6, s=10)
axs[0, 0].set_title('Grid-Based Data')
axs[0, 0].set_xlabel('X1_t (Frequency)')
axs[0, 0].set_ylabel('X2_t (Power)')
axs[0, 0].grid(True)

#plot 2: Density-Based Data
axs[0, 1].scatter(density_data['X1_t'], density_data['X2_t'], color='green', alpha=0.6, s=10)
axs[0, 1].set_title('Density-Based Data')
axs[0, 1].set_xlabel('X1_t (Frequency)')
axs[0, 1].set_ylabel('X2_t (Power)')
axs[0, 1].grid(True)

#plot 3: Interpolated Data
axs[1, 0].scatter(interpolated_data['X1_t'], interpolated_data['X2_t'], color='red', alpha=0.6, s=10)
axs[1, 0].set_title(f'Interpolated Data (alpha={alpha})')
axs[1, 0].set_xlabel('X1_t (Frequency)')
axs[1, 0].set_ylabel('X2_t (Power)')
axs[1, 0].grid(True)

#plot 4: Original Data
axs[1, 1].scatter(data['X1_t'], data['X2_t'], color='purple', alpha=0.1, s=5)
axs[1, 1].set_title('Original Data')
axs[1, 1].set_xlabel('X1_t (Frequency)')
axs[1, 1].set_ylabel('X2_t (Power)')
axs[1, 1].grid(True)

#adjust layout and display the plot
plt.tight_layout()
plt.show()

###########################################################################################################################


#set the number of bins for the histograms
bins = 50

#visualize 2x2 figure with 4 different histograms (grid based, density based, interpolated, and original)
fig, axs = plt.subplots(2, 2, figsize=(14, 10))

#plot histograms for X1_t

#original data - X1_t
axs[1, 1].hist(data['X1_t'], bins=bins, density=True, alpha=0.5, color='purple', label='Original Data (Normalized)')
axs[1, 1].set_title('Original Data (X1_t)')
axs[1, 1].set_xlabel('X1_t')
axs[1, 1].set_ylabel('Frequency Density')
axs[1, 1].legend()

#grid-based data - X1_t
axs[0, 0].hist(grid_sampled_data['X1_t'], bins=bins, density=True, alpha=0.5, color='blue', label='Grid-Based Data')
axs[0, 0].set_title('Grid-Based Data (X1_t)')
axs[0, 0].set_xlabel('X1_t')
axs[0, 0].set_ylabel('Frequency Density')
axs[0, 0].legend()

#density-based data - X1_t
axs[0, 1].hist(density_sampled_data['X1_t'], bins=bins, density=True, alpha=0.5, color='green', label='Density-Based Data')
axs[0, 1].set_title('Density-Based Data (X1_t)')
axs[0, 1].set_xlabel('X1_t')
axs[0, 1].set_ylabel('Frequency Density')
axs[0, 1].legend()

#interpolated data - X1_t
axs[1, 0].hist(interpolated_data['X1_t'], bins=bins, density=True, alpha=0.5, color='red', label=f'Interpolated Data (alpha={alpha})')
axs[1, 0].set_title(f'Interpolated Data (X1_t) - alpha={alpha}')
axs[1, 0].set_xlabel('X1_t')
axs[1, 0].set_ylabel('Frequency Density')
axs[1, 0].legend()

#ddjust layout and display the plot
plt.tight_layout()
plt.show()
