# Optimizing a Condition Monitoring System for Electric Motors

## Project Overview
This project focused on optimizing a **condition monitoring system** to track the health of electric motors and their driven loads. The objective was to develop a **data selection algorithm** that reduces a large dataset (500,000 data points) to a manageable subset of 2,500 representative data points, enabling efficient training of a predictive model. This model uses **frequency**, **power**, and **vibration levels** to monitor motor health under real-world resource constraints.

### **Key Features**
- Implementation of **grid-based** and **density-based sampling** methods.
- Flexible interpolation using a user-controlled parameter, **alpha**, for blending grid and density sampling.
- Visualizations, including **scatter plots** and **histograms**, to compare sampling methods.
- Normalized visualizations for fair comparison across datasets.

---

## Inspiration
Condition monitoring is critical in industries such as **manufacturing**, **oil and gas**, and **energy**, where unexpected motor failures can result in costly downtime and safety concerns. By applying **machine learning** to historical data, we aim to detect early warning signs of motor health deterioration and enhance **productivity** and **safety**.

---

## Sampling Methods

### **Grid-Based Sampling**
Designed for **uniform data distribution** across the operational range: 
- Data is divided into bins (max 50 bins), with up to 50 data points per bin.
- Ensures balanced training datasets by accounting for underrepresented regions.
- Challenges: Determining the optimal number of bins and points per bin while minimizing computational time.
- **Example**: Visualized in histograms, this method compensates for low-value clusters, producing a balanced distribution.

### **Density-Based Sampling**
Prioritizes **high-density regions** where the motor frequently operates: 
- Assigns higher selection probabilities to clustered data points.
- Improves model accuracy by focusing on critical operational areas.
- Still includes low-density regions for comprehensive coverage and removes outliers to avoid skewed results.
- **Example**: Histogram comparisons highlight its similarity to the original dataset while focusing on dense areas.

### **Interpolation**
Allows users to blend between **grid-based** and **density-based** sampling using the **alpha** parameter (range: 0 to 1):
- **Linear Interpolation**: Connects data points with straight lines, but struggles with non-linear patterns.
- **Radial Basis Function (RBF) Interpolation**: Provides smooth, non-linear approximations, better capturing complex data trends.
- Users can customize their approach based on the desired distribution balance.

---

## Visualization
The visualization module generates **scatter plots** and **histograms** for comparison:
1. **Scatter Plots**:
   - **Grid-Based Data** (blue): Uniform spread.
   - **Density-Based Data** (green): Clusters in high-density areas.
   - **Interpolated Data** (red): A blend of grid and density distributions.
   - **Original Data** (purple): Reference for comparison.

2. **Histograms**:
   - Show frequency density of the `X1_t` variable across datasets.
   - Highlight differences in sampling methods and their effects on data distribution.
   
**Normalization** ensures fair comparisons by bounding data ranges based on frequency density.

---

## Conclusion
This project demonstrates how **targeted data selection algorithms** improve predictive accuracy in condition monitoring systems, even under tight computational constraints. We extend our gratitude to **Baker Hughes** for providing the data and inspiration for this challenge.

---

## Elevator Pitch
The challenge aimed to optimize a **condition monitoring system** for electric motors, selecting 2,500 representative data points from 500,000 while maintaining predictive accuracy. Using **grid-based** and **density-based sampling**, we enabled users to balance uniform and cluster-focused sampling. Visual comparisons (scatter plots, histograms) illustrate how each method retains operational insights while ensuring efficient training. Our approach significantly enhances real-world applicability for machine learning in motor health monitoring.

---

## Technologies Used
- **Python** for data processing and visualization
- **Matplotlib** for scatter plots and histograms
- **Scikit-learn** for density clustering
- **NumPy** for data manipulation
- **Radial Basis Function Interpolation** for smooth data blending

---

