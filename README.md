# Federated Clustering Algorithms

This repository contains implementations of various clustering algorithms with a focus on federated learning scenarios. The project includes implementations of k-means, k-median, and k-FED (Federated Expectation Maximization) algorithms.

## Project Structure

```
Essay/
├── clustering.py          # Main clustering algorithms and utilities
├── kFED.py               # Federated Expectation Maximization implementation
├── kmedian.py            # k-median clustering algorithm
├── read_data.py          # Data loading utilities
├── utils.py              # Helper functions and visualization tools
├── start.ipynb           # Jupyter notebook with examples and experiments
├── requirements.txt      # Python dependencies
├── data/                 # Dataset directory
│   └── s-originals/     # S-datasets for clustering experiments
└── TODO                  # Project roadmap and future work
```

## Features

### Implemented Algorithms

1. **K-Means Clustering** - Standard k-means implementation with sklearn integration
2. **K-Median Clustering** - Robust clustering using median instead of mean
3. **K-FED (Federated Expectation Maximization)** - Federated learning approach for clustering
4. **Data Partitioning** - Utilities for simulating federated learning scenarios

### Key Functionalities

- **Random Data Partitioning**: Split datasets across multiple clients for federated learning simulations
- **Clustering Evaluation**: Comprehensive metrics for cluster quality assessment
- **Visualization Tools**: 2D plotting utilities for cluster visualization
- **Dataset Loading**: Support for various clustering datasets including S-datasets

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Essay
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dependencies

- numpy
- pandas
- scikit-learn
- notebook
- matplotlib
- plotly

## Usage

### Basic Clustering Example

```python
import numpy as np
from clustering import kmeans
from read_data import read_s_dataset

# Load dataset
X, y = read_s_dataset(2)

# Perform k-means clustering
centers, labels = kmeans(X, n_clusters=15)

# Evaluate clustering results
from utils import evaluation_summary
print(evaluation_summary(X, centers, y))
```

### Federated Learning Simulation

```python
from clustering import random_data_partition
from kFED import kfed

# Partition data across clients
n_clients = 5
X_clients, y_clients = random_data_partition(X, y, n_clients)

# Run federated clustering
centers = kfed(X_clients, n_clusters=15, max_iter=100)
```

### Jupyter Notebook Examples

Open `start.ipynb` for comprehensive examples and visualizations:

```bash
jupyter notebook start.ipynb
```

## Datasets

The project includes S-datasets (s1, s2, s3, s4) for clustering experiments. Each dataset contains:
- Original data files (`.txt`)
- Cluster boundary files (`.cb`, `-cb.txt`)
- Label files (`.pa`)

## Project Status

### Completed
- ✅ Basic k-means and k-median implementations
- ✅ K-FED algorithm for federated clustering
- ✅ Data partitioning utilities
- ✅ Visualization tools
- ✅ S-dataset integration

### In Progress / TODO
- 🔄 Implement packet loss simulation
- 🔄 Implement clustering matching algorithms
- 🔄 Implement trimmed k-means
- 🔄 Load more datasets (MNIST + PCA)
- 🔄 Performance optimization
- 🔄 Additional evaluation metrics

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- This project is part of research on federated learning and clustering algorithms
- Built with Python and popular scientific computing libraries
- Inspired by federated learning research in distributed machine learning