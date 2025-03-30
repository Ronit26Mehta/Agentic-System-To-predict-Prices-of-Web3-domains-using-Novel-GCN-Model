
---

# Agentic System for Web3 Domain Price Prediction using a Novel Domain-Aware GCN Model

A state-of-the-art system designed to predict the prices of Web3 domains by leveraging a novel Domain-Aware Graph Convolutional Network (DA-GCN). This project integrates data collection automation, advanced graph-based deep learning, and real-world marketplace data to deliver precise domain valuation insights.

---

## Table of Contents

- [Overview](#overview)
- [Motivation](#motivation)
- [Architecture & Design](#architecture--design)
  - [Data Collection & Preprocessing](#data-collection--preprocessing)
  - [Domain-Aware Graph Convolutional Network (DA-GCN)](#domain-aware-graph-convolutional-network-da-gcn)
  - [Graph Construction](#graph-construction)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
  - [Data Fetching](#data-fetching)
  - [Model Training & Evaluation](#model-training--evaluation)
- [Experiments & Results](#experiments--results)
- [Paper & Technical Details](#paper--technical-details)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)
- [Contact](#contact)

---

## Overview

This repository contains an **Agentic System** that predicts the prices of Web3 domains using a novel graph convolutional approach. The system combines automated data fetching, graph signal processing, and deep learning to learn non-linear relationships in domain price determinants such as user engagement, domain features, and market dynamics.

Key features:
- **Novel DA-GCN Architecture:** Integrates both structural (k-NN) and semantic (feature similarity) components.
- **Automated Data Collection:** Uses an autonomous agent to gather and preprocess marketplace data.
- **Research-Backed Methodology:** Detailed in our accompanying research paper.
- **Visualization & Reporting:** Outputs interactive graphs and detailed prediction reports.

---

## Motivation

The exponential growth of the Web3 ecosystem has led to the emergence of domain names with unique values and usage contexts. Traditional pricing models fail to capture the intricacies of domain valuation in this new digital landscape. Our project aims to:

- **Bridge the Gap:** Combine graph-based representations with domain-specific features.
- **Improve Accuracy:** Use a log-transformation of features and a convex combination of graph construction methods to enhance prediction reliability.
- **Facilitate Decision-Making:** Empower investors and brokers with predictive insights derived from marketplace data.

---

## Architecture & Design

### Data Collection & Preprocessing

The system employs an autonomous agent (found in the `agent/` directory) which:
- **Fetches Domain Data:** Retrieves up-to-date listings and engagement metrics.
- **Cleans & Normalizes Data:** Applies necessary transformations (e.g., log-transformation) to prepare data for modeling.
- **Saves Data:** Stores processed data in CSV format within the `data/` folder.

### Domain-Aware Graph Convolutional Network (DA-GCN)

Our DA-GCN model is designed to integrate complex domain features using a two-layer graph convolutional network. Key components include:

- **Input Layer:** Accepts transformed features from domain listings.
- **Graph Convolution Layers:** Two layers that capture both local (structural) and global (feature-based) interactions among domains.
- **Output Layer:** Produces a final domain price prediction.

**Key Equations:**
- **Adjacency Matrix Construction:**
  - \( A_{\text{combined}} = \alpha \times A_{\text{structural}} + (1 - \alpha) \times A_{\text{semantic}} \)
- **Graph Convolution Operation:**
  - \( H^{(l+1)} = \sigma\left( \tilde{A} H^{(l)} W^{(l)} \right) \)

Where:
- \( \tilde{A} \) is the normalized, combined adjacency matrix.
- \( W^{(l)} \) are learnable weight matrices.
- \( \sigma \) represents a non-linear activation function.

### Graph Construction

The model leverages two complementary graph structures:
- **Structural Graph (k-NN):** Based on the k-nearest neighbors of domain embeddings.
- **Semantic Graph (Feature Similarity):** Captures inherent similarities between domain features.

By combining these, the system benefits from both explicit structural relationships and implicit semantic similarities.

---

## Project Structure

```plaintext
Agentic-System/
├── agent/               
│   └── fetcher.py       # Script to collect and preprocess domain data
├── data/                
│   └── listings_data.csv  # Marketplace domain listings and engagement metrics
├── docs/                
│   └── New GNN.pdf      # Research paper detailing the model, theory, and pseudo-code
├── model/               
│   ├── domain_aware_gcn_W1.npy  # Trained weight matrix for first convolutional layer
│   ├── domain_aware_gcn_W2.npy  # Trained weight matrix for second convolutional layer
│   └── model.py         # Implementation of the DA-GCN model
├── output/              
│   ├── graph.html       # Interactive graph visualization of the domain network
│   └── image.png        # Snapshot image of the model output
└── README.md            # This README file
```

---

## Installation & Setup

### Prerequisites

- **Python 3.8+**
- **pip** for dependency management

### Installation Steps

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/yourusername/Agentic-System.git
   cd Agentic-System
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   > *Note: The `requirements.txt` file should list packages such as `numpy`, `pandas`, `PyPDF2`, `networkx`, `matplotlib`, and deep learning libraries like `torch` or `tensorflow` if applicable.*

3. **Set Up Environment Variables (if any):**
   - Create a `.env` file if the project requires API keys or specific environment settings.

---

## Usage

### Data Fetching

Run the data collection agent to automatically fetch and preprocess domain data:

```bash
python agent/fetcher.py
```

This script will:
- Download the latest listings data.
- Normalize and log-transform features.
- Save the processed data to `data/listings_data.csv`.

### Model Training & Evaluation

To train or evaluate the DA-GCN model, run the model script:

```bash
python model/model.py
```

The script performs the following:
- Loads preprocessed data from the `data/` folder.
- Constructs the combined adjacency matrix.
- Trains the two-layer graph convolutional network.
- Saves the trained weights (`domain_aware_gcn_W1.npy` and `domain_aware_gcn_W2.npy`).
- Outputs predictions and generates visualization reports in the `output/` folder.

> **Tip:** Check the model hyperparameters and adjust the configuration settings within `model.py` as needed.

---

## Experiments & Results

Our experimental results demonstrate that the DA-GCN model:
- **Improves Prediction Accuracy:** By combining structural and semantic graph information, the model consistently outperforms baseline regression techniques.
- **Handles Non-Linear Relationships:** The log-transformation of features coupled with deep graph convolutions captures complex patterns in the domain pricing data.
- **Provides Insightful Visualizations:** Interactive network graphs help understand the relationships between domains and their impact on pricing.

For a detailed evaluation, please refer to the experimental section in the paper (`docs/New GNN.pdf`).

---

## Paper & Technical Details

The accompanying research paper, **"A Novel Domain-Aware Graph Convolutional Model for Domain Pricing Prediction,"** provides an in-depth discussion on:
- **Theoretical Foundations:** Underlying mathematical models and assumptions.
- **Model Architecture:** Detailed breakdown of each network layer and graph construction method.
- **Pseudo-code & Implementation:** Step-by-step explanation of the training and evaluation processes.
- **Experimental Setup:** Data splits, hyperparameter choices, and performance metrics.

You can view the full paper in the `docs/` folder.

---

## Future Work

Our vision for future enhancements includes:
- **Real-Time Data Integration:** Incorporate live blockchain events to update predictions in real time.
- **Enhanced Feature Engineering:** Experiment with transformer-based embeddings to better capture domain name semantics.
- **Multi-Chain Support:** Extend the model to support pricing predictions across various blockchain domain systems.
- **User Interface Improvements:** Develop a more intuitive dashboard for visualizing model predictions and network graphs.

---

## Contributing

We welcome contributions from the community! To contribute:
1. **Fork the Repository.**
2. **Create a Feature Branch:** Use a descriptive branch name (e.g., `feature/data-fetcher-enhancement`).
3. **Commit Your Changes:** Ensure your code follows the project’s style guidelines.
4. **Submit a Pull Request:** Describe your changes in detail.

For major changes, please open an issue first to discuss your ideas.

---

## Acknowledgments

We would like to thank:
- **The Web3 Community:** For providing valuable insights into domain valuation.
- **Open Source Contributors:** Whose libraries and tools made this project possible.
- **Research Advisors:** For guidance on the theoretical and experimental aspects of the project.

---

