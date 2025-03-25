# Virality Prediction using LLM Embeddings and Graph-Based Models

## Overview
This project explores various machine learning (ML) and graph-based approaches to predict the virality score of textual content. Different feature extraction techniques, including LLM embeddings and TF-IDF, were combined with Graph Convolutional Networks (GCN), Graph Attention Networks (GAT), and traditional ML models to assess their effectiveness.

## Results

### Dataset 1
| Methodology | MSE (Virality Score) | RMSE |
|------------|---------------|------|
| LLM Embeddings + Additional Features + GCN | 3267 | 57.16 |
| GAT | 3382 | 58.15 |
| LLM Embeddings + Additional Features + Random Forest Regressor | 3437.0377 | 58.63 |
| TF-IDF Vectorizer + Additional Features + ML Model | 4051.6476 | 63.65 |
| TF-IDF Vectorizer + GNN | 3345.0659 | 57.84 |
| TF-IDF Vectorizer + GAT | 3407 | 58.37 |
| ML Models (XGBoost) | 3348.1619 | 57.86 |

### Dataset 2
| Methodology | MSE (Virality Score) | RMSE |
|------------|---------------|------|
| LLM Embeddings + GCN | 3432 | 58.58 |
| LLM Embeddings + GAT | 3577 | 59.81 |
| LLM Embeddings + ML Model | 14906.2740 | 122.09 |
| TF-IDF + ML Model | 11748.9201 | 108.39 |
| RAKE Features | 3414 | 58.43 |
| Word2Vec Features | 5543.564505 | 74.46 |
| LLM Embeddings + Autoencoder (Dimensionality Reduction) + GNN | 3165 | 56.26 |
| XGBoost Model | 23579.1480 | 153.56 |

### Dimensionality Reduction (D2)
| Method | Test Loss / MSE |
|--------|---------------|
| PCA | 8310845.5000 |
| SVM (Test MSE) | 3313.0532 |
| T-SNE | 3334.0610 |
| IsoMap | 12028.3340 |

## Conclusion
- **Graph-based models (GCN, GAT)** perform well with LLM embeddings.
- **ML models (Random Forest, XGBoost)** demonstrate varying performance based on feature selection.
- **Dimensionality reduction techniques** impact performance, with Autoencoder + GNN achieving the best score (3165) in D2.
- **TF-IDF and word embeddings** provide alternative approaches but may not always outperform deep learning methods.

## Future Work
- Experiment with hybrid models combining graph-based and ML techniques.
- Improve dimensionality reduction techniques for better efficiency.
- Explore other feature extraction methods for enhanced representation.

---
Due to computational limitations, we were able to work with only a subset of the data (up to 1000 samples) due to the memory-intensive nature of graph models. As a result, our accuracy is lower. However, as the size of the graph increases, the model's performance is expected to improve.

For any queries or improvements, feel free to reach out!
