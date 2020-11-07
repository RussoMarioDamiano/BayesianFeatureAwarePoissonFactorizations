# BayesianFeatureAwarePoissonFactorizations
 
This repository implements a feature-aware extension of the Bayesian parametric and nonparametric Poisson factorization models for recommender systems illustrated by Gopalan et al. (2013, 2015). This project is part of my graduation thesis at Bocconi University in Data Science.

The idea is to incorporate an observed vector of numerical item features to make probabilistic inference on user-item interaction around those features. In the included paper, that derives and describes the model in detail, price is used as a feature to derive the latent features of price sensitivity (for users) and perceived price (for items), but other numerical features such as average rating of an item can be used with no changes to the model.

It is ultimately shown that the feature-aware version of the Hierarchical parametric and Bayesian nonparametric models regularly outperform their vanilla counterparts, while also adding a significant layer of interpretability.