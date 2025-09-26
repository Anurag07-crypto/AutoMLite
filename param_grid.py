# For Decision Tree Classifier
dt_param_grid = {
    "criterion": ["gini", "entropy", "log_loss"],  # measure of quality of split
    "max_depth": [None, 5, 10, 15, 20, 25],        # maximum depth of the tree
    "min_samples_split": [2, 5, 10, 20],           # min samples to split a node
    "min_samples_leaf": [1, 2, 4, 8],              # min samples in a leaf node
    "max_features": [None, "sqrt", "log2"],        # features to consider for split
    "splitter": ["best", "random"]                 # strategy used to choose split
}

# For Decision Tree Regressor
dt_reg_param_grid = {
    "criterion": ["squared_error", "friedman_mse", "absolute_error", "poisson"],
    "max_depth": [None, 5, 10, 15, 20, 25],
    "min_samples_split": [2, 5, 10, 20],
    "min_samples_leaf": [1, 2, 4, 8],
    "max_features": [None, "sqrt", "log2"]
}

# For KNeighborsClassifier
knn_param_grid = {
    "n_neighbors": [3, 5, 7, 9, 11, 15],         # number of neighbors
    "weights": ["uniform", "distance"],          # uniform = equal, distance = closer points weigh more
    "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],  # nearest neighbor search method
    "p": [1, 2]                                  # 1 = Manhattan, 2 = Euclidean
}

# For KNeighborsRegressor
knn_reg_param_grid = {
    "n_neighbors": [3, 5, 7, 9, 11, 15],
    "weights": ["uniform", "distance"],
    "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
    "p": [1, 2]
}

# For SVM Classifier
svm_param_grid = {
    "C": [0.1, 1, 10, 100],               # regularization strength
    "kernel": ["linear", "poly", "rbf", "sigmoid"],  # kernel type
    "degree": [2, 3, 4],                  # only for 'poly' kernel
    "gamma": ["scale", "auto"],           # kernel coefficient
    "coef0": [0.0, 0.1, 0.5]              # only for 'poly' and 'sigmoid'
}

# For SVM Regressor
svr_param_grid = {
    "C": [0.1, 1, 10, 100],
    "kernel": ["linear", "poly", "rbf", "sigmoid"],
    "degree": [2, 3, 4],
    "gamma": ["scale", "auto"],
    "epsilon": [0.01, 0.1, 0.5, 1.0],     # epsilon-tube within which no penalty is given
    "coef0": [0.0, 0.1, 0.5]
}

# Random Forest Classifier
rf_param_grid = {
    "n_estimators": [50, 100, 200, 300],        # number of trees
    "criterion": ["gini", "entropy", "log_loss"],  # quality of split
    "max_depth": [None, 5, 10, 20, 30],        # max depth of each tree
    "min_samples_split": [2, 5, 10],           # min samples required to split
    "min_samples_leaf": [1, 2, 4],             # min samples required at leaf
    "max_features": ["auto", "sqrt", "log2"],  # number of features to consider
    "bootstrap": [True, False]                 # whether to use bootstrap samples
}

# Random Forest Regressor
rf_reg_param_grid = {
    "n_estimators": [50, 100, 200, 300],
    "criterion": ["squared_error", "absolute_error", "friedman_mse", "poisson"],
    "max_depth": [None, 5, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["auto", "sqrt", "log2"],
    "bootstrap": [True, False]
}
