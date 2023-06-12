from sklearn.feature_selection import SelectKBest
from scipy.stats import kendalltau

# Compute Kendall's rank correlation coefficient for each feature
kendall_scores, _ = zip(*[kendalltau(X_train[:, feature], y_train) for feature in range(X_train.shape[1])])

# Perform feature selection using SelectKBest with kendall_scores
k = 5  # Number of features to select
selector = SelectKBest(score_func=lambda X, y: kendall_scores, k=k)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

# Get the indices of the selected features
selected_feature_indices = selector.get_support(indices=True)

# Print the indices of the selected features
print("Selected feature indices:", selected_feature_indices)

# Continue with further analysis or modeling using the selected features
# ...
