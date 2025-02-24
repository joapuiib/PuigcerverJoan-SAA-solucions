import pandas as pd
import numpy as np

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

import statsmodels.api as sm

california = fetch_california_housing()
X = pd.DataFrame(california.data, columns=california.feature_names)
Y = california.target

X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size = 0.7)
print(X_train.shape)
print(X_test.shape)


def forward_selection(data, target, significance_level=0.05, max_features=None):
    initial_features = data.columns.tolist()
    best_features = []
    while len(initial_features)>0 and (max_features is None or len(best_features) < max_features):
        remaining_features = list(set(initial_features)-set(best_features))
        new_pval = pd.Series(index=remaining_features)
        for new_column in remaining_features:
            model = sm.OLS(target, sm.add_constant(data[best_features+[new_column]])).fit()
            new_pval[new_column] = model.pvalues[new_column]

        min_p_value = new_pval.min()
        if(min_p_value<significance_level):
            best_features.append(new_pval.idxmin())
        else:
            break
    return best_features


significance = 0.05
initial_features = X_train.columns.tolist()
selected_features_FS = forward_selection(X_train, y_train, significance)
removed_features = list(set(initial_features)-set(selected_features_FS))

print("Initial features: ", initial_features)
print()
print(f"## Forward selection (alpha = {significance})")
print("Selected features: ", selected_features_FS)
print("Removed features: ", removed_features)

def backward_elimination(data, target, significance_level=0.05):
    features = data.columns.tolist()
    while(len(features)>0):
        features_with_constant = sm.add_constant(data[features])
        p_values = sm.OLS(target, features_with_constant).fit().pvalues[1:]
        max_p_value = p_values.max()
        if(max_p_value >= significance_level):
            excluded_feature = p_values.idxmax()
            features.remove(excluded_feature)
        else:
            break
    return features

significance = 0.05
initial_features = X_train.columns.tolist()
selected_features_BE = backward_elimination(X_train, y_train, significance)
removed_features = list(set(initial_features)-set(selected_features_BE))

print()
print(f"## Backward elimination (alpha = {significance})")
print("Selected features: ", selected_features_BE)
print("Removed features: ", removed_features)

def stepwise_selection(data, target, forward_significance = 0.05, backward_significance = 0.05):
    initial_features = data.columns.tolist()
    best_features = []
    while (len(initial_features)>0):
        remaining_features = list(set(initial_features)-set(best_features))
        new_pval = pd.Series(index=remaining_features)
        for new_column in remaining_features:
            model = sm.OLS(target, sm.add_constant(data[best_features+[new_column]])).fit()
            new_pval[new_column] = model.pvalues[new_column]

        min_p_value = new_pval.min()
        if(min_p_value<forward_significance):
            best_features.append(new_pval.idxmin())

            while(len(best_features)>0):
                best_features_with_constant = sm.add_constant(data[best_features])
                p_values = sm.OLS(target, best_features_with_constant).fit().pvalues[1:]
                max_p_value = p_values.max()
                if(max_p_value >= backward_significance):
                    excluded_feature = p_values.idxmax()
                    best_features.remove(excluded_feature)
                    initial_features.remove(excluded_feature)
                else:
                    break
        else:
            break
    return best_features

forward_significance = 0.05
backward_significance = 0.05
initial_features = X_train.columns.tolist()
selected_features_BS = stepwise_selection(X_train, y_train, forward_significance, backward_significance)
removed_features = list(set(initial_features)-set(selected_features_BS))

print()
print(f"## Bidirectional selection (forward alpha = {forward_significance}, backward alpha = {backward_significance})")
print("Selected features: ", selected_features_BS)
print("Removed features: ", removed_features)
