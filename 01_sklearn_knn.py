#!/usr/local/anaconda3/bin//python3
# Use knn to do machine learning under pandas formats
# 1. train_test_split usage: random_state is the random seed. Default test_size = 0.25 (25%) and train_size = 1 - 0.25

import matplotlib.pyplot as plt import pandas as pd

def knn(x_pd_dataframe, y_pd_series, n_neighbors = 3, mode = 'cls'):
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neighbors import KNeighborsRegressor

    if mode == 'cls':
        ml_method = 'knn classifier'
        ml_model = KNeighborsClassifier(n_neighbors=n_neighbors)
    elif mode == 'reg':
        ml_method = 'knn regressor'
        ml_model = KNeighborsRegressor(n_neighbors=n_neighbors)

    x_train, x_test, y_train, y_test =  train_test_split(x_pd_dataframe, y_pd_series, random_state=0)
    
    ml_model.fit(x_train, y_train)
    
    # Reporting
    print("====================================================")
    print("ML training result")
    print('ML method: {}'.format(ml_method))
    print('Train scoare: {}'.format(ml_model.score(x_train, y_train)))
    print('Test scoare: {}'.format(ml_model.score(x_test, y_test)))

def main():
    ### Define hyperparameters and models
    num_neighbors = 1   # number of neighbors for knn
    mode = 'reg'        # cls for classifier. reg for regressor

    ### Applicable cases in scikit learn datasets

    #from sklearn.datasets import load_iris
    #sklearnDataset = load_iris()
    #from sklearn.datasets import load_breast_cancer
    #sklearnDataset = load_breast_cancer()
    from sklearn.datasets import load_boston
    sklearnDataset = load_boston()

    print("====================================================")
    print("|  Information of dataset")
    print("====================================================")
    print("Feature: shape = {}\nFeature names: {}".format(sklearnDataset.data.shape, sklearnDataset.feature_names))
    if hasattr(sklearnDataset, 'target_names'):
        print("Target: shape = {}\nTarget names: {}".format(sklearnDataset.target.shape, sklearnDataset.target_names))

    x_pd_dataframe = pd.DataFrame(sklearnDataset.data, columns = sklearnDataset.feature_names)

    y_pd_series = pd.Series(sklearnDataset.target)

    knn(x_pd_dataframe, y_pd_series, n_neighbors = num_neighbors, mode = mode)

    # Print out feature details
    #x_pd_dataframe.info()
    #print(x_pd_dataframe.describe())

    # Matrix scatter plot of features
    #pd.plotting.scatter_matrix(x_pd_dataframe, c=y_pd_series, figsize=(15, 15), marker='o', hist_kwds={'bins':20}, s=60, alpha=.8)
    #plt.show()

if __name__ == '__main__':
    main()
