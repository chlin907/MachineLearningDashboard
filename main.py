# Template to perform various classificaiton and regression with Scikit-learn and pandas 
# Parameters are definied in "Define hyperparameters and models, and applicable datasets"
#
# Available Dataset: Breast cancer, Iris and Boston housing price
# Method: knn, linear model, deicsion tree, random forest, svm
import matplotlib.pyplot as plt
import pandas as pd
import sklearn_helper

def main():
    #================================ Define hyperparameters and models 
    # Dataset and problem statement
    dataset_num = 3        # Int:     1 for 'breast_cancer'; 2 for 'iris'; 3 for 'boston'
                           # Remark:  'brest_caner' is binary classfication for ['malignant' 'benign'] tumor
                           #          'iris' is multiclass classfication for ['setosa' 'versicolor' 'virginica'] iris types
                           #          'boston' is regression example to predict Boston housing price
    mode = 'reg'           # Options: 'cls' for classification and 'reg' for regression
    
    run_grid_search = 'n'  # Str:     'y' or 'n'. To run grid search some default hyper parametes
    scaler_std = 'n'      # Str:     'y' or 'n'. To enable scaling by mean and std
    # knn parameters
    num_neighbors = 2   # Definition.  Float. number of neighbors for knn

    # Linear model parameters
    reg_mode = 'simple'     # Options:    Str. 'simple, 'l1', 'l2'. 
                        # Remark:     'l1' and 'l2' mean Lasso and Ridge regressoins.
    reg_param = 1       # Definition: Float. Regularization parameter
                        # Remark:     Represent alpha value in linear regression and is proportional to regularizaton strength
                        #             Represent C value in logistic regression and INVERSELY proportional to regularization strength
                        #             Be aware the reg_param definition difference
    # Decision tree parameters
    num_max_depth = None   # Definition: Int or None. Max depth of trees

    # Random forest parameters
    num_max_depth = None   # Definition: Int or None. Max depth of trees
    num_estimator = 100    # Definition: Int. Num of tress to construct a forest

    # SVM parameters
    pen_param = 0.1       # Definition: Float. Penalty parameter

    #================================ Code starts
    #================================ Data preparation: Convert features into a dataframe and label(y) into a series
    dataset_dict = {1:'breast_cancer', 2:'iris', 3:'boston'}
    dataset = dataset_dict[dataset_num]
    print('Analyze dataset {}'.format(dataset))

    sklearn_dataset = sklearn_helper.sklearn_dataset_selector(dataset)
    sklearn_helper.sklearn_dataset_info_print(sklearn_dataset)

    x_pd_dataframe = pd.DataFrame(sklearn_dataset.data, columns = sklearn_dataset.feature_names)
    if scaler_std == 'y':
        x_pd_dataframe = (x_pd_dataframe - x_pd_dataframe.mean()) / (x_pd_dataframe.std())

    y_pd_series = pd.Series(sklearn_dataset.target)

    #================================ Modeling with various models
    modeling = sklearn_helper.sklearn_model(x_pd_dataframe, y_pd_series)

    modeling.sklearn_knn(n_neighbors = num_neighbors, mode = mode, grid_search = run_grid_search)
    modeling.sklearn_linear_model(mode = mode, regular_mode = reg_mode, regular_param = reg_param, grid_search = run_grid_search)
    modeling.sklearn_decision_tree(mode = mode, max_depth = num_max_depth, grid_search = run_grid_search)
    modeling.sklearn_random_forest(mode = mode, max_depth = num_max_depth, n_estimators = num_estimator, grid_search = run_grid_search)
    modeling.sklearn_svm(mode = mode, pen_param = pen_param, grid_search = run_grid_search)

if __name__ == '__main__':
    main()
