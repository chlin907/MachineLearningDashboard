# Template to perform various classification and regression with Scikit-learn and pandas
#
# Dataset avail in SklearnDatasetSelector: Breast cancer, Iris and Boston housing price
# Model avail in SklearnModel: knn, linear model, decision tree, random forest, svm
import pandas as pd
import SklearnModel
import SklearnDatasetSelector

if __name__ == '__main__':
    # Use Sklearn dataset
    """
    Remark:  'brest_caner' is binary classfication for ['malignant' 'benign'] tumor
             'iris' is multi-class classfication for ['setosa' 'versicolor' 'virginica'] iris types
             'boston' is regression example to predict Boston housing price
    """
    """
    Users can overwrite the dataset by preparing feature matrix to
    x_pd_dataframe = Feature matrix in Pandas dataframe
    y_pd_series = data label in Pandas series
    
    """
    ######################### Beginning of input settings
    #   dataset_name: Str. 'breast_cancer' 'iris' 'boston'
    dataset_name = 'breast_cancer'

    # Job control options
    #   mode: 'cls' for classification and 'reg' for regression
    #   run_grid_search: Boolean. To run grid search pre-defined default hyper parameters
    #   run_scaler_std: Boolean. To enable scaling by mean and std
    mode = 'reg'
    run_grid_search = True
    run_scaler_std = True

    # Mode control options
    #   knn parameters:
    #   num_neighbors: Float. Number of neighbors for knn
    num_neighbors = 2

    # Linear model parameters
    #   reg_mode: Str. 'simple', 'l1', 'l2'
    #   reg_param: Float. Regularization parameter
    #              Represent alpha value in linear regression (proportional to regularization strength)
    #              Represent C value in logistic regression (INVERSELY proportional to regularization strength)
    reg_mode = 'l2'
    reg_param = 1

    # Decision tree param
    #   num_max_depth: Int or None. Max depth of trees
    num_max_depth = None

    # Random forest and Decision tree parameters
    #   num_max_depth: Int or None. Max depth of trees
    #   num_estimator: Int. Num of tress to construct a forest
    num_max_depth = None
    num_estimator = 100

    # SVM parameters
    #   pen_param: Float. Penalty parameter
    pen_param = 0.1

    #########################End of input settings

    # Code starts

    # Data preparation: Convert features into a dataframe and label(y) into a series
    print('Analyze dataset {}'.format(dataset_name))

    sklearn_dataset = SklearnDatasetSelector.SklearnDatasetSelector().select(dataset_name)

    x_pd_dataframe = pd.DataFrame(sklearn_dataset.data, columns=sklearn_dataset.feature_names)
    if run_scaler_std:
        x_pd_dataframe = (x_pd_dataframe - x_pd_dataframe.mean()) / (x_pd_dataframe.std())

    y_pd_series = pd.Series(sklearn_dataset.target)

    # Modeling with various models
    modeling = SklearnModel.SklearnModel(x_pd_dataframe, y_pd_series)

    modeling.sklearn_knn(n_neighbors=num_neighbors, mode=mode, run_grid_search=run_grid_search)
    modeling.sklearn_linear_model(mode=mode, regular_mode=reg_mode, regular_param=reg_param, run_grid_search=run_grid_search)
    modeling.sklearn_decision_tree(mode=mode, max_depth=num_max_depth, run_grid_search=run_grid_search)
    modeling.sklearn_random_forest(mode=mode, max_depth=num_max_depth, n_estimators=num_estimator, run_grid_search=run_grid_search)
    modeling.sklearn_svm(mode=mode, pen_param=pen_param, run_grid_search=run_grid_search)
