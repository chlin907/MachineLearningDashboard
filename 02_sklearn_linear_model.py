# Template to perform linear regression and logistic regression with Scikit-learn and pandas 
# Parameters are definied in "Define hyperparameters and models, and applicable datasets"

import matplotlib.pyplot as plt
import pandas as pd

def sklearn_linear_model(x_pd_dataframe, y_pd_series, mode = 'reg', regular_mode = 'simple', regular_param = 1.0):
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression, Ridge, Lasso
    from sklearn.linear_model import LogisticRegression

    ml_method = 'Linear model'

    if mode == 'reg':
        if regular_mode == 'simple':
            ml_model = LinearRegression(regular_param)
        elif regular_mode == 'l1':
            ml_model = Lasso(regular_param)
        elif regular_mode == 'l2':
            ml_model = Ridge(regular_param)
        else:
            raise Exception('Invalid regularularization mode in linear model: ', regular_mode)
    elif mode == 'cls':
        if regular_mode == 'simple':
            ml_model = LogisticRegression()
        elif regular_mode == 'l1' or regular_mode == 'l2':
            ml_model = LogisticRegression(penalty=regular_mode, C=regular_param)
        else: 
            raise Exception('Invalid regularularization mode in linear model: ', regular_mode)
    else:
        raise Exception('Invalid mode in linear model: ', mode)

    x_train, x_test, y_train, y_test =  train_test_split(x_pd_dataframe, y_pd_series, random_state=0)
    
    ml_model.fit(x_train, y_train)
    
    # Reporting
    print("========== ML training result")
    print('ML method: {} with "{}" mode,  "{}" regularization and regularization parameter "{}"'.format(ml_method, mode, regular_mode, regular_param ))
    print('Train score: {}'.format(ml_model.score(x_train, y_train)))
    print('Test score: {}'.format(ml_model.score(x_test, y_test)))
    
def main():
    #================================ Define hyperparameters and models, and applicable datasets
    num_neighbors = 1   # Definition: Int. number of neighbors for knn
    mode = 'reg'        # Options:    Str. 'reg' for linear regression and 'cls' for logistic regression
    reg_mode = 'l1'     # Options:    Str. 'simple, 'l1', 'l2'. 
                        # Remark:     'l1' and 'l2' mean Lasso and Ridge regressoins.
    reg_param = 1       # Definition: Float. Regularization parameter
                        # Remark:     Represent alpha value in linear regression and is proportional to regularizaton strength
                        #             Represent C value in logistic regression and INVERSELY proportional to regularization strength
                        #             Be aware the reg_param definition difference
    dataset = 'boston'  # Options:    'breast_cancer', 'iris', 'boston'
                        # Remark:     'brest_caner' is breast cancer dataset as binary classfication to dinstinguish ['malignant' 'benign'] tumor
                        #             'iris' is  dataset as multiclass classfication to dinstinguish ['setosa' 'versicolor' 'virginica'] iris types
                        #             'boston' is dataset as regression example to predict Boston housing price


    #================================ Applicable scikit-learn datasets
    if dataset == 'breast_cancer':
        from sklearn.datasets import load_breast_cancer   ### Binary classification example  
        sklearnDataset = load_breast_cancer()
    elif dataset == 'iris':
        from sklearn.datasets import load_iris    ### Multiclass classification example  
        sklearnDataset = load_iris()
    elif dataset == 'boston':
        from sklearn.datasets import load_boston   ### Regression example
        sklearnDataset = load_boston()
    else:
        raise Exception('Invalid dataset: ', dataset)

    #================================ Code starts
    print("========== Information of dataset")
    print("Feature: shape = {}\nFeature names: {}".format(sklearnDataset.data.shape, sklearnDataset.feature_names))
    if hasattr(sklearnDataset, 'target_names'):
        print("Target: shape = {}\nTarget names: {}".format(sklearnDataset.target.shape, sklearnDataset.target_names))

    x_pd_dataframe = pd.DataFrame(sklearnDataset.data, columns = sklearnDataset.feature_names)
    y_pd_series = pd.Series(sklearnDataset.target)

    sklearn_linear_model(x_pd_dataframe, y_pd_series, mode = mode,  regular_mode=reg_mode, regular_param =reg_param)

if __name__ == '__main__':
    main()
