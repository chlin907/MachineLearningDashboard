# Template to perform decisino tree classifier and regression with Scikit-learn and pandas 
# Parameters are definied in "Define hyperparameters and models, and applicable datasets"
import matplotlib.pyplot as plt
import pandas as pd
import sklearn_helper

def sklearn_decision_tree(x_pd_dataframe, y_pd_series, mode = 'cls', max_depth = 'None', plot_feature_importance = 'n'):
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.tree import DecisionTreeRegressor

    ml_method = 'Decision tree'
    random_state_num = 0

    ml_model = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state_num)

    if mode == 'reg':
        ml_model = DecisionTreeRegressor(max_depth=max_depth, random_state=random_state_num)
    elif mode == 'cls':
        ml_model = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state_num)
    else:
        raise Exception('Invalid mode in tree model: ', mode)

    x_train, x_test, y_train, y_test =  train_test_split(x_pd_dataframe, y_pd_series, random_state=0)
    
    ml_model.fit(x_train, y_train)
    
    # Reporting
    print("========== ML training result")
    print('ML method: {}'.format(ml_method))
    print('Train score: {}'.format(ml_model.score(x_train, y_train)))
    print('Test score: {}'.format(ml_model.score(x_test, y_test)))

    # Check feature importance
    if plot_feature_importance == 'y':
        import matplotlib.pyplot as plt
        import numpy as np
        print('Feature importance {}:'.format(ml_model.feature_importances_))
        # Plot feature importance
        n_features = x_pd_dataframe.columns.shape[0]
        plt.barh(range(n_features), ml_model.feature_importances_, align='center')
        plt.yticks(np.arange(n_features), x_pd_dataframe.columns)
        plt.xlabel('Feature importance')
        plt.ylabel('Feature')
        plt.show()

def main():
    #================================ Define hyperparameters and models, and applicable datasets
    num_max_depth = None   # Definition: Int or None. Max depth of trees
    mode = 'cls'           # Options:    Str. 'reg' for regression and 'cls' for classifier
    dataset = 'breast_cancer'     # Options:    'breast_cancer', 'iris', 'boston'
                           # Remark:     'brest_caner' is breast cancer dataset as binary classfication to dinstinguish ['malignant' 'benign'] tumor
                           #             'iris' is  dataset as multiclass classfication to dinstinguish ['setosa' 'versicolor' 'virginica'] iris types
                           #             'boston' is dataset as regression example to predict Boston housing price

    sklearn_dataset = sklearn_helper.sklearn_dataset_selector(dataset)
    sklearn_helper.sklearn_dataset_info_print(sklearn_dataset)

    x_pd_dataframe = pd.DataFrame(sklearn_dataset.data, columns = sklearn_dataset.feature_names)
    y_pd_series = pd.Series(sklearn_dataset.target)

    sklearn_decision_tree(x_pd_dataframe, y_pd_series, mode = mode, max_depth = num_max_depth)

if __name__ == '__main__':
    main()
