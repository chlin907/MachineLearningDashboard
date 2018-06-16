# Template to perform knn classificaiton and regression with Scikit-learn and pandas 
# Parameters are definied in "Define hyperparameters and models, and applicable datasets"
import matplotlib.pyplot as plt
import pandas as pd
import sklearn_helper

def sklearn_knn(x_pd_dataframe, y_pd_series, n_neighbors = 3, mode = 'cls'):
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neighbors import KNeighborsRegressor

    if mode == 'cls':
        ml_method = 'knn classifier'
        ml_model = KNeighborsClassifier(n_neighbors=n_neighbors)
    elif mode == 'reg':
        ml_method = 'knn regressor'
        ml_model = KNeighborsRegressor(n_neighbors=n_neighbors)
    else:
        raise Exception('Invalid mode in knn model: ', mode)

    x_train, x_test, y_train, y_test =  train_test_split(x_pd_dataframe, y_pd_series, random_state=0)
    
    ml_model.fit(x_train, y_train)
    
    # Reporting
    print("========== ML training result")
    print('ML method: {}'.format(ml_method))
    print('Train score: {}'.format(ml_model.score(x_train, y_train)))
    print('Test score: {}'.format(ml_model.score(x_test, y_test)))

def main():
    #================================ Define hyperparameters and models 
    num_neighbors = 2   # Definition.  Float. number of neighbors for knn
    mode = 'cls'        # Options:    'cls' for classifier and 'reg' for regressor
    dataset = 'iris'    # Options:    'breast_cancer', 'iris', 'boston'
                        # Remark:     'brest_caner' is breast cancer dataset as binary classfication to dinstinguish ['malignant' 'benign'] tumor
                        #             'iris' is  dataset as multiclass classfication to dinstinguish ['setosa' 'versicolor' 'virginica'] iris types
                        #             'boston' is dataset as regression example to predict Boston housing price


    sklearn_dataset = sklearn_helper.sklearn_dataset_selector(dataset)
    sklearn_helper.sklearn_dataset_info_print(sklearn_dataset)

    x_pd_dataframe = pd.DataFrame(sklearn_dataset.data, columns = sklearn_dataset.feature_names)
    y_pd_series = pd.Series(sklearn_dataset.target)

    sklearn_knn(x_pd_dataframe, y_pd_series, n_neighbors = num_neighbors, mode = mode)


if __name__ == '__main__':
    main()


###Code recycle
##Print out feature details
#x_pd_dataframe.info()
#print(x_pd_dataframe.describe())
#
##Matrix scatter plot of features
#pd.plotting.scatter_matrix(x_pd_dataframe, c=y_pd_series, figsize=(15, 15), marker='o', hist_kwds={'bins':20}, s=60, alpha=.8)
#plt.show()
