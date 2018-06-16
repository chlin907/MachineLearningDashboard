# Template to perform support vector machine classifier and regression with Scikit-learn and pandas 
# Parameters are definied in "Define hyperparameters and models, and applicable datasets"
import matplotlib.pyplot as plt
import pandas as pd
import sklearn_helper

def sklearn_svm(x_pd_dataframe, y_pd_series, mode = 'reg', pen_param = 1.0):
    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVR, SVC

    ml_method = 'SVM model'

    if mode == 'reg':
        ml_model = SVR(C=pen_param)
    elif mode == 'cls':
        ml_model = SVC(C=pen_param)
    else:
        raise Exception('Invalid mode in linear model: ', mode)

    x_train, x_test, y_train, y_test =  train_test_split(x_pd_dataframe, y_pd_series, random_state=0)
    
    ml_model.fit(x_train, y_train)
    
    # Reporting
    print("========== ML training result")
    print('ML method: {}'.format(ml_method))
    print('Train score: {}'.format(ml_model.score(x_train, y_train)))
    print('Test score: {}'.format(ml_model.score(x_test, y_test)))
    
def main():
    #================================ Define hyperparameters and models, and applicable datasets
    mode = 'cls'        # Options:    Str. 'reg' for linear regression and 'cls' for logistic regression
    pen_param = 1       # Definition: Float. Penalty parameter
    dataset = 'iris'  # Options:    'breast_cancer', 'iris', 'boston'
                        # Remark:     'brest_caner' is breast cancer dataset as binary classfication to dinstinguish ['malignant' 'benign'] tumor
                        #             'iris' is  dataset as multiclass classfication to dinstinguish ['setosa' 'versicolor' 'virginica'] iris types
                        #             'boston' is dataset as regression example to predict Boston housing price



    sklearn_dataset = sklearn_helper.sklearn_dataset_selector(dataset)
    sklearn_helper.sklearn_dataset_info_print(sklearn_dataset)

    x_pd_dataframe = pd.DataFrame(sklearn_dataset.data, columns = sklearn_dataset.feature_names)
    y_pd_series = pd.Series(sklearn_dataset.target)

    sklearn_svm(x_pd_dataframe, y_pd_series, mode = mode, pen_param =pen_param)

if __name__ == '__main__':
    main()
