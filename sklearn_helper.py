import sys


def sklearn_dataset_selector(dataset):
    if dataset == 'breast_cancer':
        from sklearn.datasets import load_breast_cancer   ### Binary classification example  sklearnDataset = load_breast_cancer() elif dataset == 'iris':
        sklearn_dataset = load_breast_cancer()
    elif dataset == 'iris':
        from sklearn.datasets import load_iris    ### Multiclass classification example  
        sklearn_dataset = load_iris()
    elif dataset == 'boston':
        from sklearn.datasets import load_boston   ### Regression example
        sklearn_dataset = load_boston()
    else:
        raise Exception('Invalid dataset: ', dataset)
    
    return sklearn_dataset

def sklearn_dataset_info_print(sklearn_dataset):

    print("========== Information of dataset")
    print("Feature: shape = {}\nFeature names: {}".format(sklearn_dataset.data.shape, sklearn_dataset.feature_names))
    if hasattr(sklearn_dataset, 'target_names'):
        print("Target: shape = {}\nTarget names: {}".format(sklearn_dataset.target.shape, sklearn_dataset.target_names))

####################################################################################
class sklearn_model:
    # Use for grid search only
    _grid_param_knn = {'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
    _grid_param_svm = {'C': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'gamma': [0.01, 0.05, 0.1, 0.5]}
    _grid_param_dt = {
         'min_samples_leaf': range(10, 60,10),
         'max_depth': range(3, 14, 2),
         'max_features': [2, 3],
         'min_samples_split': range(50,201,20),
    }
    _grid_param_rf = {
         'n_estimators' : [100, 200, 300, 400],
         'max_depth': range(4, 10, 2),
    }
    _grid_param_logistic_reg = {'C':[0.01,0.05,0.1,0.5,1,5,10,50,100]}
    _grid_param_ridge_lasso = {'alpha':[0.01,0.05,0.1,0.5,1,5,10,50,100]}

    def __init__(self, x_pd_dataframe_in, y_pd_series_in):
        self.x_pd_dataframe = x_pd_dataframe_in
        self.y_pd_series = y_pd_series_in
        self.x_train, self.x_test, self.y_train, self.y_test = self._train_test_split()

    def _train_test_split(self):
        from sklearn.model_selection import train_test_split
        #x_train, x_test, y_train, y_test = train_test_split(self.x_pd_dataframe, self.y_pd_series, test_size=0.2,  random_state=num_random_state)
        x_train, x_test, y_train, y_test = train_test_split(self.x_pd_dataframe, self.y_pd_series)

        return x_train, x_test, y_train, y_test

    def _cross_val(self, n_fold, ml_model):
        from sklearn.model_selection import cross_val_score
        return cross_val_score(ml_model, self.x_pd_dataframe, self.y_pd_series).mean()

    def _reporting(self, ml_method, ml_model):
        k = 3
        print("========== ML training result")
        print('ML method: {}'.format(ml_method))
        print('Train score: {}'.format(ml_model.score(self.x_train, self.y_train)))
        print('Test score : {}'.format(ml_model.score(self.x_test, self.y_test)))
        print('{}-fold cv : {}'.format(k, self._cross_val(k, ml_model)))

    def _grid_search(self, sklearn_model, grid_param, mode):  # mode = 'reg' or 'cls
        from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
        cv_num = 10  # 10-fold cv
        jobs = -1    # use all cpu's
        if mode == 'reg':
            param_search = GridSearchCV(estimator=sklearn_model, param_grid=grid_param, scoring='r2', cv=cv_num, n_jobs=jobs)
        elif mode == 'cls':
            param_search = GridSearchCV(estimator=sklearn_model, param_grid=grid_param, scoring='accuracy', cv=cv_num, n_jobs=jobs)
        else:
            raise Exception('Invalid mode in {}: '.format(sys._getframe().f_code.co_name), mode)

        param_search.fit(self.x_pd_dataframe, self.y_pd_series)

        print('Best params: %s' % param_search.best_params_)
        print('Best training accuracy: %.3f' % param_search.best_score_)

        return param_search.best_estimator_


    def sklearn_knn(self, n_neighbors = 3, mode = 'cls', grid_search='y'):
        from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    
        if mode == 'cls':
            ml_model = KNeighborsClassifier(n_neighbors=n_neighbors)
        elif mode == 'reg':
            ml_model = KNeighborsRegressor(n_neighbors=n_neighbors)
        else:
            raise Exception('Invalid mode in {}: '.format(sys._getframe().f_code.co_name), mode)
        
        ml_model.fit(self.x_train, self.y_train)

        ml_method_ext = ' ({})'.format(mode) if mode is not None else ''
        self._reporting(sys._getframe().f_code.co_name + ml_method_ext, ml_model)

        if grid_search == 'y':
            grid_param = self._grid_param_knn
            print('Grid search conditions: {}'.format(grid_param))
            self._grid_search(sklearn_model=ml_model, grid_param=grid_param, mode = mode)

    def sklearn_linear_model(self, mode = 'reg', regular_mode = 'simple', regular_param = 1.0, grid_search='y'):
        from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
    
        if mode == 'reg':
            if regular_mode == 'simple':
                ml_model = LinearRegression()
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
            raise Exception('Invalid mode in {}: '.format(sys._getframe().f_code.co_name), mode)
    
        
        ml_model.fit(self.x_train, self.y_train)
        
        ml_method_ext = ' ({} with {})'.format(mode, regular_mode) if mode is not None else ''
        self._reporting(sys._getframe().f_code.co_name + ml_method_ext, ml_model)

        if grid_search == 'y':
            if mode == 'reg' and regular_mode == 'simple':
                print('No grid search is done on simple linear regression')
            else:
                grid_param = self._grid_param_ridge_lasso if mode == 'reg' else self._grid_param_logistic_reg
                print('Grid search conditions: {}'.format(grid_param))
                self._grid_search(sklearn_model=ml_model, grid_param=grid_param, mode = mode)
    
    def sklearn_decision_tree(self, mode = 'cls', max_depth = 'None', grid_search='y'):
        from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

        if mode == 'reg':
            ml_model = DecisionTreeRegressor(max_depth=max_depth)
        elif mode == 'cls':
            ml_model = DecisionTreeClassifier(max_depth=max_depth)
        else:
            raise Exception('Invalid mode in {}: '.format(sys._getframe().f_code.co_name), mode)
    
        
        ml_model.fit(self.x_train, self.y_train)
        
        ml_method_ext = ' ({})'.format(mode) if mode is not None else ''
        self._reporting(sys._getframe().f_code.co_name + ml_method_ext, ml_model)

        if grid_search == 'y':
            grid_param = self._grid_param_dt
            print('Grid search conditions: {}'.format(grid_param))
            self._grid_search(sklearn_model=ml_model, grid_param=grid_param, mode = mode)

    
    def sklearn_random_forest(self, mode = 'cls', max_depth = 'None', n_estimators=10, grid_search = 'y'):
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    
        if mode == 'reg':
            ml_model = RandomForestRegressor(n_estimators = n_estimators, max_depth=max_depth)
        elif mode == 'cls':
            ml_model = RandomForestClassifier(n_estimators = n_estimators, max_depth=max_depth)
        else:
            raise Exception('Invalid mode in {}: '.format(sys._getframe().f_code.co_name), mode)
    
        ml_model.fit(self.x_train, self.y_train)
        
        ml_method_ext = ' ({})'.format(mode) if mode is not None else ''
        self._reporting(sys._getframe().f_code.co_name + ml_method_ext, ml_model)
    
        ## Check feature importance
        #if plot_feature_importance == 'y':
        #    import matplotlib.pyplot as plt
        #    import numpy as np
        #    print('Feature importance {}:'.format(ml_model.feature_importances_))
        #    # Plot feature importance
        #    n_features = x_pd_dataframe.columns.shape[0]
        #    plt.barh(range(n_features), ml_model.feature_importances_, align='center')
        #    plt.yticks(np.arange(n_features), x_pd_dataframe.columns)
        #    plt.xlabel('Feature importance')
        #    plt.ylabel('Feature')
        #    plt.show()

        if grid_search == 'y':
            grid_param = self._grid_param_rf
            print('Grid search conditions: {}'.format(grid_param))
            self._grid_search(sklearn_model=ml_model, grid_param=grid_param, mode = mode)
    
    def sklearn_svm(self, mode = 'reg', pen_param = 1.0, grid_search = 'y'):
        from sklearn.svm import SVR, SVC
    
        if mode == 'reg':
            ml_model = SVR(C=pen_param)
        elif mode == 'cls':
            ml_model = SVC(C=pen_param)
        else:
            raise Exception('Invalid mode in {}: '.format(sys._getframe().f_code.co_name), mode)
    
        ml_model.fit(self.x_train, self.y_train)

        ml_method_ext = ' ({})'.format(mode) if mode is not None else ''
        self._reporting(sys._getframe().f_code.co_name + ml_method_ext, ml_model)

        if grid_search == 'y':
            grid_param = self._grid_param_svm
            print('Grid search conditions: {}'.format(grid_param))
            self._grid_search(sklearn_model=ml_model, grid_param=grid_param, mode = mode)
