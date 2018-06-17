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

    def __init__(self, x_pd_dataframe_in, y_pd_series_in):
        self.x_pd_dataframe = x_pd_dataframe_in
        self.y_pd_series = y_pd_series_in
        self.x_train, self.x_test, self.y_train, self.y_test = self._train_test_split()

    def _train_test_split(self):
        from sklearn.model_selection import train_test_split

        num_random_state = 0
        x_train, x_test, y_train, y_test = train_test_split(self.x_pd_dataframe, self.y_pd_series, random_state=num_random_state)

        return x_train, x_test, y_train, y_test

    def _cross_val(self, n_fold, ml_model):
        from sklearn.model_selection import cross_val_score
        return cross_val_score(ml_model, self.x_pd_dataframe, self.y_pd_series).mean()

    def _reporting(self, ml_method, ml_model):
        print("========== ML training result")
        print('ML method: {}'.format(ml_method))
        print('Train score: {}'.format(ml_model.score(self.x_train, self.y_train)))
        print('Test score : {}'.format(ml_model.score(self.x_test, self.y_test)))
        k = 10
        print('{}-fold cv : {}'.format(k, self._cross_val(10, ml_model)))

    def sklearn_knn(self, n_neighbors = 3, mode = 'cls'):
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
    
    def sklearn_linear_model(self, mode = 'reg', regular_mode = 'simple', regular_param = 1.0):
        from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
    
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
            raise Exception('Invalid mode in {}: '.format(sys._getframe().f_code.co_name), mode)
    
        
        ml_model.fit(self.x_train, self.y_train)
        
        ml_method_ext = ' ({} with {})'.format(mode, regular_mode) if mode is not None else ''
        self._reporting(sys._getframe().f_code.co_name + ml_method_ext, ml_model)
    
    def sklearn_decision_tree(self, mode = 'cls', max_depth = 'None', plot_feature_importance = 'n'):
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

    
    def sklearn_random_forest(self, mode = 'cls', max_depth = 'None', n_estimators=10, plot_feature_importance = 'n'):
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
    
    def sklearn_svm(self, mode = 'reg', pen_param = 1.0):
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
