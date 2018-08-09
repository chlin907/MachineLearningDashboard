class SklearnDatasetSelector:
    def select(self, dataset):
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

        self._sklearn_dataset_info_print(sklearn_dataset)

        return sklearn_dataset

    def _sklearn_dataset_info_print(self, sklearn_dataset):
        print("============================================")
        print("INFORMATION OF DATASET")
        print("Feature: shape = {}\nFeature names: {}".format(sklearn_dataset.data.shape, sklearn_dataset.feature_names))
        if hasattr(sklearn_dataset, 'target_names'):
            print("Target: shape = {}\nTarget names: {}".format(sklearn_dataset.target.shape, sklearn_dataset.target_names))
        print("============================================")

