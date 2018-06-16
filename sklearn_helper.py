def sklearn_dataset_selector(dataset):
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
    
    return sklearnDataset

def sklearn_dataset_info_print(sklearnDataset):

    print("========== Information of dataset")
    print("Feature: shape = {}\nFeature names: {}".format(sklearnDataset.data.shape, sklearnDataset.feature_names))
    if hasattr(sklearnDataset, 'target_names'):
        print("Target: shape = {}\nTarget names: {}".format(sklearnDataset.target.shape, sklearnDataset.target_names))
