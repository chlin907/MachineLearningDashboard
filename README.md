# Machine Learnining Baseline Dashboard
This project presents a quick performance evaluation across few basic models and is aimed o  serve as a concise databoard before heavy feature engineering and model tuning. The machine package and data format are based on [Scikit-learn ](http://scikit-learn.org/stable/)and [Pandas](https://pandas.pydata.org/). 

Five basic supervised learning models are selected: k nearest neighbor, linear model (linear regression and logistic regression), decision tree, random forest and support vector machine. Users only need to prepare feature matrix and labels and can obtain the basics evaluation metrics as the screenshot below.

![screenshot](./misc/screenshot.png)

Grid search is also avaialable across certain pre-defined parameter ranges to perfomance quick hyper-parameter optimization. 

![screenshot_gs](./misc/screenshot_gs.png)

# Usage

* main.py: Example of parameters and how use SklearnModel.py
* SklearnModel.py: Collection of basic models
* SklearnDatasetSelector.py: Collection of three Scikit-learn classical datasets 

User can choose either use import Scikit-learn tutorial datasets collected in SklearnDatasetSelector.py or prepare their own training data and feed to

- x_pd_dataframe: Feature matrix in Pandas dataframe
- y_pd_series: Data label in Pandas series