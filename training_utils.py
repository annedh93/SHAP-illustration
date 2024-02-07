import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from bayes_opt import BayesianOptimization

class ML_model_hyperparameter_optimization:
    """
    Use this class to optimize hyperparameters of machine learning models 
    implemented with sklearn via Bayesian optimization.
    initialization:
    model: Type of ML model, e.g., XGBClassifier(random_state=42, 
    eval_metric='logloss', use_label_encoder=False).
    pbounds: Parameter bounds to be used during optimization, e.g., 
    {'max_depth':[2,15], 'n_estimators':[100,200]}
    integer_params: Indicator whether parameter should be integer value or not,
    e.g., {'max_depth':True, 'n_estimators':True}
    X: Train dataset.
    y: Train labels.
    """
    def __init__(self, model, pbounds, integer_params, X, y):
        self.model = model
        self.pbounds = pbounds
        self.integer_params = integer_params
        self.X = X
        self.y = y
    
    def objective_function(self, **parameters):
        """
        This is your objective function. It returns 5-fold cross-validated AUC 
        score.
        parameters: These parameters are optimized. The names and limits can be 
        found in self.pbounds.
        """
        parameter_names = list(self.pbounds.keys())
        params = {k:round(parameters[k]) if self.integer_params[k] else parameters[k] for k in parameter_names}
        score = []
        for rand in range(0,5):
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=rand)
            score.append(cross_val_score(self.model.set_params(**params), self.X, self.y, scoring='roc_auc', cv=skf).mean())
        return np.mean(score)

def train_model(model_specifications, train_X, train_y, test_X, test_y):
    """
    Use this function to train a machine learning model (including hyperparameter
    optimization) from the sklearn (or sklearn compliant) library. Returns 
    probabilities for train and test data and the optimal hyperparameters.
    model_specifications: These are the model specifications that can be passed
    via ** arguments (hence dictionary) to the class ML_model_hyperparameter_optimization,
    e.g. {'model':XGBClassifier(random_state=42, eval_metric='logloss', 
           use_label_encoder=False),
          'pbounds':{'max_depth':[2,15], 'n_estimators':[100,200]},
          'integer_params':{'max_depth':True, 'n_estimators':True},
          'X':train_X, 'y':train_y.combined}
    train_X: Train dataset.
    train_y: Train labels.
    test_X: Test dataset.
    test_y: Test labels.
    """
    model_instance = ML_model_hyperparameter_optimization(**model_specifications)
    optimizer = BayesianOptimization(f = model_instance.objective_function,
                                    pbounds = model_instance.pbounds, verbose = 0,
                                    random_state = 42)
    optimizer.maximize(init_points = 20, n_iter = 4)
    print("Best result: {}; f(x) = {}.".format(optimizer.max["params"], optimizer.max["target"]))

    # Get optimized parameters
    parameter_names = list(model_instance.pbounds.keys())
    int_params = model_instance.integer_params
    best_params = optimizer.max['params']
    best_params = {k:round(best_params[k]) if int_params[k] else best_params[k] for k in parameter_names}

    # Train model
    model = model_instance.model
    model.set_params(**best_params)
    model.fit(train_X, train_y)
    train_prob = model.predict_proba(train_X)
    test_prob = model.predict_proba(test_X)
    print('train auc', roc_auc_score(train_y, train_prob[:,1]))
    print('test auc', roc_auc_score(test_y, test_prob[:,1]))
    return train_prob, test_prob, best_params, model