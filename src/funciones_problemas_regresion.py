import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

class ArbolesDecision:

    def __init__(self, df, variable_respuesta):
        self.df = df
        self.variable_respuesta = variable_respuesta

    def separar_variables(self):
        self.X  = self.df.drop(columns=self.variable_respuesta)
        self.y = self.df[[self.variable_respuesta]]
    
    def separar_train_test(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y)

    def grid_search(self, params_arbol, scoring, cv):
        grid_search_arbol = GridSearchCV(DecisionTreeRegressor(), param_grid=params_arbol, cv=cv, scoring=scoring, n_jobs=-1)
        grid_search_arbol.fit(self.X_train, self.y_train)
        print(grid_search_arbol.best_estimator_)
        return grid_search_arbol.best_params_
    
    def metricas(self):
        metricas = {
            "train":{
                "r2_scores" : r2_score(self.y_train, self.y_train_pred),
                "MAE" :  mean_absolute_error(self.y_train, self.y_train_pred),
                "MSE" : mean_squared_error(self.y_train, self.y_train_pred),
                "RMSE" : np.sqrt(mean_squared_error(self.y_train, self.y_train_pred))
            },
            "test":{
                "r2_scores" : r2_score(self.y_test, self.y_test_pred),
                "MAE" :  mean_absolute_error(self.y_test, self.y_test_pred),
                "MSE" : mean_squared_error(self.y_test, self.y_test_pred),
                "RMSE" : np.sqrt(mean_squared_error(self.y_test, self.y_test_pred))
            }
        }
        return pd.DataFrame(metricas).T
    
    def fit_and_metrics(self, max_depth, max_leaf_nodes, min_samples_leaf, min_samples_split):
        modelo_arbol = DecisionTreeRegressor(max_depth=max_depth, max_leaf_nodes = max_leaf_nodes, min_samples_leaf = min_samples_leaf, min_samples_split=min_samples_split)
        modelo_arbol.fit(self.X_train, self.y_train)
        self.y_train_pred = modelo_arbol.predict(self.X_train)
        self.y_test_pred = modelo_arbol.predict(self.X_test)
        df_metricas = self.metricas()
        return df_metricas
    
    def grid_fit_metrics(self, params_arbol, scoring="neg_mean_squared_error", cv=5):
        print("Las mejores métricas para el modelo de DecisionTreeRegressor son:")
        dict_best_params = self.grid_search(params_arbol, scoring=scoring, cv=cv)
        df_metricas = self.fit_and_metrics(dict_best_params["max_depth"], dict_best_params["max_leaf_nodes"], dict_best_params["min_samples_leaf"], dict_best_params["min_samples_split"])
        return df_metricas
