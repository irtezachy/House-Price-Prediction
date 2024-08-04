import os
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import pickle

# Function to train the model
def linear_regression(X_train, y_train):
    # train model
    lrmodel = LinearRegression().fit(X_train,y_train)
    
    os.makedirs('models', exist_ok=True)
    # Save the trained model
    with open('models/linear_regression.pkl', 'wb') as f:
        pickle.dump(lrmodel, f)

    return lrmodel

def decision_tree_regression(X_train, y_train):
    dtr_model = DecisionTreeRegressor(max_depth=5, max_features=12, random_state=567).fit(X_train,y_train)
    
    # Save the trained model
    with open('models/decision_tree_regressor.pkl', 'wb') as f:
        pickle.dump(dtr_model, f)

    return dtr_model

def random_forest_regression(X_train, y_train):
    rf_model = RandomForestRegressor(n_estimators=300, criterion='absolute_error', max_depth=5, max_features=12).fit(X_train,y_train)
    
    # Save the trained model
    with open('models/random_forest_regressi.pkl', 'wb') as f:
        pickle.dump(rf_model, f)

    return rf_model