from sklearn.model_selection import train_test_split
from src.data.make_dataset import load_dataset
from src.features.build_features import Train_Split
from src.models.train_model import linear_regression,decision_tree_regression, random_forest_regression
from src.models.predict_model import accuracy_model
from src.visualization.visualize import plot_tree

df=load_dataset('House-Price-Prediction\\data\\raw\\final.csv')
                
# Split the dataset into train test
x_train, x_test, y_train, y_test =Train_Split(df)
   
#Train the Linear regression model
lrmodel = linear_regression(x_train, y_train)

# Show the metrics of models
print("Coefficients of Linear Regression:", lrmodel.coef_)
print("Intercept of Linear Regression:", lrmodel.intercept_)
    
# Display evaluation metrics for linear regression
lr_train_mae, lr_test_mae = accuracy_model(lrmodel, x_train, x_test, y_train, y_test)
print('Linear Regression Train error is', lr_train_mae)
print('Linear Regression Test error is', lr_test_mae)
    
#Train the Decision Tree Regressor model
dtr_model = decision_tree_regression(x_train, y_train)
        
# Display evaluation metrics for Decision Tree Regressor
dtr_train_mae, dtr_test_mae = accuracy_model(dtr_model, x_train, x_test, y_train, y_test)
print('Decision Tree Regressor Train error is', dtr_train_mae)
print('Decision Tree Regressor Test error is', dtr_test_mae)
#Plot the decision tree
plot_tree(dtr_model, dtr_model.feature_names_in_, save_path='reports/figures/tree1.png')
    
#Train the Random Forest Regressor
rfr_model = random_forest_regression(x_train, y_train)
        
# Display evaluation metrics for Random Forest Regressor
rfr_train_mae, rfr_test_mae = accuracy_model(rfr_model, x_train, x_test, y_train, y_test)
print('Random Forest Regressor Train error is', rfr_train_mae)
print('Random Forest Regressor Test error is', rfr_test_mae)
    
#Plot the Random Forest Regressor
plot_tree(rfr_model.estimators_[2], dtr_model.feature_names_in_, save_path='reports/figures/tree2.png')