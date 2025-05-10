from data_prep_and_feature_eng import *

logged_model = 'runs:/7dcc09eb929c4ed4a55fa408f0bc706b/HoltWintersResultsWrapper/ETS'

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)

# Predict on a Pandas DataFrame.
loaded_model.predict(X_test)