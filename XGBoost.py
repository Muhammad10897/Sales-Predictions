from data_prep_and_feature_eng import *


"""2.Model Selection & Implementation

We will compare ARIMA, Exponential Smoothing, Random Forest, XGBoost, and LSTM.

2.4 XGBoost"""

# XGBoost

def xgb_reg(X_train, y_train, X_test, y_test, plot_name):
    mlflow.set_experiment(f'sales_forcasting')
    with mlflow.start_run(run_name="XGBoost") as run:
        mlflow.set_tag('Reg', 'xgb')
        n_estimators=200
        learning_rate=0.1
        model = XGBRegressor(n_estimators=n_estimators, learning_rate=learning_rate, random_state=42)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        ## metrics
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)

        # # Log params, metrics, and model 
        mlflow.log_params({'n_estimators': n_estimators, 'learning_rate': learning_rate})
        mlflow.log_metrics({'mae': mae, 'rmse': rmse, 'r2':r2})
        mlflow.sklearn.log_model(model, f'{model.__class__.__name__}/{plot_name}')

        ## Model Performance Comparison
        plt.figure(figsize=(12, 6))
        plt.title(f'{plot_name}')
        plt.plot(test['date'], test['sales'], label='Actual Sales', color='black')
        plt.plot(test['date'], predictions, label='XGBoost', linestyle='--')
        plt.xlabel('Date')
        plt.ylabel('Sales')
        plt.legend()

        # # Save the plot to MLflow
        model_per_comp = plt.gcf()
        mlflow.log_figure(figure=model_per_comp, artifact_file=f'{plot_name}_model_per_comp.png')
        plt.close()

def main():

    # ---------------- Calling the above function -------------------- ##

    ## 1. without considering the imabalancing data
    xgb_reg(X_train, y_train, X_test, y_test, plot_name='xgb')




if __name__ == '__main__':
    ## Call the main function 
    main()