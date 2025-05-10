from data_prep_and_feature_eng import *


"""2.Model Selection & Implementation

We will compare ARIMA, Exponential Smoothing, Random Forest, XGBoost, and LSTM.

2.2 Exponential Smoothing (ETS)"""

# Exponential Smoothing (ETS)
def ETS_reg(train, test, plot_name):
    mlflow.set_experiment(f'sales_forcasting')
    with mlflow.start_run(run_name="ETS") as run:
        mlflow.set_tag('Reg', 'ETS')
        trend = 'add'
        seasonal='add'
        seasonal_periods=7
        model = ExponentialSmoothing(train['sales'], trend=trend, seasonal=seasonal, seasonal_periods=seasonal_periods).fit()
        predictions = model.forecast(len(test))

        ## metrics
        mae = mean_absolute_error(test['sales'], predictions)
        rmse = np.sqrt(mean_squared_error(test['sales'], predictions))
        r2 = r2_score(test['sales'], predictions)

        # # Log params, metrics, and model 
        mlflow.log_params({'trend': trend, 'seasonal': seasonal, 'seasonal_periods':seasonal_periods})
        mlflow.log_metrics({'mae': mae, 'rmse': rmse, 'r2':r2})
        mlflow.sklearn.log_model(model, f'{model.__class__.__name__}/{plot_name}')

        ## Model Performance Comparison
        plt.figure(figsize=(12, 6))
        plt.title(f'{plot_name}')
        plt.plot(test['date'], test['sales'], label='Actual Sales', color='black')
        plt.plot(test['date'], predictions, label='ETS', linestyle='--')
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
    ETS_reg(train, test, plot_name='ETS')




if __name__ == '__main__':
    ## Call the main function 
    main()