from data_prep_and_feature_eng import *

"""2.Model Selection & Implementation

We will compare ARIMA, Exponential Smoothing, Random Forest, XGBoost, and LSTM.

2.1 ARIMA (AutoRegressive Integrated Moving Average)"""

#ARIMA (AutoRegressive Integrated Moving Average)

def arima_reg(train, test, plot_name):
    mlflow.set_experiment(f'sales_forcasting')
    with mlflow.start_run(run_name="ARIMA") as run:
        mlflow.set_tag('Reg', 'ARIMA') 
        history = list(train['sales'])
        predictions = []

        for t in range(len(test)):
            p, d, q = 7, 1, 0
            order = (p,d,q)
            model = ARIMA(history, order=order)
            model_fit = model.fit()
            output = model_fit.forecast()
            pred = output[0]
            predictions.append(pred)
            history.append(test['sales'].iloc[t])

        ## metrics
        mae = mean_absolute_error(test['sales'], predictions)
        rmse = np.sqrt(mean_squared_error(test['sales'], predictions))
        r2 = r2_score(test['sales'], predictions)

        ## Log params, metrics, and model 
        mlflow.log_params({'p': p, 'd': d, 'q': q})
        mlflow.log_metrics({'mae': mae, 'rmse': rmse, 'r2':r2})
        mlflow.sklearn.log_model(model, f'{model.__class__.__name__}/{plot_name}')

        ## Model Performance Comparison
        plt.figure(figsize=(12, 6))
        plt.title(f'{plot_name}')
        plt.plot(test['date'], test['sales'], label='Actual Sales', color='black')
        plt.plot(test['date'], predictions, label='ARIMA', linestyle='--')
        plt.xlabel('Date')
        plt.ylabel('Sales')
        plt.legend()

        # # Save the plot to MLflow
        model_per_comp = plt.gcf()
        mlflow.log_figure(figure=model_per_comp, artifact_file=f'{plot_name}_model_per_comp.png')
        plt.close()

def main():

    ##---------------- Calling the above function -------------------- ##

    ## 1. without considering the imabalancing data
    arima_reg(train= train, test=test, plot_name='arima')




if __name__ == '__main__':
    ## Call the main function 
    main()