from data_prep_and_feature_eng import *


"""2.Model Selection & Implementation

We will compare ARIMA, Exponential Smoothing, Random Forest, XGBoost, and LSTM.

2.3 Random Forest"""

#Random Forest

def random_forest_reg(X_train, y_train, plot_name, n_estimators, max_depth):
    mlflow.set_experiment(f'sales_forcasting')
    with mlflow.start_run(run_name="Random Forest") as run:
        mlflow.set_tag('Reg', 'forest') 
        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=45)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        ## metrics
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)

        # # Log params, metrics, and model 
        mlflow.log_params({'n_estimators': n_estimators, 'max_depth': max_depth})
        mlflow.log_metrics({'mae': mae, 'rmse': rmse, 'r2':r2})
        mlflow.sklearn.log_model(model, f'{model.__class__.__name__}/{plot_name}')

        ## Model Performance Comparison
        plt.figure(figsize=(12, 6))
        plt.title(f'{plot_name}')
        plt.plot(test['date'], test['sales'], label='Actual Sales', color='black')
        plt.plot(test['date'], predictions, label='Random Forest', linestyle='--')
        plt.xlabel('Date')
        plt.ylabel('Sales')
        plt.legend()

        # # Save the plot to MLflow
        model_per_comp = plt.gcf()
        mlflow.log_figure(figure=model_per_comp, artifact_file=f'{plot_name}_model_per_comp.png')
        plt.close()

def main(n_estimators: int, max_depth: int):

    # ---------------- Calling the above function -------------------- ##

    ## 1. without considering the imabalancing data
    random_forest_reg(X_train=X_train, y_train=y_train, plot_name='random_forest', 
                n_estimators=n_estimators, max_depth=max_depth)




if __name__ == '__main__':
    ## Take input from user via CLI using argparser library
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_estimators', '-n', type=int, default=100)
    parser.add_argument('--max_depth', '-d', type=int, default=15)
    args = parser.parse_args()

    ## Call the main function 
    main(n_estimators=args.n_estimators, max_depth=args.max_depth)