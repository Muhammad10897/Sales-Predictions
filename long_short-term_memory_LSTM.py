from data_prep_and_feature_eng import *

# To suppress Tensorflow message
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Suppress oneDNN messages

"""2.Model Selection & Implementation

We will compare ARIMA, Exponential Smoothing, Random Forest, XGBoost, and LSTM.

2.5 LSTM"""

# LSTM

def LSTM_reg(train, test, plot_name):
    mlflow.set_experiment(f'sales_forcasting')
    with mlflow.start_run(run_name="LSTM") as run:
        mlflow.set_tag('Reg', 'LSTM')
        scaler = MinMaxScaler()
        scaled_train = scaler.fit_transform(train[['sales']])
        scaled_test = scaler.transform(test[['sales']])

        def create_sequences(data, n_steps=7):
            X, y = [], []
            for i in range(n_steps, len(data)):
                X.append(data[i-n_steps:i, 0])
                y.append(data[i, 0])
            return np.array(X), np.array(y)

        n_steps =  7
        lstm_units = 50
        activation =  'relu'
        optimizer = 'adam'
        loss = 'mse'
        epochs = 100
        patience = 5
        dense_units = 1
        
        X_train_seq, y_train_seq = create_sequences(scaled_train, n_steps)
        X_test_seq, y_test_seq = create_sequences(scaled_test, n_steps)

        X_train_seq = X_train_seq.reshape((X_train_seq.shape[0], X_train_seq.shape[1], 1))
        X_test_seq = X_test_seq.reshape((X_test_seq.shape[0], X_test_seq.shape[1], 1))

        model = Sequential()
        model.add(LSTM(lstm_units, activation=activation, input_shape=(n_steps, 1)))
        model.add(Dense(dense_units))
        model.compile(optimizer=optimizer, loss=loss)

        early_stop = EarlyStopping(monitor='val_loss', patience=patience)
        model.fit(X_train_seq, y_train_seq, epochs=epochs, verbose=0, validation_data=(X_test_seq, y_test_seq), callbacks=[early_stop])

        scaled_preds = model.predict(X_test_seq, verbose=0)
        predictions = scaler.inverse_transform(scaled_preds).flatten()
        actuals = scaler.inverse_transform(y_test_seq.reshape(-1, 1)).flatten()

        ## metrics
        mae = mean_absolute_error(actuals, predictions)
        rmse = np.sqrt(mean_squared_error(actuals, predictions))
        r2 = r2_score(actuals, predictions)

        # # Log params, metrics, and model 
        mlflow.log_params({
            'n_steps': n_steps,
            'lstm_units': lstm_units,
            'activation': activation,
            'optimizer': optimizer,
            'loss': loss,
            'epochs': epochs,
            'patience': patience,
            'dense_units': dense_units
        })
        mlflow.log_metrics({'mae': mae, 'rmse': rmse, 'r2':r2})
        model_path = f'models/{plot_name}'  # Updated path for clarity
        mlflow.sklearn.log_model(model, f'{model.__class__.__name__}/{plot_name}')

        ## Model Performance Comparison
        plt.figure(figsize=(12, 6))
        plt.title(f'{plot_name}')
        plt.plot(test['date'], test['sales'], label='Actual Sales', color='black')
        plt.plot(test['date'][-len(predictions):], predictions, label='LSTM', linestyle='--')
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
    LSTM_reg(train, test, plot_name='LSTM')

if __name__ == '__main__':
    ## Call the main function 
    main()