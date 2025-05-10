## MLFLOW PROJECT
``` bash
# Get conda channels
conda config --show channels

# Build a MLFlow project, if you use one entry point with name (main)
mlflow run . --experiment-name <exp-name> # here it is {sales-prediction}

# If you have multiple entry points
mlflow run -e random_forest . --experiment-name sales-prediction
mlflow run -e XGBoost . --experiment-name sales-prediction
mlflow run -e LSTM . --experiment-name sales-prediction
mlflow run -e ETS . --experiment-name sales-prediction
mlflow run -e ARIMA . --experiment-name sales-prediction

```

```
## MLFLOW Models
``` bash
# serve the model via REST
mlflow models serve -m "path" --port 8000 --env-manager=local
mlflow models serve -m "file:///C:/Users/muham/Desktop/DEPI_project/milestone4_mlflow/mlruns/126697526254345711/4556e4289854492bb3090d0e620dfd4c/artifacts/XGBRegressor/xgb" --port 8000 --env-manager=local

# it will open in this link
http://localhost:8000/invocations
```

``` python
# exmaple of data to be sent


## multiple samples
{
    "dataframe_split": {
        "columns": [
            "transactions",
            "dcoilwtico",
            "holiday_type",
            "is_weekend",
            "month",
            "day_of_week",
            "day_of_month",
            "sales_lag_1",
            "sales_lag_7",
            "sales_lag_14",
            "sales_lag_30"
        ],
        "data": [
            [2318844.0, 97.65, 1, 0, 1, 3, 31, 281061.127052, 247245.690995, 267498.515975, 2511.618999],
            [2583966.0, 97.46, 1, 0, 2, 4, 1, 271254.217996, 290022.771930, 296130.850028, 496092.417944],
            [2485890.0, 96.21, 1, 0, 2, 0, 4, 486336.820180, 285460.169953, 311211.265950, 477350.121229],
            [2514798.0, 96.68, 1, 0, 2, 1, 5, 344308.715017, 264488.818076, 296214.728983, 519695.401088],
            [2484603.0, 96.44, 1, 0, 2, 2, 6, 321245.839130, 281061.127052, 283258.453032, 336122.801066]
        ]
    }
}
```

``` bash 
# if you want to use curl

curl -X POST \
  http://localhost:8000/invocations \
  -H 'Content-Type: application/json' \
  -d '{
    "dataframe_split": {
        "columns": [
            "transactions",
            "dcoilwtico",
            "holiday_type",
            "is_weekend",
            "month",
            "day_of_week",
            "day_of_month",
            "sales_lag_1",
            "sales_lag_7",
            "sales_lag_14",
            "sales_lag_30"
        ],
        "data": [
            [2318844.0, 97.65, 1, 0, 1, 3, 31, 281061.127052, 247245.690995, 267498.515975, 2511.618999],
            [2583966.0, 97.46, 1, 0, 2, 4, 1, 271254.217996, 290022.771930, 296130.850028, 496092.417944],
            [2485890.0, 96.21, 1, 0, 2, 0, 4, 486336.820180, 285460.169953, 311211.265950, 477350.121229],
            [2514798.0, 96.68, 1, 0, 2, 1, 5, 344308.715017, 264488.818076, 296214.728983, 519695.401088],
            [2484603.0, 96.44, 1, 0, 2, 2, 6, 321245.839130, 281061.127052, 283258.453032, 336122.801066]
        ]
    }


# if you want to use Powershell
Invoke-RestMethod -Uri "http://localhost:8000/invocations" -Method Post -Headers @{"Content-Type" = "application/json"} -Body '{
    "dataframe_split": {
        "columns": [
            "transactions",
            "dcoilwtico",
            "holiday_type",
            "is_weekend",
            "month",
            "day_of_week",
            "day_of_month",
            "sales_lag_1",
            "sales_lag_7",
            "sales_lag_14",
            "sales_lag_30"
        ],
        "data": [
            [2318844.0, 97.65, 1, 0, 1, 3, 31, 281061.127052, 247245.690995, 267498.515975, 2511.618999],
            [2583966.0, 97.46, 1, 0, 2, 4, 1, 271254.217996, 290022.771930, 296130.850028, 496092.417944],
            [2485890.0, 96.21, 1, 0, 2, 0, 4, 486336.820180, 285460.169953, 311211.265950, 477350.121229],
            [2514798.0, 96.68, 1, 0, 2, 1, 5, 344308.715017, 264488.818076, 296214.728983, 519695.401088],
            [2484603.0, 96.44, 1, 0, 2, 2, 6, 321245.839130, 281061.127052, 283258.453032, 336122.801066]
        ]
    }
}

```

```