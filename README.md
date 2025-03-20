# Predicting S&P 500 Prices Using an Attention-Based LSTM Model
## Abstract
This repo presents a comprehensive approach to predicting the S&P 500 index prices using an attention-based Long Short-Term Memory (LSTM) model. The model leverages historical stock data to forecast future prices, incorporating hyperparameter optimization via Optuna to enhance predictive accuracy. The results demonstrate the model's effectiveness in capturing market trends, with a Mean Absolute Percentage Error (MAPE) of 2.64%. The repo also provides a detailed explanation of the methodology, including data preprocessing, model architecture, and evaluation metrics.

## 1. Introduction
The S&P 500 index is a critical benchmark for the U.S. stock market, and its accurate prediction is of significant interest to investors and financial analysts. Traditional statistical methods often fall short in capturing the complex, non-linear patterns in financial time series data. Recent advancements in deep learning, particularly in sequence modeling with LSTM networks, have shown promise in this domain. This paper introduces an attention-based LSTM model that not only captures temporal dependencies but also identifies the most relevant time steps for prediction, thereby improving forecast accuracy.

## 2. Methodology
### 2.1 Data Collection and Preprocessing
The historical data for the S&P 500 index is obtained using the yfinance library, which provides access to Yahoo Finance's extensive financial data. The data is preprocessed to remove multi-index column levels and reset the index for clarity. The 'Close' price is selected as the target variable for prediction.

```ruby
def load_data():
    data = yf.download(tickers="^GSPC", interval="1d", auto_adjust=True)
    df_data = pd.DataFrame(data)
    df_data.columns = df_data.columns.droplevel(1)
    df_data = df_data.rename_axis("Date").reset_index()
    return df_data
```

The data is normalized using the MinMaxScaler to scale values between 0 and 1, ensuring that the model is not biased towards larger values. A sliding window approach is employed to create input sequences (X) and target values (y), where each input sequence consists of a specified number of historical days (lookback), and the target value is the 'Close' price on the next day.

```ruby
def preprocess_data(df_data, lookback):
    data = df_data['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i-lookback:i, 0])
        y.append(scaled_data[i, 0])
    return np.array(X), np.array(y), scaler
```

### 2.2 Model Architecture
The model architecture consists of an LSTM layer followed by an attention mechanism. The attention mechanism computes attention scores for each time step in the LSTM output, which are then used to create a context vector summarizing the most relevant information. This context vector is passed through a fully connected layer to produce the final prediction.

```ruby
def attention_block(inputs):
    attention = Dense(1, activation='tanh')(inputs)
    attention = tf.nn.softmax(attention, axis=1)
    context = Multiply()([inputs, attention])
    context = Lambda(lambda x: tf.reduce_sum(x, axis=1))(context)
    return context

def build_model(lookback, lstm_units, dropout_rate=0.2, learning_rate=0.001):
    inputs = Input(shape=(lookback, 1))
    lstm_out = LSTM(lstm_units, return_sequences=True, dropout=dropout_rate)(inputs)
    attention_out = attention_block(lstm_out)
    outputs = Dense(1)(attention_out)
    model = Model(inputs, outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse')
    return model
```
  
### 2.3 Hyperparameter Optimization
Hyperparameter optimization is performed using Optuna, a framework designed for automated hyperparameter tuning. The objective function defines the hyperparameters to optimize, including the number of LSTM units, dropout rate, learning rate, and batch size. The validation loss is used as the metric to minimize.

```ruby
def objective(trial, X_train, y_train, X_val, y_val, lookback):
    lstm_units = trial.suggest_int('lstm_units', 32, 256, log=True)
    dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.5)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
    model = build_model(lookback, lstm_units, dropout_rate, learning_rate)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    pruning_callback = TFKerasPruningCallback(trial, 'val_loss')
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=batch_size, callbacks=[early_stopping, pruning_callback], verbose=0)
    return min(history.history['val_loss'])
```

### 2.4 Model Training and Evaluation
The final model is trained using the best hyperparameters identified by Optuna. Early stopping is employed to prevent overfitting, and the model's performance is evaluated using metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and Mean Absolute Percentage Error (MAPE).

```ruby
def train_final_model(X_train, y_train, X_val, y_val, lookback, best_params):
    model = build_model(lookback, best_params['lstm_units'], best_params['dropout_rate'], best_params['learning_rate'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, batch_size=best_params['batch_size'], callbacks=[early_stopping], verbose=1)
    return model
```

## 3. Results
The model demonstrates strong predictive capabilities for S&P 500 price forecasting. Below are the key performance metrics and visualizations that illustrate the model's effectiveness.

### 3.1 Model Training Performance
The training history shows the convergence of both training and validation loss over epochs. The consistent gap between training and validation loss indicates the model's ability to generalize well without overfitting.
![alt text](https://github.com/Isdinval/Predicting-S-P-500-Prices-Using-an-Attention-Based-LSTM-Model/blob/main/Images/model_training_hisotry.png?raw=true)



### 3.2 Performance Metrics
The model achieves the following performance metrics on the test set:

- Mean Absolute Error (MAE): $65.53
- Mean Squared Error (MSE): $9159.02
- Root Mean Squared Error (RMSE): $95.70
- Mean Absolute Percentage Error (MAPE): 2.64%

The model has been saved as 'sp500_prediction_model.h5' for future use.

### 3.3 Long-term Prediction Accuracy
The following visualization demonstrates the model's accuracy in predicting S&P 500 prices over the entire historical period (test data). The close alignment between actual and predicted prices highlights the model's effectiveness in capturing long-term market trends.
![alt text](https://github.com/Isdinval/Predicting-S-P-500-Prices-Using-an-Attention-Based-LSTM-Model/blob/main/Images/actual_vs_predicted_prices.png?raw=true)


### 3.4 Recent Performance (Last 90 Days)
This visualization focuses on the most recent 90 days, providing a closer look at the model's accuracy in the current market conditions. The recent MAE of $163.73 and MAPE of 2.75% demonstrate the model's reliability in the current market environment.
![alt text](https://github.com/Isdinval/Predicting-S-P-500-Prices-Using-an-Attention-Based-LSTM-Model/blob/main/Images/actual_vs_predicted_prices_zoom_90_days.png?raw=true)


### 3.5 Next Day Prediction
Based on the model's analysis of historical patterns, the forecast for March 20, 2025 predicts an S&P 500 closing price of $5782.15, representing a 2.15% increase ($121.85) from the previous closing price of $5660.30. The prediction includes a confidence interval of ±$159.01.
![alt text](https://github.com/Isdinval/Predicting-S-P-500-Prices-Using-an-Attention-Based-LSTM-Model/blob/main/Images/next_day_prediction.png?raw=true)

```ruby
def predict_next_day(model, df_data, sequence_length=30):
    most_recent_data = df_data['Close'].values[-sequence_length:]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(df_data['Close'].values.reshape(-1, 1))
    scaled_data = scaler.transform(most_recent_data.reshape(-1, 1))
    X_next = scaled_data.reshape(1, sequence_length, 1)
    scaled_prediction = model.predict(X_next)
    predicted_price = scaler.inverse_transform(scaled_prediction)[0][0]
    return predicted_price
```

## 4. Conclusion
This repo presents a robust approach to predicting S&P 500 prices using an attention-based LSTM model. The integration of hyperparameter optimization and attention mechanisms significantly enhances the model's predictive capabilities. The model achieves a MAPE of 2.64%, demonstrating its effectiveness in capturing market trends and providing accurate forecasts.
The visualizations clearly show the model's ability to track both long-term market trends and short-term price movements. The consistency between actual and predicted prices validates the model's reliability for financial forecasting applications.
Future work could explore the incorporation of additional features, such as sentiment analysis from news articles, technical indicators, or macroeconomic variables, to further improve forecast accuracy. Additionally, exploring ensemble methods that combine predictions from multiple model architectures could potentially enhance robustness in volatile market conditions.
