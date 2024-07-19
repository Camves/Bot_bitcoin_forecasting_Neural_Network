import numpy as np
import pandas as pd
from binance.client import Client
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import GlorotUniform
from tensorflow.keras.models import load_model
#from tensorflow.compat.v1.keras.backend import set_session
import tensorflow as tf
import matplotlib.pyplot as plt
import talib
import time
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os
from datetime import datetime

# Verificar se a GPU está disponível
print("GPU Available: ", tf.test.is_gpu_available())

# Configuração de GPU para alocação de memória
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
       
    except RuntimeError as e:
        print(e)
        
# Configuração da API
api_key = 'SUA_API_KEY'
api_secret = 'SEU_API_SECRET'

client = Client(api_key, api_secret)

# Configuração da carteira
max_wallet_value = 10  # Valor máximo da carteira em USDT
investment_percentage = 0.1  # Porcentagem máxima do valor da carteira que pode ser investida

# Função para obter dados históricos
def get_historical_data(symbol, interval, start, end):
    klines = client.get_historical_klines(symbol, interval, start, end)
    data = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
    ])
    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
    data.set_index('timestamp', inplace=True)
    return data.astype(float)


# Calcular indicadores técnicos
def calculate_indicators(data):
    data['SMA_50'] = data['close'].rolling(window=50).mean()
    data['SMA_200'] = data['close'].rolling(window=200).mean()
    
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    data['Upper_Band'], data['Middle_Band'], data['Lower_Band'] = talib.BBANDS(data['close'], timeperiod=20)
    
    data['MACD'], data['MACD_Signal'], data['MACD_Hist'] = talib.MACD(data['close'], fastperiod=12, slowperiod=26, signalperiod=9)
    
    return data

# Preparar dados de treinamento e teste
def prepare_data(data, look_back, scaler=None, is_train=True):
    # Garantir que não há valores nulos
    data = data.dropna()

    features = ['close', 'SMA_50', 'SMA_200', 'RSI', 'Upper_Band', 'Middle_Band', 'Lower_Band', 'MACD', 'MACD_Signal', 'MACD_Hist']
    
    if is_train:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data[features])
    else:
        scaled_data = scaler.transform(data[features])

    x, y = [], []
    for i in range(look_back, len(scaled_data)):
        x.append(scaled_data[i-look_back:i, :])
        y.append(scaled_data[i, 0])
    
    x = np.array(x)
    y = np.array(y)
    return x, y, scaler


# Função para prever preços com Monte Carlo e redes neurais
def predict_prices(model, data, look_back, scaler):
    data = data.dropna()
    features = ['close', 'SMA_50', 'SMA_200', 'RSI', 'Upper_Band', 'Middle_Band', 'Lower_Band', 'MACD', 'MACD_Signal', 'MACD_Hist']
    # Buscar os últimos `look_back` pontos e garantir que temos todas as colunas necessárias:
    inputs = data[features].iloc[-look_back:]
    inputs_scaled = scaler.transform(inputs)
    
    # Garantir que os dados estão na forma correta:
    inputs_scaled = np.reshape(inputs_scaled, (1, look_back, inputs_scaled.shape[1]))
    print(inputs_scaled)
    predicted_price = model.predict(inputs_scaled)
    print(predicted_price)
    pred_df = pd.DataFrame(predicted_price, columns=['close'])
    zeros = np.zeros((pred_df.shape[0], inputs_scaled.shape[2] - 1))
    pred_df = pd.concat([pred_df, pd.DataFrame(zeros)], axis=1)
    
    predicted_price = scaler.inverse_transform(pred_df)
    print(predicted_price[0][0])
    return predicted_price[0][0]


# Funções de compra e venda
def buy(symbol, quantity):
    try:
        order = client.order_market_buy(
            symbol=symbol,
            quantity=quantity
        )
        return order
    except Exception as e:
        print(f"Erro ao comprar: {e}")
        return None

def sell(symbol, quantity):
    try:
        order = client.order_market_sell(
            symbol=symbol,
            quantity=quantity
        )
        return order
    except Exception as e:
        print(f"Erro ao vender: {e}")
        return None

# Estratégia de negociação
def trading_strategy():
    model = load_model('mercado_crypto.h5')
    #model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    balance = client.get_asset_balance(asset='USDT')
    usdt_balance = float(balance['free'])
    print("minha carteira: "+str(balance))
    max_investment = max_wallet_value * investment_percentage
    print("analisando")
    #if usdt_balance > 10:
    # Usar a predição mais recente:
    predicted_price = predict_prices(model, data, look_back, scaler)
    print("predito: " + str(predicted_price))
    current_price = data['close'].iloc[-1]
    print("atual: " + str(current_price))
    print("limiar: " + str(current_price * 1.005))
    if predicted_price > current_price * 1.005:
        investment_amount = min(max_investment, usdt_balance)
        buy('BTCUSDT', 0.00001)
        print("comprei")
    elif predicted_price < current_price * 0.995:
        btc_balance = client.get_asset_balance(asset='BTC')
        print("vendi")
        sell('BTCUSDT', btc_balance)

print("coletando dados")
now = datetime.utcnow()
dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
data = get_historical_data('BTCUSDT', Client.KLINE_INTERVAL_1MINUTE, '26 Apr 2024', dt_string)
print("dados coletados")
data = calculate_indicators(data)

# Separar em conjunto de treinamento e teste
train_size = int(len(data) * 0.8)
train_data = data[:train_size]
look_back = 30
train_x, train_y, scaler = prepare_data(train_data, look_back, is_train=True)
test_data = data[train_size:]
test_x, test_y, _ = prepare_data(test_data, look_back, scaler=scaler, is_train=False)

# Verificação das dimensões dos dados
print(f"Train x shape: {train_x.shape}")
print(f"Train y shape: {train_y.shape}")
print(f"Test x shape: {test_x.shape}")
print(f"Test y shape: {test_y.shape}")

# Criação e compilação do modelo de rede neural
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(look_back, train_x.shape[2]), kernel_initializer=GlorotUniform()))
model.add(LSTM(50, kernel_initializer=GlorotUniform()))
model.add(Dense(1, kernel_initializer=GlorotUniform()))
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Treinamento do modelo
model.fit(train_x, train_y, epochs=10, batch_size=32, verbose=1)
model.save('mercado_crypto.h5')

# Previsão com dados de teste
predictions = model.predict(test_x)

# Transformar previsões de volta à escala original
predicted_prices = scaler.inverse_transform(np.concatenate([predictions, np.zeros((predictions.shape[0], test_x.shape[2] - predictions.shape[1]))], axis=1))[:, 0]
actual_prices = scaler.inverse_transform(np.concatenate([test_y.reshape(-1, 1), np.zeros((test_y.shape[0], test_x.shape[2] - 1))], axis=1))[:, 0]

# Avaliação do modelo
mse = mean_squared_error(actual_prices, predicted_prices)
mae = mean_absolute_error(actual_prices, predicted_prices)

print(f'Mean Squared Error: {mse}')
print(f'Mean Absolute Error: {mae}')

# Visualizar Previsões vs Preços Reais
plt.figure(figsize=(10, 6))
plt.plot(actual_prices, label='Actual Prices')
plt.plot(predicted_prices, label='Predicted Prices')
plt.title('Actual Prices vs Predicted Prices')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()

#model = load_model('mercado_crypto.h5')
#model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

while True:
        
        try:
            now = datetime.utcnow()
            dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
            print("coletando dados")
            data = get_historical_data('BTCUSDT', Client.KLINE_INTERVAL_1MINUTE, '26 Jun 2024',  dt_string)
            print("dados coletados")
            data = calculate_indicators(data)
            trading_strategy()
            time.sleep(30)

        except:
            pass


