# Projeto de Previsão de Preços de Criptomoedas com LSTM

Este projeto implementa um sistema de previsão de preços de criptomoedas utilizando redes neurais LSTM e a API da Binance. A estratégia de negociação é baseada em previsões de preços utilizando indicadores técnicos.

## Pré-requisitos

- Python 3.8+
- Biblioteca `numpy`
- Biblioteca `pandas`
- Biblioteca `binance`
- Biblioteca `scikit-learn`
- Biblioteca `tensorflow`
- Biblioteca `matplotlib`
- Biblioteca `talib`

## Instalação

1. Clone o repositório:

    ```bash
    git clone https://github.com/Camves/Bot_bitcoin_forecasting_Neural_Network
    cd Bot_bitcoin_forecasting_Neural_Network
    ```

2. Instale as dependências:

    ```bash
    pip install numpy pandas binance scikit-learn tensorflow matplotlib ta-lib
    ```

## Configuração

1. Configure a sua chave da API da Binance:

    ```python
    api_key = 'SUA_API_KEY'
    api_secret = 'SEU_API_SECRET'
    ```

2. Certifique-se de que a GPU está configurada corretamente para alocação de memória, se disponível.

## Uso

Execute o script para iniciar a coleta de dados e a negociação:

```bash
python crypto.py
```

O script coletará dados históricos do par BTC/USDT da Binance, calculará indicadores técnicos, treinará um modelo LSTM e realizará previsões para decisões de compra e venda.

## Exemplo de Saída

```
GPU Available:  True
1 Physical GPUs, 1 Logical GPUs
coletando dados
dados coletados
Train x shape: (N, 30, 10)
Train y shape: (N,)
Test x shape: (M, 30, 10)
Test y shape: (M,)
Epoch 1/10
...
Mean Squared Error: X.XXXX
Mean Absolute Error: Y.YYYY

```

## Estrutura do Código
- get_historical_data(symbol, interval, start, end): Função para obter dados históricos da Binance.
- calculate_indicators(data): Função para calcular indicadores técnicos como SMA, RSI, Bandas de Bollinger e MACD.
- prepare_data(data, look_back, scaler, is_train): Função para preparar dados de treinamento e teste.
- predict_prices(model, data, look_back, scaler): Função para prever preços usando o modelo treinado.
- buy(symbol, quantity): Função para comprar um ativo.
- sell(symbol, quantity): Função para vender um ativo.
- trading_strategy(): Estratégia de negociação baseada em previsões de preços.
- main(): Função principal que coleta dados, calcula indicadores, treina o modelo e realiza a estratégia de negociação.
- 
## Contribuição
Sinta-se à vontade para fazer um fork do projeto, abrir issues ou pull requests.

## Licença
Este projeto está licenciado sob a licença MIT. Veja o arquivo LICENSE para mais detalhes.
