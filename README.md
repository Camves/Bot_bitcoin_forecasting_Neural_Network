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
