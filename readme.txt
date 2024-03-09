1 - To create user_data
reqtrade create-userdir --userdir user_data

2 - Copy strategies files from root directory into user_data/strategies/
cp *.py user_data/strategies/

3 - List strategies to verify
freqtrade list-strategies 

4 - Download backtest data
freqtrade download-data --exchange binance -t 5m --days 120 --config config.json
freqtrade download-data --exchange binance -t 15m --days 120 --config config.json
freqtrade download-data --exchange binance -t 30m --days 120 --config config.json
freqtrade download-data --exchange binance -t 1h --days 120 --config config.json
freqtrade download-data --exchange binance -t 4h --days 120 --config config.json

5 - Backtesting
6 - Hyperopt