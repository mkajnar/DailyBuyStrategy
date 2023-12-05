#!/bin/sh

docker exec -it freqtrade_bt freqtrade hyperopt --strategy HPStrategy --config user_data/config.json --hyperopt-loss SharpeHyperOptLossDaily --spaces buy sell roi trailing stoploss -e 3000 --timerange=20230101-
