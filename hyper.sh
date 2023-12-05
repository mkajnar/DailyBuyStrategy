#!/bin/sh

docker exec -it freqtrade freqtrade hyperopt --strategy HPStartegyDCA --config user_data/config.json --hyperopt-loss SharpeHyperOptLossDaily --spaces buy sell roi trailing stoploss -e 3000 --timerange=20230101-
