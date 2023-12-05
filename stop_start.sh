#!/bin/bash

chmod 777 -R *
mkdir user_data/backtest_results
mkdir user_data/data
mkdir user_data/data/kucoin
mkdir user_data/freqmodels
mkdir user_data/hyperopt_results
mkdir user_data/hyperopts
mkdir user_data/logs
mkdir user_data/notebooks
mkdir user_data/plot
mkdir user_data/strategies
chmod 777 -R user_data

docker-compose down
docker-compose up -d
sh ./download_data_backtest.sh
docker-compose logs -f
