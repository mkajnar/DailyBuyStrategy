#!/bin/sh

docker exec -it freqtrade_bt freqtrade download-data --timeframes 1m 5m 15m 1h --config user_data/config.json --timerange 20230101-
