#!/bin/sh

docker exec -it freqtrade freqtrade download-data --timeframes 1m 5m 15m 1h --timerange 20230101-
