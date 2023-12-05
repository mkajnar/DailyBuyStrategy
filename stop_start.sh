#!/bin/bash

base_directory="user_data"

if [ ! -d "$base_directory" ]; then
    mkdir $base_directory
fi

directories=(
    "backtest_results"
    "data"
    "freqmodels"
    "hyperopt_results"
    "hyperopts"
    "logs"
    "notebooks"
    "plot"
    "strategies"
	"data/kucoin"
)

for dir in "${directories[@]}"; do
    if [ ! -d "$base_directory/$dir" ]; then
        mkdir "$base_directory/$dir"
    fi
done

chmod 777 -R user_data

docker-compose down
docker-compose up -d
docker-compose logs -f




