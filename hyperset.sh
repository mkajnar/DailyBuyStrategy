#!/bin/bash

# Exportuje hyperopt list
rm user_data/output.csv
docker exec -it freqtrade freqtrade hyperopt-list --export-csv /freqtrade/user_data/output.csv

# Získá řádek s největším celkovým ziskem
BEST_EPOCH=$(awk -F ',' '{print $2 "," $6}' user_data/output.csv | sort -t ',' -k2 -nr | head -n 1 | cut -d',' -f1)

# Pokud není BEST_EPOCH prázdný, spustí hyperopt-show s číslem nejlepší epochy
if [ -n "$BEST_EPOCH" ]; then
    docker exec -it freqtrade freqtrade hyperopt-show -n $BEST_EPOCH
else
    echo "Nepodařilo se najít nejlepší epochu!"
fi
