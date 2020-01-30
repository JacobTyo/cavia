#!/usr/bin/env bash

# update the logger so I can load the object on other computers
for i in $(ls ./cva_sine_result_files/); do
    python fixOldLoggers.py --path ./cva_sine_result_files/${i} ;
done
