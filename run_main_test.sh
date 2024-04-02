#!/bin/bash
python src/cf_search.py --dataset ohiot1dm --horizon 6 --back-horizon 24 --test-group hyper --fraction-std 0.5 --output output.csv

python src/cf_search.py --dataset ohiot1dm --horizon 6 --back-horizon 24 --test-group hypo --fraction-std 0.5 --output output2.csv

python src/cf_search.py --dataset simulated --horizon 10 --back-horizon 40 --test-group hyper --fraction-std 0.5 --output output_simu.csv

python src/cf_search.py --dataset simulated --horizon 10 --back-horizon 40 --test-group hypo --fraction-std 1 --output output_simu2.csv
