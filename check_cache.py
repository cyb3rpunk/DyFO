import pickle
import glob
import os

for f in glob.glob('d:/projetos/DyFO/results/prepared_data_cache_*.pkl'):
    try:
        with open(f, 'rb') as fh:
            data = pickle.load(fh)
            
        print(f'File: {os.path.basename(f)}')
        print(f'  Keys: {list(data.keys())}')
        if 'sorted_dates' in data:
            print(f'  Dates: {min(data["sorted_dates"])} to {max(data["sorted_dates"])}')
        if 'ticker_to_idx' in data:
            print(f'  Tickers: {len(data["ticker_to_idx"])}')
        print('-' * 40)
    except Exception as e:
        print(f'Error reading {f}: {e}')
