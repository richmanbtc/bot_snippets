import pandas as pd
import requests
import json

def fetch_fear_greedy():
    url = 'https://api.alternative.me/fng/?limit=3000'
    df = pd.DataFrame(json.loads(requests.get(url).text)['data'])
    df = df[df['time_until_update'].isna()]
    df = df.drop(columns=['time_until_update', 'value_classification'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)
    df['value'] = df['value'].astype('float')
    df = df.sort_values('timestamp')
    df = df.set_index('timestamp')
    df = df.rename(columns={ 'value': 'fear_greedy_index' })
    return df
