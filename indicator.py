import ccxt
import pandas as pd
from ta.volatility import BollingerBands
import numpy as np
import simplejson as json
import datetime
import time
import pandas as pd
import numpy as np
import scipy.stats as stats
########################################################################


def read_setting():
    with open('setting.json') as json_file:
        return json.load(json_file)


config = read_setting()


def edit_config(key, value):
    with open('setting.json') as json_file:
        data = json.load(json_file)
        data[key] = value
        json.dump(data, open("setting.json", "w"))

########################################################################


ftx = ccxt.ftx({
    'verbose': False,
    'apiKey': config["apiKey"],
    'secret': config["secret"],
    'enableRateLimit': True,
    'headers': {
        'FTX-SUBACCOUNT': config["sub_account"],
    },
})

bars = ftx.fetch_ohlcv(config['symbol'], timeframe='1d', limit=181)
df = pd.DataFrame(bars[:-35], columns=['timestamp',
                                       'open', 'high', 'low', 'close', 'volume'])
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
df['OHLC'] = (df['open']+df['high']+df['low']+df['close'])/4
list = []
for i in df['OHLC']:
    list.append(i)
data = np.array(list)
x = stats.zscore(data)
print(df)
print(x)
