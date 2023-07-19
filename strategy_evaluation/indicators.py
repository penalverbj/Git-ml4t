import pandas as pd


def author():
    return 'jpb6'


'''
Indicators I will use:
EMA
MACD - uses ema
Boilenger Bands - uses sma
SMA
CCI
'''


def sma(data, window):
    mean = data.rolling(window=window, center=False).mean()
    mean.rename(columns={mean.columns[0]: "mean"}, inplace=True)
    return mean


def bolinger_bands(data, window, threshold, symbol):
    avg = sma(data, window)
    std = data.rolling(window=window, center=False).std()

    upper_band = pd.DataFrame(avg['mean'] + threshold * std[symbol])
    upper_band.rename(columns={upper_band.columns[0]: "upper"}, inplace=True)

    lower_band = pd.DataFrame(avg['mean'] - threshold * std[symbol])
    lower_band.rename(columns={lower_band.columns[0]: "lower"}, inplace=True)

    df_out = pd.concat([avg, upper_band, lower_band], axis=1)
    return df_out


def ema(data, window):
    ema = data.ewm(com=(window - 1) / 2).mean()
    ema.rename(columns={ema.columns[0]: "ema"}, inplace=True)
    return ema


def macd(data, window_ema1, window_ema2, window_signal):
    ema1 = ema(data, window_ema1)
    ema2 = ema(data, window_ema2)

    macd = ema1 - ema2
    macd.rename(columns={macd.columns[0]: "macd"}, inplace=True)

    signal = ema(macd, window_signal)
    signal.rename(columns={signal.columns[0]: "signal"}, inplace=True)

    df_out = pd.concat([macd, signal], axis=1)
    return df_out


def cci(data, window, symbol):
    max = data.rolling(window).max()
    min = data.rolling(window).min()
    close = data.rolling(window).agg(lambda rows: rows[-1])
    tp = (max + min + close) / 3
    avg = sma(tp, window)
    mad = data.rolling(window).apply(lambda x: pd.Series(x).mad())
    nom = pd.DataFrame(tp[symbol] - avg['mean'])
    nom.rename(columns={nom.columns[0] : 'nom'}, inplace=True)
    cci = pd.DataFrame(nom['nom'] / (.015 * mad[symbol]))
    cci.rename(columns={cci.columns[0]: "cci"}, inplace=True)
    return cci
