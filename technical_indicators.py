import numpy as np

def get_sma(df):
    
    # Simple moving average
    df['sma_10'] = df['Close'].rolling(window = 10).mean()
    df['sma_20'] = df['Close'].rolling(window = 20).mean()
    df['sma_50'] = df['Close'].rolling(window = 50).mean()
    df['sma_100'] = df['Close'].rolling(window = 100).mean()
    df['sma_200'] = df['Close'].rolling(window = 200).mean()

    # Get the sma on the previous day
    df['close_lag_1'] = df['Close'].shift(1)
    df['sma_10_lag_1'] = df['sma_10'].shift(1)
    df['sma_20_lag_1'] = df['sma_20'].shift(1)
    df['sma_50_lag_1'] = df['sma_50'].shift(1)
    df['sma_100_lag_1'] = df['sma_100'].shift(1)
    df['sma_200_lag_1'] = df['sma_200'].shift(1)

    # Get above signals
    df['sma_above20'] = (df['Close'] > df['sma_20']).astype(int)
    df['sma_above50'] = (df['Close'] > df['sma_50']).astype(int)
    df['sma_above100'] = (df['Close'] > df['sma_100']).astype(int)
    df['sma_above200'] = (df['Close'] > df['sma_200']).astype(int)

    df['sma_10above20'] = (df['sma_10'] > df['sma_20']).astype(int)
    df['sma_10above50'] = (df['sma_10'] > df['sma_50']).astype(int)
    df['sma_10above100'] = (df['sma_10'] > df['sma_100']).astype(int)
    df['sma_10above200'] = (df['sma_10'] > df['sma_200']).astype(int)

    # Get bullish crossover signals
    df['sma_cut20'] = ((df['close_lag_1'] < df['sma_20_lag_1']) & (df['sma_above20']==True)).astype(int)
    df['sma_cut50'] = ((df['close_lag_1'] < df['sma_50_lag_1']) & (df['sma_above50']==True)).astype(int)
    df['sma_cut100'] = ((df['close_lag_1'] < df['sma_100_lag_1']) & (df['sma_above100']==True)).astype(int)
    df['sma_cut200'] = ((df['close_lag_1'] < df['sma_200_lag_1']) & (df['sma_above200']==True)).astype(int)

    df['sma_10cut20'] = ((df['sma_10_lag_1'] < df['sma_20_lag_1']) & (df['sma_10above20']==True)).astype(int)
    df['sma_10cut50'] = ((df['sma_10_lag_1'] < df['sma_50_lag_1']) & (df['sma_10above50']==True)).astype(int)
    df['sma_10cut100'] = ((df['sma_10_lag_1'] < df['sma_100_lag_1']) & (df['sma_10above100']==True)).astype(int)
    df['sma_10cut200'] = ((df['sma_10_lag_1'] < df['sma_200_lag_1']) & (df['sma_10above200']==True)).astype(int)

    # Get bearish crossover signals
    df['sma_cut20down'] = ((df['close_lag_1'] > df['sma_20_lag_1']) & (df['Close'] < df['sma_20'])).astype(int)
    df['sma_cut50down'] = ((df['close_lag_1'] > df['sma_50_lag_1']) & (df['Close'] < df['sma_50'])).astype(int)
    df['sma_cut100down'] = ((df['close_lag_1'] > df['sma_100_lag_1']) & (df['Close'] < df['sma_100'])).astype(int)
    df['sma_cut200down'] = ((df['close_lag_1'] > df['sma_200_lag_1']) & (df['Close'] < df['sma_200'])).astype(int)

    df['sma_10cut20down'] = ((df['sma_10_lag_1'] > df['sma_20_lag_1']) & (df['sma_10'] < df['sma_20'])).astype(int)
    df['sma_10cut50down'] = ((df['sma_10_lag_1'] > df['sma_50_lag_1']) & (df['sma_10'] < df['sma_50'])).astype(int)
    df['sma_10cut100down'] = ((df['sma_10_lag_1'] > df['sma_100_lag_1']) & (df['sma_10'] < df['sma_100'])).astype(int)
    df['sma_10cut200down'] = ((df['sma_10_lag_1'] > df['sma_200_lag_1']) & (df['sma_10'] < df['sma_200'])).astype(int)
    
    # Del unneccesary cols
    df.drop(['sma_10', 'sma_50', 'sma_100', 'sma_200', 'sma_10_lag_1', 'sma_20_lag_1', 'sma_50_lag_1', 'sma_100_lag_1', 'sma_200_lag_1'], axis=1, inplace=True)
    return df

def get_ema(df):

    # Exponential moving average
    df['ema_10'] = df['Close'].ewm(span = 10, adjust=False).mean()
    df['ema_20'] = df['Close'].ewm(span = 20, adjust=False).mean()
    df['ema_50'] = df['Close'].ewm(span = 50, adjust=False).mean()
    df['ema_100'] = df['Close'].ewm(span = 100, adjust=False).mean()
    df['ema_200'] = df['Close'].ewm(span = 200, adjust=False).mean()

    # Get the ema on the previous day
    df['close_lag_1'] = df['Close'].shift(1)
    df['ema_10_lag_1'] = df['ema_10'].shift(1)
    df['ema_20_lag_1'] = df['ema_20'].shift(1)
    df['ema_50_lag_1'] = df['ema_50'].shift(1)
    df['ema_100_lag_1'] = df['ema_100'].shift(1)
    df['ema_200_lag_1'] = df['ema_200'].shift(1)

    # Get above signals
    df['ema_above20'] = (df['Close'] > df['ema_20']).astype(int)
    df['ema_above50'] = (df['Close'] > df['ema_50']).astype(int)
    df['ema_above100'] = (df['Close'] > df['ema_100']).astype(int)
    df['ema_above200'] = (df['Close'] > df['ema_200']).astype(int)

    df['ema_10above20'] = (df['ema_10'] > df['ema_20']).astype(int)
    df['ema_10above50'] = (df['ema_10'] > df['ema_50']).astype(int)
    df['ema_10above100'] = (df['ema_10'] > df['ema_100']).astype(int)
    df['ema_10above200'] = (df['ema_10'] > df['ema_200']).astype(int)

    # Get bullish crossover signals
    df['ema_cut20'] = ((df['close_lag_1'] < df['ema_20_lag_1']) & (df['ema_above20']==True)).astype(int)
    df['ema_cut50'] = ((df['close_lag_1'] < df['ema_50_lag_1']) & (df['ema_above50']==True)).astype(int)
    df['ema_cut100'] = ((df['close_lag_1'] < df['ema_100_lag_1']) & (df['ema_above100']==True)).astype(int)
    df['ema_cut200'] = ((df['close_lag_1'] < df['ema_200_lag_1']) & (df['ema_above200']==True)).astype(int)

    df['ema_10cut20'] = ((df['ema_10_lag_1'] < df['ema_20_lag_1']) & (df['ema_10above20']==True)).astype(int)
    df['ema_10cut50'] = ((df['ema_10_lag_1'] < df['ema_50_lag_1']) & (df['ema_10above50']==True)).astype(int)
    df['ema_10cut100'] = ((df['ema_10_lag_1'] < df['ema_100_lag_1']) & (df['ema_10above100']==True)).astype(int)
    df['ema_10cut200'] = ((df['ema_10_lag_1'] < df['ema_200_lag_1']) & (df['ema_10above200']==True)).astype(int)

    # Get bearish crossover signals
    df['ema_cut20down'] = ((df['close_lag_1'] > df['ema_20_lag_1']) & (df['Close'] < df['ema_20'])).astype(int)
    df['ema_cut50down'] = ((df['close_lag_1'] > df['ema_50_lag_1']) & (df['Close'] < df['ema_50'])).astype(int)
    df['ema_cut100down'] = ((df['close_lag_1'] > df['ema_100_lag_1']) & (df['Close'] < df['ema_100'])).astype(int)
    df['ema_cut200down'] = ((df['close_lag_1'] > df['ema_200_lag_1']) & (df['Close'] < df['ema_200'])).astype(int)

    df['ema_10cut20down'] = ((df['ema_10_lag_1'] > df['ema_20_lag_1']) & (df['ema_10'] < df['ema_20'])).astype(int)
    df['ema_10cut50down'] = ((df['ema_10_lag_1'] > df['ema_50_lag_1']) & (df['ema_10'] < df['ema_50'])).astype(int)
    df['ema_10cut100down'] = ((df['ema_10_lag_1'] > df['ema_100_lag_1']) & (df['ema_10'] < df['ema_100'])).astype(int)
    df['ema_10cut200down'] = ((df['ema_10_lag_1'] > df['ema_200_lag_1']) & (df['ema_10'] < df['ema_200'])).astype(int)
    
    
    # Del unneccesary cols
    df.drop(['ema_10', 'ema_20', 'ema_50', 'ema_100', 'ema_200', 'ema_10_lag_1', 'ema_20_lag_1', 'ema_50_lag_1', 'ema_100_lag_1', 'ema_200_lag_1'], axis=1, inplace=True)
    return df

def get_macd(df):

    # Exponential moving average
    df['ema_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['ema_26'] = df['Close'].ewm(span=26, adjust=False).mean()

    # MACD
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()

    # Get the macd on the previous day
    df['macd_lag_1'] = df['macd'].shift(1)
    df['macd_signal_lag_1'] = df['macd_signal'].shift(1)

    # Get bullish crossover signals
    df['macd_crossover'] = ((df['macd_lag_1'] < df['macd_signal_lag_1']) & (df['macd'] > df['macd_signal'])).astype(int)

    # Get bearish crossover signals
    df['macd_crossoverdown'] = ((df['macd_lag_1'] > df['macd_signal_lag_1']) & (df['macd'] < df['macd_signal'])).astype(int)
    
    
    # Del unneccesary cols
    df.drop(['ema_12', 'ema_26', 'macd_signal', 'macd_lag_1', 'macd_signal_lag_1'], axis=1, inplace=True)
    return df

def get_stochastic_osc(df):

    # Generate fast and slow stochastic oscillators
    df['lowest_14'] = df['Close'].rolling(window = 14).min()
    df['highest_14'] = df['Close'].rolling(window = 14).max()
    df['stochastic_fast'] = 100.0*(df['Close'] - df['lowest_14'])/(df['highest_14'] - df['lowest_14'])
    df['stochastic_slow'] = df['stochastic_fast'].rolling(window = 3).mean()

    # Get the stochastics on the previous day
    df['stochastic_fast_lag_1'] = df['stochastic_fast'].shift(1)
    df['stochastic_slow_lag_1'] = df['stochastic_slow'].shift(1)

    # Get bullish crossover signals
    df['stochastic_fastcutslow'] = ((df['stochastic_fast_lag_1'] < df['stochastic_slow_lag_1']) & (df['stochastic_fast'] > df['stochastic_slow'])).astype(int)

    # Get bearish crossover signals
    df['stochastic_fastcutslowdown'] = ((df['stochastic_fast_lag_1'] > df['stochastic_slow_lag_1']) & (df['stochastic_fast'] < df['stochastic_slow'])).astype(int)

    # Get overbought/oversold signals
    df['stochastic_overs'] = (df['stochastic_fast'] < 20).astype(int)
    df['stochastic_overb'] = (df['stochastic_fast'] > 80).astype(int)
    
    # Del unneccesary cols
    df.drop(['lowest_14', 'highest_14', 'stochastic_fast_lag_1', 'stochastic_slow_lag_1'], axis=1, inplace=True)
    return df

def get_rsi(df):
    
    # Create returns column
    df['daily_ret'] = 100.0 * ((df['Close'] / df['Close'].shift(1)) - 1)
    
    # Get gain and loss columns
    df['gain'] = df['daily_ret']
    df.loc[df['gain']<0, 'gain'] = 0

    df['loss'] = df['daily_ret']
    df.loc[df['loss']>0, 'loss'] = 0
    df['loss'] = abs(df['loss'])

    # Get avg_gain, avg_loss columns
    df['avg_gain'] = df['gain'].rolling(window = 14).mean()*13
    df['avg_loss'] = df['loss'].rolling(window = 14).mean()*13
    
    df.iloc[15:,-2] += df.iloc[15:,-4]
    df['avg_gain'] /= 14
    df.iloc[15:,-1] += df.iloc[15:,-3]
    df['avg_loss'] /= 14
    
    df['rsi'] = 100 - 100/(1+(df['avg_gain']/df['avg_loss']))
    df['rsi_ob'] = (df['rsi'] > 70).astype(int)
    df['rsi_os'] = (df['rsi'] < 30).astype(int)
    
    # Del unneccesary cols
    df.drop(['gain', 'loss'], axis=1, inplace=True)
    return df

def get_atr(df):

    # Get atr
    df['close_lag_1'] = df['Close'].shift(1)
    df['tr'] = np.maximum(df['High']-df['Low'], np.maximum(abs(df['High']-df['close_lag_1']), abs(df['Low']-df['close_lag_1'])))
    df['atr'] = df['tr'].rolling(window = 14).mean()

    # Get bullish atr signal - buy when next day's price is above yesterday's closing + atr
    df['atr_signal'] = (df['High'] > ((df['close_lag_1'] + df['atr']))).astype(int)

    # Get bearish atr signal - sell when next day's price is below yesterday's closing - atr
    df['atr_signaldown'] = (df['Low'] < ((df['close_lag_1'] - df['atr']))).astype(int)

    # Del unneccesary cols
    df.drop(['close_lag_1'], axis=1, inplace=True)
    return df

def get_adx(df):

    df['high_lag_1'] = df['High'].shift(1)
    df['low_lag_1'] = df['Low'].shift(1)

    def comp_pdm(high, high_lag_1, low, low_lag_1):
        if (high-high_lag_1) > (low_lag_1-low):
            return(max(high-high_lag_1, 0))
        else:
            return 0

    df['+dm'] = df.apply(lambda row: comp_pdm(row['High'], row['high_lag_1'], row['Low'], row['low_lag_1']), axis=1)

    def comp_mdm(high, high_lag_1, low, low_lag_1):
        if (low_lag_1-low) > (high-high_lag_1):
            return(max(low_lag_1-low, 0))
        else:
            return 0

    df['-dm'] = df.apply(lambda row: comp_mdm(row['High'], row['high_lag_1'], row['Low'], row['low_lag_1']), axis=1)

    # Get smoothed +/- directional movement
    df['smoothed+dm'] = 0
    df.loc[df.index[14], 'smoothed+dm'] = df[1:15]['+dm'].sum()
    df.loc[df.index[15]:,'smoothed+dm'] = df.loc[df.index[14]:df.index[-2],'smoothed+dm'].to_numpy()*13/14 + df.loc[df.index[15]:,'+dm']
    
    df['smoothed-dm'] = 0
    df.loc[df.index[14], 'smoothed-dm'] = df[1:15]['-dm'].sum()
    df.loc[df.index[15]:,'smoothed-dm'] = df.loc[df.index[14]:df.index[-2],'smoothed-dm'].to_numpy()*13/14 + df.loc[df.index[15]:,'-dm']

    df['14tr'] = 0
    df.loc[df.index[14], '14tr'] = df[1:15]['tr'].sum()
    df.loc[df.index[15]:,'14tr'] = df.loc[df.index[14]:df.index[-2],'14tr'].to_numpy()*13/14 + df.loc[df.index[15]:,'tr']

    # Get +/- directional index
    df['+di'] = 100.0*(df['smoothed+dm']/df['14tr'])
    df['-di'] = 100.0*(df['smoothed-dm']/df['14tr'])

    # Get directional movement index
    df['dx'] = 100.0 * (abs(df['+di']-df['-di'])/abs(df['+di']+df['-di']))

    # Get average directional movement index
    df.loc[df.index[27], 'adx'] = df[14:28]['dx'].mean()
    df.loc[df.index[28]:,'adx'] = df.loc[df.index[27]:df.index[-2],'adx'].to_numpy()*13/14 + df.loc[df.index[28]:,'dx']/14

    # Get adx strength and trendless
    df['adx_strength'] = (df['adx'] > 25).astype(int)
    df['adx_trendless'] = (df['adx'] < 20).astype(int)

    # Get adx signals
    df['+di_lag_1'] = df['+di'].shift(1)
    df['-di_lag_1'] = df['-di'].shift(1)
    df['adx_bull'] = ((df['+di_lag_1'] < df['-di_lag_1']) & (df['+di'] > df['-di']) & (df['adx_strength']==True)).astype(int)
    df['adx_bear'] = ((df['-di_lag_1'] < df['+di_lag_1']) & (df['-di'] > df['+di']) & (df['adx_strength']==True)).astype(int)

    # Del unneccesary cols
    df.drop(['+di', '-di', '+di_lag_1', '-di_lag_1', 'tr', '+dm', '-dm', 'smoothed+dm', 'smoothed-dm', '14tr', 'dx', 'tr'], axis=1, inplace=True)
    return df

def get_bollinger_bands(df):

    df['std_20'] = df['Close'].rolling(window = 20).std()
    df['bollinger_upp'] = df['sma_20'] + 2*df['std_20']
    df['bollinger_low'] = df['sma_20'] - 2*df['std_20']

    # Get dist between bollinger bands and the price
    df['bollinger_upp_dist'] = df['bollinger_upp'] - df['Close']
    df['bollinger_low_dist'] = df['Close'] - df['bollinger_low']

    # Get signals
    df['bollinger_ob'] = (df['Close'] > df['bollinger_upp']).astype(int)
    df['bollinger_os'] = (df['Close'] < df['bollinger_low']).astype(int)
    
    # Del unneccesary cols
    df.drop(['std_20', 'bollinger_upp', 'bollinger_low'], axis=1, inplace=True)
    return df