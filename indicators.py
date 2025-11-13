import pandas as pd
import numpy as np
import vectorbt as vbt


class TechnicalIndicators:
    """
    A class to calculate technical indicators for trading analysis.
    
    Usage:
        from indicators import TechnicalIndicators
        
        # Initialize
        ti = TechnicalIndicators()
        
        # Calculate indicators from CSV file
        df = ti.calculate_from_csv('data.csv')
        
        # Or calculate from DataFrame
        df = ti.calculate_from_dataframe(your_dataframe)
    """
    
    def __init__(self):
        self.signal_indicators = ['RSI', 'MACD', 'MA', 'HA', 'STOCH', 'BBANDS', 'CCI', 'OBV']
        self.continuous_indicators = ['CMF', 'VWAP', 'ATR', 'VOLATILITY', 'PARKINSON', 'PRICE_ACTION']
    
    def calculate_from_csv(self, csv_file):
        """
        Calculate indicators from a CSV file.
        
        Args:
            csv_file (str): Path to CSV file with OHLCV data
            
        Returns:
            pd.DataFrame: DataFrame with calculated indicators
        """
        df = pd.read_csv(csv_file)
        
        # Handle both 'Datetime' and 'Open Time' column names
        if 'Open Time' in df.columns:
            df['Open Time'] = pd.to_datetime(df['Open Time'])
        elif 'Datetime' in df.columns:
            df['Datetime'] = pd.to_datetime(df['Datetime'])
            df['Open Time'] = df['Datetime']
        
        return self.calculate_from_dataframe(df)
    
    def calculate_from_dataframe(self, df):
        """
        Calculate indicators from a DataFrame.
        
        Args:
            df (pd.DataFrame): DataFrame with OHLCV data (Open, High, Low, Close, Volume)
            
        Returns:
            pd.DataFrame: DataFrame with calculated indicators
        """
        df = df.copy()
        
        # Initialize all strength columns
        for name in self.signal_indicators:
            df[f'{name}_strength'] = 0.0
        
        # Calculate all indicators
        self._calc_RSI(df)
        self._calc_MACD(df)
        self._calc_MA(df)
        self._calc_HA(df)
        self._calc_OBV(df)
        self._calc_STOCH(df)
        self._calc_BBANDS(df)
        self._calc_CCI(df)
        self._calc_CMF(df)
        self._calc_VWAP(df)
        self._calc_ATR(df)
        self._calc_VOLATILITY(df)
        self._calc_PARKINSON(df)
        self._calc_PRICE_ACTION(df)
        
        # Drop intermediate columns
        df.drop(columns=['HA_Close', 'HA_Open', 'HA_High', 'HA_Low',
                        'MA_short', 'MA_long', '%K', '%D', 
                        'BB_lower', 'BB_upper', 'BB_std'], 
               errors='ignore', inplace=True)
        
        # Create signal columns
        self._create_signal_columns(df)
        
        # Drop entry/exit columns
        entry_exit_cols = [f'{ind}_entries' for ind in self.signal_indicators if f'{ind}_entries' in df.columns] + \
                          [f'{ind}_exits' for ind in self.signal_indicators if f'{ind}_exits' in df.columns]
        df.drop(columns=entry_exit_cols, inplace=True)
        
        # Final validation: Ensure all strengths are in [0,1]
        for ind in self.signal_indicators:
            strength_col = f'{ind}_strength'
            if strength_col in df.columns:
                df[strength_col] = df[strength_col].clip(0, 1).fillna(0)
        
        return df
    
    def _calc_RSI(self, df):
        """Calculate RSI indicator"""
        rsi = vbt.RSI.run(df['Close'], window=14)
        df['RSI'] = rsi.rsi
        df['RSI_entries'] = df['RSI'] < 30
        df['RSI_exits'] = df['RSI'] > 70
        
        df.loc[df['RSI_entries'], 'RSI_strength'] = ((30 - df['RSI']) / 30).clip(0, 1)
        df.loc[df['RSI_exits'], 'RSI_strength'] = ((df['RSI'] - 70) / 30).clip(0, 1)
    
    def _calc_MACD(self, df):
        """Calculate MACD indicator"""
        macd = vbt.MACD.run(df['Close'], fast_window=12, slow_window=26, signal_window=9)
        df['MACD_entries'] = macd.macd_crossed_above(macd.signal)
        df['MACD_exits'] = macd.macd_crossed_below(macd.signal)
        
        macd_diff = (macd.macd - macd.signal).abs()
        macd_range = macd_diff.max() - macd_diff.min()
        
        if macd_range > 0:
            norm_macd = ((macd_diff - macd_diff.min()) / macd_range).clip(0, 1)
        else:
            norm_macd = pd.Series(0.0, index=df.index)
        
        signal_mask = df['MACD_entries'] | df['MACD_exits']
        df.loc[signal_mask, 'MACD_strength'] = norm_macd[signal_mask]
    
    def _calc_MA(self, df):
        """Calculate Moving Average indicator"""
        ma_short = vbt.MA.run(df['Close'], window=20).ma
        ma_long = vbt.MA.run(df['Close'], window=50).ma

        df['MA_short'] = ma_short
        df['MA_long'] = ma_long
        df['MA_entries'] = ma_short > ma_long
        df['MA_exits'] = ma_short < ma_long

        ma_diff = (ma_short - ma_long).abs()
        ma_diff_max = ma_diff.max() if ma_diff.max() != 0 else 1
        
        signal_changes = (df['MA_entries'] != df['MA_entries'].shift(1)) | \
                        (df['MA_exits'] != df['MA_exits'].shift(1))
        
        df.loc[signal_changes, 'MA_strength'] = (ma_diff / ma_diff_max).clip(0, 1)
    
    def _calc_HA(self, df):
        """Calculate Heiken Ashi indicator"""
        df["HA_Close"] = (df["Open"] + df["High"] + df["Low"] + df["Close"]) / 4
        df["HA_Open"] = df["Open"]
        
        for i in range(1, len(df)):
            df.loc[i, "HA_Open"] = (df.loc[i - 1, "HA_Open"] + df.loc[i - 1, "HA_Close"]) / 2
        
        df["HA_High"] = df[["High", "HA_Open", "HA_Close"]].max(axis=1)
        df["HA_Low"] = df[["Low", "HA_Open", "HA_Close"]].min(axis=1)
        
        green_flat_bottom = (df["HA_Close"] > df["HA_Open"]) & (df["HA_Low"] == df["HA_Open"])
        red_flat_top = (df["HA_Close"] < df["HA_Open"]) & (df["HA_High"] == df["HA_Open"])
        
        df["HA_entries"] = green_flat_bottom
        df["HA_exits"] = red_flat_top
        
        ha_body = (df["HA_Close"] - df["HA_Open"]).abs()
        ha_range = df["HA_High"] - df["HA_Low"]
        ha_strength = (ha_body / ha_range.replace(0, np.nan)).fillna(0).clip(0, 1)
        
        signal_mask = green_flat_bottom | red_flat_top
        df.loc[signal_mask, "HA_strength"] = ha_strength[signal_mask]
    
    def _calc_OBV(self, df):
        """Calculate On-Balance Volume indicator"""
        obv = vbt.OBV.run(df['Close'], df['Volume'])
        obv_diff = obv.obv.diff()
        
        df['OBV_entries'] = obv_diff > 0
        df['OBV_exits'] = obv_diff < 0
        
        obv_diff_abs = obv_diff.abs()
        obv_max = obv_diff_abs.max()
        
        if obv_max > 0:
            norm_obv = (obv_diff_abs / obv_max).clip(0, 1)
        else:
            norm_obv = pd.Series(0.0, index=df.index)
        
        signal_mask = df['OBV_entries'] | df['OBV_exits']
        df.loc[signal_mask, 'OBV_strength'] = norm_obv[signal_mask]
    
    def _calc_STOCH(self, df):
        """Calculate Stochastic indicator"""
        stoch = vbt.STOCH.run(df['High'], df['Low'], df['Close'])
        df['%K'] = stoch.percent_k
        df['%D'] = stoch.percent_d
        
        df['STOCH_entries'] = df['%K'] > df['%D']
        df['STOCH_exits'] = df['%K'] < df['%D']
        
        stoch_strength = ((df['%K'] - df['%D']).abs() / 100).clip(0, 1)
        
        signal_changes = (df['STOCH_entries'] != df['STOCH_entries'].shift(1)) | \
                        (df['STOCH_exits'] != df['STOCH_exits'].shift(1))
        
        df.loc[signal_changes, 'STOCH_strength'] = stoch_strength[signal_changes]
    
    def _calc_BBANDS(self, df):
        """Calculate Bollinger Bands indicator"""
        bb = vbt.BBANDS.run(df['Close'])
        df['BB_lower'] = bb.lower
        df['BB_upper'] = bb.upper
        df['BB_std'] = bb.bandwidth
        
        df['BBANDS_entries'] = df['Close'] < df['BB_lower']
        df['BBANDS_exits'] = df['Close'] > df['BB_upper']
        
        bb_std_safe = df['BB_std'].replace(0, np.nan).fillna(1)
        
        entry_strength = ((df['BB_lower'] - df['Close']) / bb_std_safe).clip(0, 1)
        exit_strength = ((df['Close'] - df['BB_upper']) / bb_std_safe).clip(0, 1)
        
        df.loc[df['BBANDS_entries'], 'BBANDS_strength'] = entry_strength[df['BBANDS_entries']]
        df.loc[df['BBANDS_exits'], 'BBANDS_strength'] = exit_strength[df['BBANDS_exits']]
    
    def _calc_CCI(self, df, window=20, constant=0.015):
        """Calculate Commodity Channel Index indicator"""
        tp = (df['High'] + df['Low'] + df['Close']) / 3
        tp_ma = tp.rolling(window=window).mean()
        tp_md = tp.rolling(window=window).apply(
            lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
        )
        
        tp_md_safe = tp_md.replace(0, np.nan).fillna(1)
        cci = (tp - tp_ma) / (constant * tp_md_safe)
        
        df['CCI_entries'] = (cci < -100) | (cci > 100)
        df['CCI_exits'] = ((cci > -50) & (cci < 0)) | ((cci < 50) & (cci > 0))
        
        norm_cci = (cci.clip(-300, 300) / 300 * 0.5 + 0.5).clip(0, 1)
        
        signal_mask = df['CCI_entries'] | df['CCI_exits']
        df.loc[signal_mask, 'CCI_strength'] = norm_cci[signal_mask]
    
    def _calc_CMF(self, df, window=20):
        """Calculate Chaikin Money Flow indicator"""
        mfm = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / \
              (df['High'] - df['Low']).replace(0, np.nan)
        mfv = mfm * df['Volume']
        df['CMF'] = (mfv.rolling(window).sum() / 
                     df['Volume'].rolling(window).sum()).clip(-1, 1).fillna(0)
    
    def _calc_VWAP(self, df):
        """Calculate Volume Weighted Average Price indicator"""
        df['VWAP'] = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()
    
    def _calc_ATR(self, df, window=14):
        """Calculate Average True Range indicator"""
        tr = df[['High', 'Close']].max(axis=1) - df[['Low', 'Close']].min(axis=1)
        df['ATR'] = tr.rolling(window).mean()
    
    def _calc_VOLATILITY(self, df, window=20):
        """Calculate Volatility indicator"""
        returns = df['Close'].pct_change()
        df['VOLATILITY'] = returns.rolling(window).std()
    
    def _calc_PARKINSON(self, df, window=20):
        """Calculate Parkinson Volatility indicator"""
        parkinson = (1 / (4 * np.log(2))) * (np.log(df['High'] / df['Low']) ** 2)
        df['PARKINSON'] = parkinson.rolling(window).mean()
    
    def _calc_PRICE_ACTION(self, df, window=20):
        """Calculate Price Action indicator"""
        high = df['High'].rolling(window).max()
        low = df['Low'].rolling(window).min()
        df['dist_from_high'] = (df['Close'] - high) / high
        df['dist_from_low'] = (df['Close'] - low) / low
        body = (df['Close'] - df['Open']).abs()
        rng = df['High'] - df['Low']
        df['PRICE_ACTION'] = (body / rng.replace(0, np.nan)).fillna(0).clip(0, 1)
    
    def _create_signal_columns(self, df):
        """Create signal columns from entry/exit columns"""
        for ind in self.signal_indicators:
            entry_col = f'{ind}_entries'
            exit_col = f'{ind}_exits'
            signal_col = f'{ind}_signal'

            if entry_col not in df.columns or exit_col not in df.columns:
                print(f"Skipping {ind}: missing columns")
                continue

            df[signal_col] = 0
            df.loc[df[entry_col], signal_col] = 1  
            df.loc[df[exit_col] & (df[signal_col] != 1), signal_col] = -1


# Convenience function for backward compatibility
def live_indicators(csv_file):
    """
    Legacy function wrapper for backward compatibility.
    
    Args:
        csv_file (str): Path to CSV file
        
    Returns:
        pd.DataFrame: DataFrame with calculated indicators
    """
    ti = TechnicalIndicators()
    return ti.calculate_from_csv(csv_file)