from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.strategy import merge_informative_pair, DecimalParameter, IntParameter

class Elliot(): 

    base_nb_candles_sell = 22
    base_nb_candles_buy = 12
    ewo_low = 10.289
    ewo_high = 3.001
    fast_ewo = 50
    slow_ewo = 200
    low_offset = 0.987
    rsi_buy = 58
    high_offset = 1.014
    buy_ema_cofi = 0.97
    buy_fastk = 20
    buy_fastd = 20
    buy_adx = 30
    buy_ewo_high = 3.55


    def use_hyperopts(self, base_nb_candles_sell, base_nb_candles_buy, low_offset, ewo_high, ewo_low, rsi_buy, high_offset, buy_ema_cofi, buy_fastk, buy_fastd, buy_adx, buy_ewo_high):
        self.base_nb_candles_sell = base_nb_candles_sell
        self.base_nb_candles_buy = base_nb_candles_buy
        self.low_offset = low_offset
        self.ewo_high = ewo_high
        self.ewo_low = ewo_low
        self.rsi_buy = rsi_buy
        self.high_offset = high_offset
        self.buy_ema_cofi = buy_ema_cofi
        self.buy_fastk = buy_fastk
        self.buy_fastd = buy_fastd
        self.buy_adx = buy_adx
        self.buy_ewo_high = buy_ewo_high

    def populate_indicators(self, dataframe: DataFrame):
        """
        Populates the given dataframe with additional indicators.

        Parameters:
        dataframe (DataFrame): The dataframe to populate with indicators.

        Returns:
        DataFrame: The dataframe with populated indicators.
        """
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['rsi_fast'] = ta.RSI(dataframe, timeperiod=4)

        dataframe['EWO'] = self.__EWO(dataframe, self.fast_ewo, self.slow_ewo)
        # dataframe['rsi_slow'] = ta.RSI(dataframe, timeperiod=20)

        for val in self.base_nb_candles_buy.range:
            dataframe[f'ma_buy_{val}'] = ta.EMA(dataframe, timeperiod=val)
        for val in self.base_nb_candles_sell.range:
            dataframe[f'ma_sell_{val}'] = ta.EMA(dataframe, timeperiod=val)

        return dataframe

    def populate_entry_trend_v1(self, dataframe: DataFrame, conditions: list):
        buy1ewo = (
                (dataframe['rsi_fast'] < 35) &
                (dataframe['close'] < (dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'] * self.low_offset.value)) &
                (dataframe['EWO'] > self.ewo_high.value) &
                (dataframe['rsi'] < self.rsi_buy.value) &
                (dataframe['volume'] > 0) &
                (dataframe['close'] < (
                        dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset.value))
        )
        dataframe.loc[buy1ewo, 'enter_tag'] += 'buy1eworsi_'
        conditions.append(buy1ewo)
        return (dataframe, conditions)
    
    def populate_entry_trend_v2(self, dataframe: DataFrame, conditions: list):
        buy2ewo = (
                (dataframe['rsi_fast'] < 35) &
                (dataframe['close'] < (dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'] * self.low_offset.value)) &
                (dataframe['EWO'] < self.ewo_low.value) &
                (dataframe['volume'] > 0) &
                (dataframe['close'] < (
                        dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset.value))
        )
        dataframe.loc[buy2ewo, 'enter_tag'] += 'buy2ewo_'
        conditions.append(buy2ewo)
        return (dataframe, conditions)
    
    def populate_entry_trend_cofi(self, dataframe: DataFrame, conditions: list):
        is_cofi = (
                (dataframe['open'] < dataframe['ema_8'] * self.buy_ema_cofi.value) &
                (qtpylib.crossed_above(dataframe['fastk'], dataframe['fastd'])) &
                (dataframe['fastk'] < self.buy_fastk.value) &
                (dataframe['fastd'] < self.buy_fastd.value) &
                (dataframe['adx'] > self.buy_adx.value) &
                (dataframe['EWO'] > self.buy_ewo_high.value)
        )
        dataframe.loc[is_cofi, 'enter_tag'] += 'cofi_'
        conditions.append(is_cofi)
        return (dataframe, conditions)

    def __EWO(self, dataframe: DataFrame, ema_length=5, ema2_length=3):
        df = dataframe.copy()
        ema1 = ta.EMA(df, timeperiod=ema_length)
        ema2 = ta.EMA(df, timeperiod=ema2_length)
        return (ema1 - ema2) / df['close'] * 100