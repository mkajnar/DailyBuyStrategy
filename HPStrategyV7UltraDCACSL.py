import logging
import os
import sys
from functools import reduce

import numpy
import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame
from typing import Optional, Union, List

from pandas_ta import stdev

from freqtrade.enums import ExitCheckTuple
from freqtrade.persistence import Trade, Order
from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter, IStrategy, IntParameter,
                                informative)
from datetime import timedelta, datetime, timezone
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import pandas_ta as pta


class HPStrategyV7UltraDCACSL(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = '5m'
    leverage_value = 3
    stoploss = -1

    csl = {}
    minimal_roi = {
        "0": 0.007 * leverage_value
    }

    allprofits = {}
    candle_open_prices = {}

    process_only_new_candles = True
    startup_candle_count = 50

    trailing_stop = True
    trailing_only_offset_is_reached = False
    trailing_stop_positive = 0.003 * leverage_value
    trailing_stop_positive_offset = 0.007 * leverage_value

    use_exit_signal = True
    use_custom_stoploss = True
    ignore_roi_if_entry_signal = False
    position_adjustment_enable = True

    dca_threshold_pct_k = DecimalParameter(0.90, 0.99, default=0.94, decimals=2, space='buy',
                                           optimize=position_adjustment_enable)
    dca_threshold_pct = DecimalParameter(0.1, 0.5, default=0.3 / leverage_value, decimals=2, space='buy',
                                         optimize=position_adjustment_enable)

    dca_multiplier = DecimalParameter(0.5, 3, default=2.71, decimals=2, space='buy',
                                      optimize=position_adjustment_enable)

    dca_limit = IntParameter(1, 5, default=1, space='buy', optimize=position_adjustment_enable)

    donchian_period = IntParameter(5, 50, default=23, space='buy', optimize=True)

    rsi_treshold = IntParameter(10, 50, default=35, space='buy', optimize=True)

    cci_treshold = IntParameter(-200, -10, default=-100, space='buy', optimize=True)

    kick_off_threshold = DecimalParameter(-0.99, 0, default=-0.03, decimals=2, space='sell', optimize=True)

    red_candles_before_buy = IntParameter(1, 5, default=1, space='buy', optimize=True)

    candle_time_threshold = DecimalParameter(0.01, 1, default=0.36, decimals=2, space='buy', optimize=False)

    exit_profit_offset = 0.001
    exit_profit_only = False

    order_types = {
        'entry': 'market',
        'exit': 'market',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    def calc_donchian_channels(self, dataframe, period: int):
        dataframe["upperDon"] = dataframe["high"].rolling(period).max()
        dataframe["lowerDon"] = dataframe["low"].rolling(period).min()
        dataframe["midDon"] = (dataframe["upperDon"] + dataframe["lowerDon"]) / 2
        return dataframe

    def mid_don_cross_over(self, dataframe):
        dataframe["position_m"] = np.nan
        dataframe["position_m"] = np.where(dataframe["close"] > dataframe["midDon"], 1, dataframe["position_m"])
        dataframe["position_m"] = dataframe["position_m"].ffill().fillna(0)
        return dataframe

    def don_channel_breakout(self, dataframe):
        dataframe["position_b"] = np.nan
        dataframe["position_b"] = np.where(dataframe["close"] > dataframe["upperDon"].shift(1), 1,
                                           dataframe["position_b"])
        dataframe["position_b"] = dataframe["position_b"].ffill().fillna(0)
        return dataframe

    def don_reversal(self, dataframe):
        dataframe["position_r"] = np.nan
        dataframe["position_r"] = np.where(dataframe["close"] < dataframe["lowerDon"].shift(1), 1,
                                           dataframe["position_r"])
        dataframe["position_r"] = dataframe["position_r"].ffill().fillna(0)
        return dataframe

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: Optional[str], side: str,
                 **kwargs) -> float:
        return self.leverage_value

    def calculate_price_change_coefficient(self, dataframe: DataFrame, candles_count: int = 1) -> DataFrame:
        dataframe['price_change_pct'] = dataframe['close'].pct_change(periods=candles_count)
        dataframe['price_change_coeff'] = dataframe['price_change_pct'].apply(
            lambda x: max(min(x, 0.99), -0.99) if x != 0 else 0.01 * np.sign(x)
        )
        dataframe[f'price_change_coeff_{candles_count}'] = dataframe['price_change_coeff'].rolling(
            window=candles_count).mean() * self.leverage_value
        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['cci'] = ta.CCI(dataframe, timeperiod=20)
        # Swing high/low
        dataframe = self.calc_swings(dataframe)
        dataframe = self.calc_donchian_channels(dataframe=dataframe, period=self.donchian_period.value)
        dataframe = self.mid_don_cross_over(dataframe=dataframe)
        dataframe = self.don_reversal(dataframe=dataframe)
        dataframe = self.don_channel_breakout(dataframe=dataframe)
        dataframe = self.calculate_price_change_coefficient(dataframe, 3)
        return dataframe

    def calc_swings(self, dataframe):
        dataframe['swing_low'] = (dataframe['close'].shift(2) > dataframe['close'].shift(1)) & \
                                 (dataframe['close'].shift(1) < dataframe['close']).astype(int)
        dataframe['swing_high'] = (dataframe['close'].shift(2) < dataframe['close'].shift(1)) & \
                                  (dataframe['close'].shift(1) > dataframe['close']).astype(int)
        return dataframe

    def check_red_candles(self, dataframe: DataFrame, n: int) -> DataFrame:
        red_candle = dataframe['close'] < dataframe['open']
        dataframe.loc[:, 'red_candles_in_row'] = red_candle.rolling(window=n).sum() == n
        return dataframe

    def timeframe_to_minutes(self, timeframe: str) -> int:
        if 'm' in timeframe:
            return int(timeframe.replace('m', ''))
        elif 'h' in timeframe:
            return int(timeframe.replace('h', '')) * 60
        elif 'd' in timeframe:
            return int(timeframe.replace('d', '')) * 1440
        else:
            raise ValueError(f"Unsupported timeframe: {timeframe}")

    def is_candle_open_more_than_threshold(self, dataframe: DataFrame, threshold: float) -> DataFrame:
        candle_timeframe_minutes = self.timeframe_to_minutes(timeframe=self.timeframe)
        t_time = pd.to_timedelta(candle_timeframe_minutes * threshold, unit='minutes')
        current_time = pd.Timestamp.utcnow()
        dataframe.loc[:, 'time_since_open'] = current_time - dataframe['date']
        dataframe.loc[:, 'is_open_more_than_threshold'] = dataframe['time_since_open'] >= t_time
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        self.check_red_candles(dataframe, self.red_candles_before_buy.value)
        self.is_candle_open_more_than_threshold(dataframe, threshold=self.candle_time_threshold.value)

        dataframe.loc[
            (
                    # ((dataframe['position_r'] == 1) |
                    #  (dataframe['swing_low'] == 1) |
                    #  (dataframe['position_m'] == 1) |
                    #  (dataframe['position_b'] == 1)) &
                    (dataframe['red_candles_in_row'] &
                     dataframe['is_open_more_than_threshold'] &
                     (dataframe['rsi'] < self.rsi_treshold.value) &
                     (dataframe['cci'] < self.cci_treshold.value))
            ), ['enter_long', 'enter_tag']
        ] = (1, 'swing_low')
        return dataframe

    def custom_exit(self, pair: str, trade: Trade, current_time: datetime, current_rate: float,
                    current_profit: float, **kwargs) -> Optional[Union[str, bool]]:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()
        if last_candle['price_change_coeff_3'] <= self.kick_off_threshold.value:
            return 'kick_off'

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['swing_high'] == 1)
            ), ['exit_long', 'exit_tag']
        ] = (1, 'swing_high')
        dataframe.loc[
            (
                (dataframe['price_change_coeff_3'] <= self.kick_off_threshold.value)
            ), ['exit_long', 'exit_tag']
        ] = (1, 'kick_off')
        return dataframe

    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                           rate: float, time_in_force: str, exit_reason: str,
                           current_time: datetime, **kwargs) -> bool:

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        profit_ratio = trade.calc_profit_ratio(rate)

        if 'swing' in exit_reason or 'trailing' in exit_reason:
            confirm_pf = profit_ratio > self.exit_profit_offset
            if confirm_pf:
                logging.info(f"[CTE] {pair} profit ratio: {profit_ratio}, confirmed profit {profit_ratio}")
            return confirm_pf

        if 'stop_loss' in exit_reason or 'kick_off' in exit_reason or 'force' in exit_reason:
            return True

        logging.info(f"[CTE] {pair} profit ratio: {profit_ratio}, reason: {exit_reason} not confirmed")
        return False

    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime, current_rate: float,
                        current_profit: float, after_fill: bool, **kwargs) -> Optional[float]:
        if pair in self.csl.keys():
            logging.info(f"[SPL] {pair} current profit: {current_profit}% has a stop loss -{self.csl[pair]}%")
            return -self.csl[pair]
        else:
            return self.stoploss

    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: Optional[float], max_stake: float,
                            leverage: float, entry_tag: Optional[str], side: str,
                            **kwargs) -> float:
        t = proposed_stake / self.dca_limit.value + 1
        if t < min_stake:
            return min_stake
        return t

    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                              current_rate: float, current_profit: float,
                              min_stake: Optional[float], max_stake: float,
                              current_entry_rate: float, current_exit_rate: float,
                              current_entry_profit: float, current_exit_profit: float,
                              **kwargs) -> Optional[float]:
        filled_entries = trade.select_filled_orders(trade.entry_side)

        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        last_candle = dataframe.iloc[-1]
        candle_open_price = last_candle['open']
        logging.info(f"[ADJ] {trade.pair} candle_open_price: {candle_open_price}")
        if self.candle_open_prices.get(trade.pair, None) == candle_open_price:
            logging.info(f"[ADJ] {trade.pair} candle_open_price: {candle_open_price} already processed")
            return None
        if len(filled_entries) > self.dca_limit.value:
            return None
        try:
            if current_profit < -self.dca_threshold_pct.value:
                self.candle_open_prices[trade.pair] = candle_open_price
                self.csl[trade.pair] = (self.dca_threshold_pct.value * self.dca_threshold_pct_k.value)
                logging.info(
                    f"[ADJ] {trade.pair} current SL adjusted to {self.csl[trade.pair]}, REBUY {trade.stake_amount * self.dca_multiplier.value} USDT")
                return trade.stake_amount * self.dca_multiplier.value
            else:
                self.csl[trade.pair] = (self.dca_threshold_pct.value * 1.25)
                return None
        except Exception as e:
            return None
