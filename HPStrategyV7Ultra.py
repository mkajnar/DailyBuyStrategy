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
import datetime
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import pandas_ta as pta


class HPStrategyV7Ultra(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = '5m'
    leverage_value = 3
    stoploss = -0.02 * leverage_value

    minimal_roi = {
        "0": 0.01 * leverage_value
    }

    process_only_new_candles = True
    startup_candle_count = 50
    position_adjustment_enable = False
    trailing_stop = True
    trailing_only_offset_is_reached = False
    trailing_stop_positive = 0.001 * leverage_value
    trailing_stop_positive_offset = 0.003 * leverage_value

    use_exit_signal = True
    ignore_roi_if_entry_signal = True

    exit_profit_offset = 0.001 * leverage_value
    exit_profit_only = True

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

    def mid_don_cross_over(self, dataframe, period: int = 20, shorts: bool = True):
        dataframe["position_m"] = np.nan
        dataframe["position_m"] = np.where(dataframe["close"] > dataframe["midDon"], 1, dataframe["position_m"])
        dataframe["position_m"] = dataframe["position_m"].ffill().fillna(0)
        return dataframe

    def don_channel_breakout(self, dataframe, period=20, shorts=True):
        dataframe["position_b"] = np.nan
        dataframe["position_b"] = np.where(dataframe["close"] > dataframe["upperDon"].shift(1), 1,
                                           dataframe["position_b"])
        dataframe["position_b"] = dataframe["position_b"].ffill().fillna(0)
        return dataframe

    def don_reversal(self, dataframe, period=20, shorts=True):
        dataframe["position_r"] = np.nan
        dataframe["position_r"] = np.where(dataframe["close"] < dataframe["lowerDon"].shift(1), 1,
                                           dataframe["position_r"])
        dataframe["position_r"] = dataframe["position_r"].ffill().fillna(0)
        return dataframe

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: Optional[str], side: str,
                 **kwargs) -> float:
        return self.leverage_value

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Swing high/low
        dataframe = self.calc_swings(dataframe)
        dataframe = self.calc_donchian_channels(dataframe=dataframe, period=20)
        dataframe = self.mid_don_cross_over(dataframe=dataframe)
        dataframe = self.don_reversal(dataframe=dataframe)
        dataframe = self.don_channel_breakout(dataframe=dataframe)
        return dataframe

    def calc_swings(self, dataframe):
        dataframe['swing_low'] = (dataframe['close'].shift(2) > dataframe['close'].shift(1)) & \
                                 (dataframe['close'].shift(1) < dataframe['close']).astype(int)
        dataframe['swing_high'] = (dataframe['close'].shift(2) < dataframe['close'].shift(1)) & \
                                  (dataframe['close'].shift(1) > dataframe['close']).astype(int)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                    (dataframe['position_r'] == 1)
                    |
                    (dataframe['swing_low'] == 1)
                    |
                    (dataframe['position_m'] == 1)
                    |
                    (dataframe['position_b'] == 1)
            ), ['enter_long', 'enter_tag']
        ] = (1, 'swing_low')
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['swing_high'] == 1)
            ), ['exit_long', 'exit_tag']
        ] = (1, 'swing_high')
        return dataframe

    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                           rate: float, time_in_force: str, exit_reason: str,
                           current_time: datetime, **kwargs) -> bool:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        profit_ratio = trade.calc_profit_ratio(rate)
        if 'swing' in exit_reason or 'trailing' in exit_reason:
            return profit_ratio > 0
        return True
