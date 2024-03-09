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


class HPStrategyV7Ultra(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = '5m'
    leverage_value = 3
    stoploss = -0.08 * leverage_value
    minimal_roi = {
        "0": 0.007 * leverage_value
    }
    allprofits = {}
    process_only_new_candles = True
    startup_candle_count = 50
    position_adjustment_enable = False
    trailing_stop = True
    trailing_only_offset_is_reached = False
    trailing_stop_positive = 0.003 * leverage_value
    trailing_stop_positive_offset = 0.007 * leverage_value

    use_exit_signal = True
    ignore_roi_if_entry_signal = True

    donchian_period = IntParameter(5, 50, default=20, space='buy', optimize=True)
    total_positive_profit_threshold = IntParameter(1, 30, default=16, space='sell', optimize=True)
    total_negative_profit_threshold = IntParameter(-30, -1, default=-10, space='sell', optimize=True)
    exit_profit_offset_par = DecimalParameter(0.001, 0.05, default=0.021, space='sell', decimals=3, optimize=True)
    exit_profit_offset = exit_profit_offset_par.value * leverage_value

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

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Swing high/low
        dataframe = self.calc_swings(dataframe)
        dataframe = self.calc_donchian_channels(dataframe=dataframe, period=self.donchian_period.value)
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
        logging.info(f"[CTE] {pair} exit reason: {exit_reason}")
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        profit_ratio = trade.calc_profit_ratio(rate)
        logging.info(f"[CTE] {pair} profit ratio: {profit_ratio}, exit reason: {exit_reason}")

        if (exit_reason.startswith('stop') and 'loss' in exit_reason) or 'trailing' in exit_reason:
            confirm_sl = profit_ratio < -self.stoploss
            if confirm_sl:
                logging.info(f"[CTE] {pair} profit ratio: {profit_ratio}, confirmed stoploss {self.stoploss}")
                exit_reason = "stoploss"
                return confirm_sl
        if 'swing' in exit_reason or 'trailing' in exit_reason:
            confirm_pf = profit_ratio > self.exit_profit_offset
            if confirm_pf:
                logging.info(f"[CTE] {pair} profit ratio: {profit_ratio}, confirmed profit {profit_ratio}")
            return confirm_pf
        return True

    def custom_exit(self, pair: str, trade: Trade, current_time: datetime, current_rate: float,
                    current_profit: float, **kwargs) -> Optional[Union[str, bool]]:
        logging.info(f"[CE] {pair} current profit: {current_profit}")
        total_profit = 0
        if len(self.allprofits.keys()) >= 0:
            total_profit = sum(self.allprofits.values())
        if current_profit != self.allprofits.get(pair, 0):
            logging.info(f"[CE] Current profit: {current_profit} for {pair} is set to all profits dictionary")
            self.allprofits[pair] = current_profit
            logging.info(f"[CE] Total profit: {total_profit}")

        c = len(self.allprofits.keys()) >= self.max_open_trades
        if total_profit > self.total_positive_profit_threshold.value and c:
            logging.info(
                f"[CE] Total profit: {total_profit} is bigger than {self.total_positive_profit_threshold.value}, sell all positions...")
            return True
        if total_profit < self.total_negative_profit_threshold.value and c:
            logging.info(
                f"[CE] Total profit: {total_profit} is less than {self.total_negative_profit_threshold.value}, sell all positions...")
            return True
        return None
