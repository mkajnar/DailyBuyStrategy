import logging
import os
import sys

import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame
from typing import Optional, Union, List

from pandas_ta import stdev

from freqtrade.enums import ExitCheckTuple
from freqtrade.persistence import Trade, Order
from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter, IStrategy, IntParameter)
import datetime
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import pandas_ta as pta


class HPStrategyV7(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = '5m'
    leverage_value = 10

    minimal_roi = {
        "0": 0.03
    }

    last_dca_timeframe = {}
    max_entry_position_adjustment = 5
    max_dca_multiplier = 5.5
    open_trade_limit = 15
    position_adjustment_enable = True
    dca_threshold_pct = DecimalParameter(0.01, 0.20, default=0.15, decimals=2, space='buy',
                                         optimize=position_adjustment_enable)
    candles_before_dca = IntParameter(1, 10, default=5, space='buy', optimize=True)

    rolling_ha_treshold = IntParameter(3, 10, default=7, space='buy', optimize=True)
    trailing_stop_positive = 0.003 * leverage_value
    trailing_stop_positive_offset = 0.01 * leverage_value

    stoploss = -0.10 * leverage_value
    use_exit_signal = False
    ignore_roi_if_entry_signal = False
    use_custom_stoploss = True
    # exit_profit_offset = 0.001 * leverage_value
    exit_profit_only = True
    order_types = {
        'entry': 'market',
        'exit': 'market',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: Optional[str], side: str,
                 **kwargs) -> float:
        return self.leverage_value

    # def calculate_heiken_ashi(self, dataframe):
    #     if dataframe.empty:
    #         raise ValueError("DataFrame je prázdný")
    #     heiken_ashi = pd.DataFrame(index=dataframe.index)
    #     heiken_ashi['HA_Close'] = (dataframe['open'] + dataframe['high'] + dataframe['low'] + dataframe['close']) / 4
    #     heiken_ashi['HA_Open'] = heiken_ashi['HA_Close'].shift(1)
    #     heiken_ashi['HA_Open'].iloc[0] = heiken_ashi['HA_Close'].iloc[0]
    #     heiken_ashi['HA_High'] = heiken_ashi[['HA_Open', 'HA_Close']].join(dataframe['high'], how='inner').max(axis=1)
    #     heiken_ashi['HA_Low'] = heiken_ashi[['HA_Open', 'HA_Close']].join(dataframe['low'], how='inner').min(axis=1)
    #     # Aplikace klouzavého průměru
    #     heiken_ashi['HA_Close'] = heiken_ashi['HA_Close'].rolling(window=self.rolling_ha_treshold.value).mean()
    #     heiken_ashi['HA_Open'] = heiken_ashi['HA_Open'].rolling(window=self.rolling_ha_treshold.value).mean()
    #     heiken_ashi['HA_High'] = heiken_ashi['HA_High'].rolling(window=self.rolling_ha_treshold.value).mean()
    #     heiken_ashi['HA_Low'] = heiken_ashi['HA_Low'].rolling(window=self.rolling_ha_treshold.value).mean()
    #
    #     return heiken_ashi
    #
    # def should_already_sell(self, dataframe):
    #     heiken_ashi = self.calculate_heiken_ashi(dataframe)
    #     last_candle = heiken_ashi.iloc[-1]
    #     if last_candle['HA_Close'] > last_candle['HA_Open']:
    #         return False
    #     else:
    #         return True

    def adjust_entry_price(self, trade: Trade, order: Optional[Order], pair: str,
                           current_time: datetime, proposed_rate: float, current_order_rate: float,
                           entry_tag: Optional[str], side: str, **kwargs) -> float:

        return proposed_rate

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                            time_in_force: str, current_time: datetime, entry_tag: Optional[str],
                            side: str, **kwargs) -> bool:
        return (Trade.get_open_trade_count() < self.open_trade_limit)

    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                           rate: float, time_in_force: str, exit_reason: str,
                           current_time: datetime, **kwargs) -> bool:
        logging.info(f"Checking CTE - {exit_reason} for pair {pair} at rate {rate} - S1")
        # if 'force_exit' in exit_reason and trade.calc_profit_ratio(rate) < 0:
        #     return False
        if 'trailing' in exit_reason and trade.calc_profit_ratio(rate) < 0:
            return False
        if 'roi' in exit_reason and trade.calc_profit_ratio(rate) < 0:
            return False
        return True

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                        current_profit: float, **kwargs) -> float:
        if current_profit > self.trailing_stop_positive_offset:
            logging.info(f"Checking CSL - {current_profit} for pair {pair} at rate {current_rate} - S1")
            new_stop_loss = current_profit - self.trailing_stop_positive
            logging.info(f"Checking CSL - {new_stop_loss} for pair {pair} at rate {current_rate} - S2")
            r = max(new_stop_loss, self.stoploss)
            logging.info(f"Checking CSL - {r} for pair {pair} at rate {current_rate} - S3")
            return r
        return -1

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['cci'] = ta.CCI(dataframe, timeperiod=14)
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        dataframe['cci_buy_signal'] = (
                (dataframe['cci'] < -100) & (dataframe['rsi'] < 50) & (dataframe['volume'] > 0)).astype(int)
        dataframe['smi'] = pta.smi(dataframe, length=20, scalar=2)
        dataframe = self.prepare_rti(dataframe)
        return dataframe

    def prepare_rti(self, dataframe):
        try:
            trend_data_count = 100
            trend_sensitivity_percentage = 95
            signal_length = 20
            dataframe['upper_trend'] = dataframe['close'] + stdev(dataframe['close'], 2)
            dataframe['lower_trend'] = dataframe['close'] - stdev(dataframe['close'], 2)
            upper_array = dataframe['upper_trend'].rolling(window=trend_data_count).apply(
                lambda x: np.sort(x)[-int(trend_sensitivity_percentage / 100 * len(x))])
            lower_array = dataframe['lower_trend'].rolling(window=trend_data_count).apply(
                lambda x: np.sort(x)[int((100 - trend_sensitivity_percentage) / 100 * len(x)) - 1])
            dataframe['RTI'] = (dataframe['close'] - lower_array) / (upper_array - lower_array) * 100
            dataframe['MA_RTI'] = ta.ema(dataframe['RTI'], length=signal_length)
        except:
            pass

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[(dataframe['cci'] < -100) &
                      (dataframe['rsi'] < 50) &
                      (dataframe['volume'] > 0), ['enter_long', 'enter_tag']] = (1, 'cci_buy')
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[(dataframe['cci'] > 100) &
                      (dataframe['rsi'] > 80) &
                      (dataframe['volume'] > 0), ['exit_long', 'exit_tag']] = (1, 'cci_sell')
        return dataframe

        # This is called when placing the initial order (opening trade)

    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: Optional[float], max_stake: float,
                            leverage: float, entry_tag: Optional[str], side: str,
                            **kwargs) -> float:
        return proposed_stake / self.max_dca_multiplier

    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                              current_rate: float, current_profit: float,
                              min_stake: Optional[float], max_stake: float,
                              current_entry_rate: float, current_exit_rate: float,
                              current_entry_profit: float, current_exit_profit: float,
                              **kwargs) -> Optional[float]:

        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()
        previous_candle = dataframe.iloc[-2].squeeze()

        if ((current_profit > (self.dca_threshold_pct.value * 3)) and (trade.nr_of_successful_exits == 0)):
            return -(trade.stake_amount / 2)

        if trade.id in self.last_dca_timeframe.keys():
            td = current_time - self.last_dca_timeframe[trade.id]
            if ((td.total_seconds() / 60)
                    < (self.timeframe_to_minutes(self.timeframe) * self.candles_before_dca.value)):
                return None

        filled_entries = trade.select_filled_orders(trade.entry_side)
        count_of_entries = trade.nr_of_successful_entries - trade.nr_of_successful_exits
        if ((count_of_entries >= self.max_entry_position_adjustment) or
                (last_candle['close'] < previous_candle['close']) or
                (current_profit > -self.dca_threshold_pct.value)):
            return None

        if ((current_profit < -self.dca_threshold_pct.value * count_of_entries) and (
                (last_candle['cci_buy_signal'] == 1) or (previous_candle['cci_buy_signal'] == 1))):
            try:
                averaged_stake = filled_entries[0].stake_amount
            except Exception as exception:
                averaged_stake = 5
                pass
            if (count_of_entries > 0):
                averaged_stake = (sum([entry.stake_amount
                                       for entry in filled_entries if entry.stake_amount > 0]) / count_of_entries)
            stake_amount = averaged_stake * (1 + (count_of_entries * 0.25))
            self.last_dca_timeframe[trade.id] = current_time
            return stake_amount

    def timeframe_to_minutes(self, timeframe: str) -> int:
        """Convert a timeframe string to minutes."""
        if 'm' in timeframe:
            return int(timeframe.replace('m', ''))
        elif 'h' in timeframe:
            return int(timeframe.replace('h', '')) * 60
        elif 'd' in timeframe:
            return int(timeframe.replace('d', '')) * 1440
        else:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
