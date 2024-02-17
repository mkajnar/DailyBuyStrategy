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
from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter, IStrategy, IntParameter)
import datetime
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import pandas_ta as pta


class HPStrategyV7(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = '15m'
    leverage_value = 3

    minimal_roi = {
        "0": 0.03
    }

    is_opt_sl = True
    sl1 = DecimalParameter(0.01, 0.50, default=0.35, space='sell', decimals=2, optimize=is_opt_sl)
    sl3 = DecimalParameter(0.01, 0.35, default=0.25, space='sell', decimals=2, optimize=is_opt_sl)
    sl5 = DecimalParameter(0.01, 0.25, default=0.2, space='sell', decimals=2, optimize=is_opt_sl)
    sl10 = DecimalParameter(0.01, 0.20, default=0.15, space='sell', decimals=2, optimize=is_opt_sl)
    stoplosses = {1: -sl1.value, 3: -sl3.value, 5: -sl5.value, 10: -sl10.value}

    is_opt_adj = True
    adj1 = IntParameter(1, 5, default=5, space='sell', optimize=is_opt_adj)
    adj3 = IntParameter(1, 5, default=3, space='sell', optimize=is_opt_adj)
    adj5 = IntParameter(1, 5, default=2, space='sell', optimize=is_opt_adj)
    adj10 = IntParameter(1, 5, default=1, space='sell', optimize=is_opt_adj)
    adjustments = {1: adj1.value, 3: adj3.value, 5: adj5.value, 10: adj10.value}

    is_opt_m = True
    m1 = DecimalParameter(0.1, 10, default=1.5, space='sell', decimals=1, optimize=is_opt_m)
    m3 = DecimalParameter(0.1, 10, default=2.25, space='sell', decimals=1, optimize=is_opt_m)
    m5 = DecimalParameter(0.1, 10, default=5.5, space='sell', decimals=1, optimize=is_opt_m)
    m10 = DecimalParameter(0.1, 10, default=7.5, space='sell', decimals=1, optimize=is_opt_m)
    multis = {1: m1.value, 3: m3.value, 5: m5.value, 10: m10.value}

    last_dca_timeframe = {}
    max_entry_position_adjustment = adjustments[leverage_value]
    max_dca_multiplier = multis[max_entry_position_adjustment]
    open_trade_limit = IntParameter(1, 20, default=5, space='buy', optimize=True)
    is_opt_position_adjustment = True

    position_adjustment_enable = True
    dca_threshold_pct = DecimalParameter(0.01, 0.15, default=0.125, decimals=3, space='buy', optimize=position_adjustment_enable)

    max_hold_time = DecimalParameter(1, 24, default=8, space='sell', optimize=True, decimals=1)
    candles_before_dca = IntParameter(1, 10, default=5, space='buy', optimize=True)
    rolling_ha_treshold = IntParameter(3, 10, default=7, space='buy', optimize=True)

    is_optimize_32 = False
    buy_rsi_fast_32 = IntParameter(20, 70, default=45, space='buy', optimize=is_optimize_32)
    buy_rsi_32 = IntParameter(15, 50, default=35, space='buy', optimize=is_optimize_32)
    buy_sma15_32 = DecimalParameter(0.900, 1, default=0.961, decimals=3, space='buy', optimize=is_optimize_32)
    buy_cti_32 = DecimalParameter(-1, 0, default=-0.58, decimals=2, space='buy', optimize=is_optimize_32)
    sell_fastx = IntParameter(50, 100, default=70, space='sell', optimize=True)
    close_perc_threshold_buy = DecimalParameter(0.750, 1.000, default=0.999, decimals=3, space='buy', optimize=True)
    is_optimize_deadfish = False
    sell_deadfish_bb_width = DecimalParameter(0.03, 0.75, default=0.05, space='sell', optimize=is_optimize_deadfish)
    sell_deadfish_profit = DecimalParameter(-0.15, -0.05, default=-0.05, space='sell', optimize=is_optimize_deadfish)
    sell_deadfish_bb_factor = DecimalParameter(0.90, 1.20, default=1.0, space='sell', optimize=is_optimize_deadfish)
    sell_deadfish_volume_factor = DecimalParameter(1, 2.5, default=1.0, space='sell', optimize=is_optimize_deadfish)

    trailing_stop_positive = 0.003
    trailing_stop_positive_offset = 0.008
    stoploss = stoplosses[leverage_value]

    fast_ewo = 100
    slow_ewo = 200

    use_exit_signal = False
    ignore_roi_if_entry_signal = False
    use_custom_stoploss = True
    exit_profit_offset = 0.001 * leverage_value
    exit_profit_only = False
    order_types = {
        'entry': 'market',
        'exit': 'market',
        'stoploss': 'market',
        'stoploss_on_exchange': True
    }

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: Optional[str], side: str,
                 **kwargs) -> float:
        return self.leverage_value

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                        current_profit: float, **kwargs) -> float:

        if current_profit > self.trailing_stop_positive_offset:
            logging.info(f"Checking CSL - {current_profit} for pair {pair} at rate {current_rate} - S1")
            new_stop_loss = current_profit - self.trailing_stop_positive
            logging.info(f"Checking CSL - {new_stop_loss} for pair {pair} at rate {current_rate} - S2")
            r = max(new_stop_loss, self.stoploss)
            logging.info(f"Checking CSL - {r} for pair {pair} at rate {current_rate} - S3")
            return r

        dca_count = trade.nr_of_successful_entries - trade.nr_of_successful_exits + 1
        if current_time - trade.open_date_utc > datetime.timedelta(hours=self.max_hold_time.value):
            logging.info(f"Position for {pair} over time {self.max_hold_time.value} hours, close position.")
            return 1
        if dca_count >= self.max_entry_position_adjustment:
            if current_profit < 0:
                logging.info(f"Position for {pair} after {dca_count} DCA is not profitable, close position.")
            return 1

    def calculate_heiken_ashi(self, dataframe):
        if dataframe.empty:
            raise ValueError("DataFrame je prázdný")
        heiken_ashi = pd.DataFrame(index=dataframe.index)
        heiken_ashi['HA_Close'] = (dataframe['open'] + dataframe['high'] + dataframe['low'] + dataframe['close']) / 4

        # První řádek pro HA_Open musí být stejný jako HA_Close, protože není předchozího řádku
        heiken_ashi['HA_Open'] = heiken_ashi['HA_Close'].shift(1)
        heiken_ashi['HA_Open'].iloc[0] = heiken_ashi['HA_Close'].iloc[0]

        # Výpočet HA_High a HA_Low s použitím jen historických dat
        heiken_ashi['HA_High'] = heiken_ashi[['HA_Open', 'HA_Close']].join(dataframe[['high']], how='inner').max(axis=1)
        heiken_ashi['HA_Low'] = heiken_ashi[['HA_Open', 'HA_Close']].join(dataframe[['low']], how='inner').min(axis=1)

        # Aplikace klouzavého průměru
        rolling_window = self.rolling_ha_treshold.value
        heiken_ashi['HA_Close'] = heiken_ashi['HA_Close'].rolling(window=rolling_window).mean()
        heiken_ashi['HA_Open'] = heiken_ashi['HA_Open'].rolling(window=rolling_window).mean()
        heiken_ashi['HA_High'] = heiken_ashi['HA_High'].rolling(window=rolling_window).mean()
        heiken_ashi['HA_Low'] = heiken_ashi['HA_Low'].rolling(window=rolling_window).mean()

        return heiken_ashi

    def should_already_sell(self, dataframe):
        heiken_ashi = self.calculate_heiken_ashi(dataframe)
        last_candle = heiken_ashi.iloc[-1].squeeze()
        if last_candle['HA_Close'] > last_candle['HA_Open']:
            return False
        else:
            return True

    def adjust_entry_price(self, trade: Trade, order: Optional[Order], pair: str,
                           current_time: datetime, proposed_rate: float, current_order_rate: float,
                           entry_tag: Optional[str], side: str, **kwargs) -> float:

        return proposed_rate

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                            time_in_force: str, current_time: datetime, entry_tag: Optional[str],
                            side: str, **kwargs) -> bool:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        latest_candle = dataframe.iloc[-1].squeeze()
        previous_candle = dataframe.iloc[-2].squeeze()
        return (Trade.get_open_trade_count() < self.open_trade_limit.value)

    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                           rate: float, time_in_force: str, exit_reason: str,
                           current_time: datetime, **kwargs) -> bool:
        logging.info(f"Checking CTE - {exit_reason} for pair {pair} at rate {rate} - S1")
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        # if 'force_exit' in exit_reason and trade.calc_profit_ratio(rate) < 0:
        #     return False
        if 'trailing' in exit_reason and trade.calc_profit_ratio(rate) > 0:
            return self.should_already_sell(dataframe=dataframe)
        if 'trailing' in exit_reason and trade.calc_profit_ratio(rate) < 0:
            return False
        if 'roi' in exit_reason and trade.calc_profit_ratio(rate) < 0:
            return False
        if 'roi' in exit_reason and trade.calc_profit_ratio(rate) > 0:
            return self.should_already_sell(dataframe=dataframe)
        return True

    def ewo(self, dataframe, ema_length=5, ema2_length=35):
        df = dataframe.copy()
        ema1 = ta.EMA(df, timeperiod=ema_length)
        ema2 = ta.EMA(df, timeperiod=ema2_length)
        emadif = (ema1 - ema2) / df['close'] * 100
        return emadif

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['cci'] = ta.CCI(dataframe, timeperiod=14)
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)

        dataframe['sma_15'] = ta.SMA(dataframe, timeperiod=15)
        dataframe['cti'] = pta.cti(dataframe["close"], length=20)
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['rsi_fast'] = ta.RSI(dataframe, timeperiod=4)
        dataframe['rsi_slow'] = ta.RSI(dataframe, timeperiod=20)
        # profit sell indicators
        stoch_fast = ta.STOCHF(dataframe, 5, 3, 0, 3, 0)
        dataframe['fastk'] = stoch_fast['fastk']
        dataframe['cci'] = ta.CCI(dataframe, timeperiod=20)
        dataframe['sma_15'] = ta.SMA(dataframe, timeperiod=15)
        dataframe['cti'] = pta.cti(dataframe["close"], length=20)
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['rsi_fast'] = ta.RSI(dataframe, timeperiod=4)
        dataframe['rsi_slow'] = ta.RSI(dataframe, timeperiod=20)

        # ewo indicators
        dataframe['ema_8'] = ta.EMA(dataframe, timeperiod=8)
        dataframe['ema_16'] = ta.EMA(dataframe, timeperiod=16)

        # profit sell indicators
        stoch_fast = ta.STOCHF(dataframe, 5, 5, 1, 5, 1)
        dataframe['fastd'] = stoch_fast['fastd']
        dataframe['fastk'] = stoch_fast['fastk']

        # loss sell indicators
        bollinger2 = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband2'] = bollinger2['lower']
        dataframe['bb_middleband2'] = bollinger2['mid']
        dataframe['bb_upperband2'] = bollinger2['upper']

        dataframe['bb_width'] = (
                (dataframe['bb_upperband2'] - dataframe['bb_lowerband2']) / dataframe['bb_middleband2'])

        dataframe['volume_mean_12'] = dataframe['volume'].rolling(12).mean().shift(1)
        dataframe['volume_mean_24'] = dataframe['volume'].rolling(24).mean().shift(1)

        dataframe['mfi'] = ta.MFI(dataframe, timeperiod=14)
        dataframe['cap'] = (dataframe['volume'].shift(0) * dataframe['close'].shift(0))

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
            dataframe['rti'] = (dataframe['close'] - lower_array) / (upper_array - lower_array) * 100
            dataframe['MA_RTI'] = ta.ema(dataframe['rti'], length=signal_length)
        except:
            pass

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        dataframe.loc[:, 'enter_tag'] = ''
        buy_1 = (
                ((dataframe['rsi_slow'] < dataframe['rsi_slow'].shift(1)) &
                 (dataframe['rsi_fast'] < self.buy_rsi_fast_32.value) &
                 (dataframe['rsi'] > self.buy_rsi_32.value) &
                 (dataframe['close'] < dataframe['sma_15'] * self.buy_sma15_32.value) &
                 (dataframe['cti'] < self.buy_cti_32.value)) |
                (qtpylib.crossed_above(dataframe['rsi_fast'], dataframe['rsi_slow']))
        )
        conditions.append(buy_1)
        dataframe.loc[buy_1, 'enter_tag'] += 'buy_1'
        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions),
                'enter_long'] = 1

        return dataframe

    def custom_exit(self, pair: str, trade: Trade, current_time: datetime, current_rate: float,
                    current_profit: float, **kwargs) -> Optional[Union[str, bool]]:

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        current_candle = dataframe.iloc[-1].squeeze()

        # stoploss - deadfish
        if ((current_profit < self.sell_deadfish_profit.value)
                and (current_candle['bb_width'] < self.sell_deadfish_bb_width.value)
                and (current_candle['close'] > current_candle['bb_middleband2'] * self.sell_deadfish_bb_factor.value)
                and (current_candle['volume_mean_12'] < current_candle[
                    'volume_mean_24'] * self.sell_deadfish_volume_factor.value)):
            return "sell_stoploss_deadfish"

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

        if (current_profit < -self.dca_threshold_pct.value * count_of_entries):
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
        if 'm' in timeframe:
            return int(timeframe.replace('m', ''))
        elif 'h' in timeframe:
            return int(timeframe.replace('h', '')) * 60
        elif 'd' in timeframe:
            return int(timeframe.replace('d', '')) * 1440
        else:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
