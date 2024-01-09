import datetime
import json
import logging
import math
import os
from datetime import datetime
from datetime import timedelta, timezone
from functools import reduce
from typing import Optional, List
from enum import Enum

import numpy as np
import talib.abstract as ta
from pandas import DataFrame
from talib import CDLDOJI
from technical.indicators import ichimoku

import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.enums.tradingmode import TradingMode
from freqtrade.persistence import Trade, LocalTrade, Order
from freqtrade.strategy import merge_informative_pair, DecimalParameter, IntParameter
from freqtrade.strategy.interface import IStrategy

from modules.lambo2 import Lambo
from modules.elliot import Elliot


def pct_change(a, b):
    return (b - a) / a


class LogLevel(Enum):
    VERBOSE = 0
    INFO = 1
    WARNING = 2
    ERROR = 3


log_level = LogLevel.ERROR


class HPSDivergence(IStrategy):
    INTERFACE_VERSION = 3

    support_dict = {}
    resistance_dict = {}

    timeframed_drops = {'1m': -0.01,
                        '5m': -0.05,
                        '15m': -0.05,
                        '30m': -0.075,
                        '1h': -0.1
                        }

    max_safety_orders = 3
    lowest_prices = {}
    highest_prices = {}
    price_drop_percentage = {}
    locked = []
    pairs_close_to_high = []

    out_open_trades_limit = 10
    is_optimize_cofi = False
    use_exit_signal = True
    exit_profit_only = True
    # sell_profit_offset = 0.015
    ignore_roi_if_entry_signal = False
    position_adjustment_enable = True
    order_time_in_force = {
        'entry': 'gtc',
        'exit': 'gtc'
    }

    # timeframe = '1m'
    inf_1h = '1h'
    process_only_new_candles = True
    startup_candle_count = 400
    plot_config = {
        'main_plot': {
            'sma_9': {'color': 'red'},
            'sma_2': {'color': 'blue'},
        },
    }

    config_defaults = {
        "dca_min_rsi": 35
    }

    sell_params = {
        "base_nb_candles_sell": 22,
        "high_offset": 1.014,
        "high_offset_2": 1.01
    }

    # lambo2
    lambo2 = Lambo()
    lambo2_ema_14_factor = DecimalParameter(0.8, 1.2, decimals=3, default=lambo2.lambo2_ema_14_factor, space='buy',
                                            optimize=True)
    lambo2_rsi_4_limit = IntParameter(10, 60, default=lambo2.lambo2_rsi_4_limit, space='buy', optimize=True)
    lambo2_rsi_14_limit = IntParameter(10, 60, default=lambo2.lambo2_rsi_14_limit, space='buy', optimize=True)
    lambo2.use_hyperopts(lambo2_ema_14_factor, lambo2_rsi_4_limit, lambo2_rsi_14_limit)

    elliot = Elliot()
    base_nb_candles_sell = IntParameter(8, 20, default=elliot.base_nb_candles_sell, space='sell', optimize=False)
    base_nb_candles_buy = IntParameter(8, 20, default=elliot.base_nb_candles_buy, space='buy', optimize=False)
    low_offset = DecimalParameter(0.975, 0.995, default=elliot.low_offset, space='buy', optimize=True)
    ewo_high = DecimalParameter(3.0, 5, default=elliot.ewo_high, space='buy', optimize=True)
    ewo_low = DecimalParameter(-20.0, -7.0, default=-elliot.ewo_low, space='buy', optimize=True)
    rsi_buy = IntParameter(30, 70, default=elliot.rsi_buy, space='buy', optimize=False)
    high_offset = DecimalParameter(1.000, 1.010, default=elliot.high_offset, space='sell', optimize=True)
    buy_ema_cofi = DecimalParameter(0.96, 0.98, default=elliot.buy_ema_cofi, optimize=True)
    buy_fastk = IntParameter(20, 30, default=elliot.buy_fastk, optimize=True)
    buy_fastd = IntParameter(20, 30, default=elliot.buy_fastd, optimize=True)
    buy_adx = IntParameter(20, 30, default=elliot.buy_adx, optimize=True)
    buy_ewo_high = DecimalParameter(2, 12, default=elliot.buy_ewo_high, optimize=True)
    elliot.use_hyperopts(base_nb_candles_sell, base_nb_candles_buy, low_offset, ewo_high, ewo_low, rsi_buy, high_offset,
                         buy_ema_cofi, buy_fastk, buy_fastd, buy_adx, buy_ewo_high)

    # Stop loss management
    stoploss = -0.9

    trailing_stop = True
    trailing_stop_positive = 0.002
    trailing_stop_positive_offset = 0.008
    trailing_only_offset_is_reached = True

    minimal_roi = {
        "0": 0.5,
        "120": 0.3,
        "240": 0.1,
        "360": 0.07,
        "480": 0.05,
        "720": 0.03,
        "960": 0.01,
        "1440": 0.005,
        "2880": 0.003,
        "4320": 0.001,
        "5760": 0.000
    }

    order_types = {
        'entry': 'market',
        'exit': 'market',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    high_offset = DecimalParameter(1.000, 1.010, default=sell_params['high_offset'], space='sell', optimize=True)
    high_offset_2 = DecimalParameter(1.000, 1.010, default=sell_params['high_offset_2'], space='sell', optimize=True)

    timeframes_in_minutes = {
        '1m': 1,
        '5m': 5,
        '15m': 15,
        '30m': 30,
        '1h': 60,
        '4h': 240,
        '1d': 1440
    }

    dca_min_rsi = IntParameter(35, 75, default=config_defaults['dca_min_rsi'], space='buy', optimize=True)

    @property
    def protections(self):
        return [
            {
                "method": "CooldownPeriod",
                "stop_duration_candles": 3
            },
            {
                "method": "MaxDrawdown",
                "lookback_period_candles": 48,
                "trade_limit": 20,
                "stop_duration_candles": 4,
                "max_allowed_drawdown": 0.2
            },
            {
                "method": "StoplossGuard",
                "lookback_period_candles": 24,
                "trade_limit": 4,
                "stop_duration_candles": 2,
                "only_per_pair": False
            },
            {
                "method": "LowProfitPairs",
                "lookback_period_candles": 6,
                "trade_limit": 2,
                "stop_duration_candles": 60,
                "required_profit": 0.02
            },
            {
                "method": "LowProfitPairs",
                "lookback_period_candles": 24,
                "trade_limit": 4,
                "stop_duration_candles": 2,
                "required_profit": 0.01
            }
        ]

    def version(self) -> str:
        return "HPDivergence v1.0"

    def pivot_points(self, high, low, period=10):
        pivot_high = high.rolling(window=2 * period + 1, center=True).max()
        pivot_low = low.rolling(window=2 * period + 1, center=True).min()
        return high == pivot_high, low == pivot_low

    def calculate_support_resistance(self, df, period=10, loopback=290):
        high_pivot, low_pivot = self.pivot_points(df['high'], df['low'], period)
        df['resistance'] = df['high'][high_pivot]
        df['support'] = df['low'][low_pivot]
        return df

    def calculate_support_resistance_dicts(self, pair: str, df: DataFrame):
        try:
            df = self.calculate_support_resistance(df)
            self.support_dict[pair] = self.calculate_dynamic_clusters(df['support'].dropna().tolist(), 4)
            self.resistance_dict[pair] = self.calculate_dynamic_clusters(df['resistance'].dropna().tolist(), 4)
        except Exception as ex:
            if (log_level.value <= 3): logging.error(str(ex))

    def calculate_dynamic_clusters(self, values, max_clusters):
        """
        Dynamically calculates the averaged clusters from the given list of values.

         Args:
         values (list): List of values to cluster.
         max_clusters (int): Maximum number of clusters to create.

         Returns:
         list: List of average values for each cluster created.
        """

        def cluster_values(threshold):
            sorted_values = sorted(values)
            clusters = []
            current_cluster = [sorted_values[0]]

            for value in sorted_values[1:]:
                if value - current_cluster[-1] <= threshold:
                    current_cluster.append(value)
                else:
                    clusters.append(current_cluster)
                    current_cluster = [value]

            clusters.append(current_cluster)
            return clusters

        threshold = 0.3  # Initial threshold value
        while True:
            clusters = cluster_values(threshold)
            if len(clusters) <= max_clusters:
                break
            threshold += 0.3

        # Calculation of means for each cluster
        cluster_averages = [round(sum(cluster) / len(cluster), 2) for cluster in clusters]
        return cluster_averages

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:


        if 'enter_tag' not in dataframe.columns:
            dataframe.loc[:, 'enter_tag'] = ''
        if 'exit_tag' not in dataframe.columns:
            dataframe.loc[:, 'exit_tag'] = ''
        if 'enter_long' not in dataframe.columns:
            dataframe.loc[:, 'enter_long'] = 0
        if 'exit_long' not in dataframe.columns:
            dataframe.loc[:, 'exit_long'] = 0
            
        dataframe['price_history'] = dataframe['close'].shift(1)
        data_last_bbars = dataframe[-30:].copy()
        low_min = dataframe['low'].rolling(window=14).min()
        high_max = dataframe['high'].rolling(window=14).max()
        dataframe['stoch_k'] = 100 * (dataframe['close'] - low_min) / (high_max - low_min)
        dataframe['stoch_d'] = dataframe['stoch_k'].rolling(window=3).mean()

        pair = metadata['pair']
        if self.config['stake_currency'] in ['USDT', 'BUSD']:
            btc_info_pair = f"BTC/{self.config['stake_currency']}"
            if (self.config['trading_mode'] == TradingMode.FUTURES):
                btc_info_pair = btc_info_pair + f":{self.config['stake_currency']}"
        else:
            btc_info_pair = "BTC/USDT"
            if (self.config['trading_mode'] == TradingMode.FUTURES):
                btc_info_pair = btc_info_pair + f":{self.config['stake_currency']}"

        btc_info_tf = self.dp.get_pair_dataframe(btc_info_pair, self.inf_1h)
        btc_info_tf = self.info_tf_btc_indicators(btc_info_tf, metadata)
        dataframe = merge_informative_pair(dataframe, btc_info_tf, self.timeframe, self.inf_1h, ffill=True)
        drop_columns = [f"{s}_{self.inf_1h}" for s in ['date', 'open', 'high', 'low', 'close', 'volume']]
        dataframe.drop(columns=dataframe.columns.intersection(drop_columns), inplace=True)

        btc_base_tf = self.dp.get_pair_dataframe(btc_info_pair, self.timeframe)
        btc_base_tf = self.base_tf_btc_indicators(btc_base_tf, metadata)
        dataframe = merge_informative_pair(dataframe, btc_base_tf, self.timeframe, self.timeframe, ffill=True)
        drop_columns = [f"{s}_{self.timeframe}" for s in ['date', 'open', 'high', 'low', 'close', 'volume']]
        dataframe.drop(columns=dataframe.columns.intersection(drop_columns), inplace=True)

        dataframe['hma_50'] = qtpylib.hull_moving_average(dataframe['close'], window=50)
        dataframe['sma_2'] = ta.SMA(dataframe, timeperiod=2)
        dataframe['sma_9'] = ta.SMA(dataframe, timeperiod=9)
        dataframe['sma_50'] = ta.SMA(dataframe, timeperiod=50)

        # RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['rsi_fast'] = ta.RSI(dataframe, timeperiod=4)
        dataframe['rsi_slow'] = ta.RSI(dataframe, timeperiod=20)

        # Lambo2
        dataframe = self.lambo2.populate_indicators(dataframe)
        # Elliot
        dataframe = self.elliot.populate_indicators(dataframe)

        # Cofi
        stoch_fast = ta.STOCHF(dataframe, 5, 3, 0, 3, 0)
        dataframe['fastd'] = stoch_fast['fastd']
        dataframe['fastk'] = stoch_fast['fastk']
        dataframe['adx'] = ta.ADX(dataframe)
        dataframe['ema_8'] = ta.EMA(dataframe, timeperiod=8)
        dataframe['ema_21'] = ta.EMA(dataframe, timeperiod=21)
        dataframe['ema_50'] = ta.EMA(dataframe, timeperiod=50)

        condition = dataframe['ema_8'] > dataframe['ema_14']
        percentage_difference = 100 * (dataframe['ema_8'] - dataframe['ema_14']).abs() / dataframe['ema_14']
        dataframe['ema_pct_diff'] = percentage_difference.where(condition, -percentage_difference)
        dataframe['prev_ema_pct_diff'] = dataframe['ema_pct_diff'].shift(1)

        crossover_up = (dataframe['ema_8'].shift(1) < dataframe['ema_14'].shift(1)) & (
                dataframe['ema_8'] > dataframe['ema_14'])

        close_to_crossover_up = (dataframe['ema_8'] < dataframe['ema_14']) & (
                dataframe['ema_8'].shift(1) < dataframe['ema_14'].shift(1)) & (
                                        dataframe['ema_8'] > dataframe['ema_8'].shift(1))

        ema_buy_signal = ((dataframe['ema_pct_diff'] < 0) & (dataframe['prev_ema_pct_diff'] < 0) & (
                dataframe['ema_pct_diff'].abs() < dataframe['prev_ema_pct_diff'].abs()))

        dataframe['ema_diff_buy_signal'] = ((ema_buy_signal | crossover_up | close_to_crossover_up)
                                            & (dataframe['rsi'] <= 55) & (dataframe['volume'] > 0))

        dataframe = self.pump_dump_protection(dataframe, metadata)

        # Calculation of Fibonacci retracement levels
        dataframe['high_max'] = dataframe['high'].rolling(window=30).max()  # posledních 30 svíček
        dataframe['low_min'] = dataframe['low'].rolling(window=30).min()

        # Calculation of Fibonacci levels
        diff = dataframe['high_max'] - dataframe['low_min']
        dataframe['fib_236'] = dataframe['high_max'] - 0.236 * diff
        dataframe['fib_382'] = dataframe['high_max'] - 0.382 * diff
        dataframe['fib_500'] = dataframe['high_max'] - 0.500 * diff
        dataframe['fib_618'] = dataframe['high_max'] - 0.618 * diff
        dataframe['fib_786'] = dataframe['high_max'] - 0.786 * diff

        # MACD a Volatility Factor
        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']

        # Volatility calculation (using rolling standard deviation)
        dataframe['volatility'] = dataframe['close'].rolling(window=14).std()

        # Normalizace volatility
        min_volatility = dataframe['volatility'].rolling(window=14).min()
        max_volatility = dataframe['volatility'].rolling(window=14).max()
        dataframe['volatility_factor'] = (dataframe['volatility'] - min_volatility) / \
                                         (max_volatility - min_volatility)

        # MACD adjustment based on volatility
        dataframe['macd_adjusted'] = dataframe['macd'] * (1 - dataframe['volatility_factor'])
        dataframe['macdsignal_adjusted'] = dataframe['macdsignal'] * (1 + dataframe['volatility_factor'])

        dataframe = self.percentage_drop_indicator(dataframe, 9, threshold=0.21)

        ichi = ichimoku(dataframe)
        dataframe['senkou_span_a'] = ichi['senkou_span_a']
        dataframe['senkou_span_b'] = ichi['senkou_span_b']

        # Creating weights for a weighted average
        weights = np.linspace(1, 0, 300)  # Weights from 1 (newest) to 0 (oldest)
        weights /= weights.sum()  # Normalizing the weights so that their sum is 1

        # Calculation of weighted average RSI for the last 300 candles
        dataframe['weighted_rsi'] = dataframe['rsi'].rolling(window=300).apply(
            lambda x: np.sum(weights * x[-300:]), raw=False
        )

        # Add 'jstkr' signal
        # Produces 1 when the sum of 'macd' and 'macd_signal' is negative and 'rsi' <= 30
        dataframe['jstkr'] = ((dataframe['macd'] + dataframe['macdsignal'] < -0.01) & (dataframe['rsi'] <= 17)).astype(
            int)
        dataframe['jstkr_2'] = ((abs(dataframe['macd'] - dataframe['macdsignal']) / dataframe['macd'].abs() > 0.2) & (
                dataframe['rsi'] <= 25)).astype('int')
        dataframe['jstkr_3'] = ((abs(dataframe['macd'] - dataframe['macdsignal']) / dataframe['macd'].abs() > 0.04) & (
                dataframe['rsi_fast'] <= 10)).astype('int')

        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        # Add timeframe trend
        resampled_frame = dataframe.resample('5T', on='date').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })
        resampled_frame['higher_tf_trend'] = (resampled_frame['close'] > resampled_frame['open']).astype(int)
        resampled_frame['higher_tf_trend'] = resampled_frame['higher_tf_trend'].replace({1: 1, 0: -1})
        dataframe['higher_tf_trend'] = dataframe['date'].map(resampled_frame['higher_tf_trend'])

        dataframe['doji_candle'] = (
                    CDLDOJI(dataframe['open'], dataframe['high'], dataframe['low'], dataframe['close']) > 0).astype(int)

        self.calculate_support_resistance_dicts(metadata['pair'], dataframe)
        dataframe = self.dynamic_stop_loss_take_profit(dataframe=dataframe)

        return dataframe

    def dynamic_stop_loss_take_profit(self, dataframe: DataFrame) -> DataFrame:
        # Příklad: Stop-loss a take-profit založený na ATR
        atr = ta.ATR(dataframe, timeperiod=14)
        dataframe['stop_loss'] = dataframe['low'].shift(1) - atr.shift(1) * 0.8
        dataframe['take_profit'] = dataframe['high'].shift(1) + atr.shift(1) * 2.5
        return dataframe

    def calculate_dca_price(self, base_value, decline, target_percent):
        return (((base_value / 100) * abs(decline)) / target_percent) * 100

    def populate_entry_trend_sr(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Checking the distance to support and resistance
        if metadata['pair'] in self.support_dict and metadata['pair'] in self.resistance_dict:
            supports = self.support_dict[metadata['pair']]
            resistances = self.resistance_dict[metadata['pair']]

            if supports and resistances:
                # Calculating the nearest support and resistance level for each candle
                dataframe['nearest_support'] = dataframe['close'].apply(
                    lambda x: min([support for support in supports if support <= x], default=x,
                                  key=lambda support: abs(x - support))
                )
                dataframe['nearest_resistance'] = dataframe['close'].apply(
                    lambda x: min([resistance for resistance in resistances if resistance >= x], default=x,
                                  key=lambda resistance: abs(x - resistance))
                )

                # Calculation of the percentage difference between the price and the nearest support/resistance
                dataframe['distance_to_support_pct'] = (
                                                               dataframe['nearest_support'] - dataframe['close']) / \
                                                       dataframe['close'] * 100
                dataframe['distance_to_resistance_pct'] = (
                                                                  dataframe['nearest_resistance'] - dataframe[
                                                              'close']) / dataframe['close'] * 100

                # Generating buy signals based on support and resistance
                buy_threshold = 0.1  # 0.1 %
                dataframe.loc[
                    (dataframe['distance_to_support_pct'] >= 0) &
                    (dataframe['distance_to_support_pct'] <= buy_threshold) &
                    (dataframe['distance_to_resistance_pct'] >= buy_threshold),
                    'buy_signal'
                ] = 1

                dataframe.loc[
                    (dataframe['distance_to_support_pct'] >= 0) &
                    (dataframe['distance_to_support_pct'] <= buy_threshold) &
                    (dataframe['distance_to_resistance_pct'] >= buy_threshold),
                    'enter_tag'
                ] += 'sr_buy_mid'

                # Remove helper columns
                dataframe.drop(
                    ['nearest_support', 'nearest_resistance', 'distance_to_support_pct', 'distance_to_resistance_pct'],
                    axis=1, inplace=True)

        # Adding conditions for EMA and volume
        dataframe.loc[(dataframe['volume'] > 0) & (dataframe['ema_diff_buy_signal'].astype(int) > 0), 'buy_ema'] = 1
        dataframe.loc[
            (dataframe['volume'] > 0) & (dataframe['ema_diff_buy_signal'].astype(int) > 0), 'enter_tag'] += 'ema_dbs_'

        # Generating buy signals only if both conditions are met
        dataframe.loc[(dataframe['buy_signal'] == 1) & (dataframe['buy_ema'] == 1) & (
                dataframe['rsi'] <= dataframe['weighted_rsi']), 'enter_long'] = 1

        # Remove helper columns
        if 'buy_support' in dataframe.columns:
            dataframe.drop(['buy_support'], axis=1, inplace=True)
        if 'buy_ema' in dataframe.columns:
            dataframe.drop(['buy_ema'], axis=1, inplace=True)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[:, 'enter_tag'] = ''
        dataframe.loc[:, 'sell_tag'] = ''
        conditions = []

        last_candle = dataframe.iloc[-1]
        if last_candle['doji_candle'] == 1:
            logging.info(f"Doji detected on {metadata['pair']}")
            if not self.is_pair_locked(pair=metadata['pair']):
                self.lock_pair(pair=metadata['pair'], until=datetime.now(timezone.utc) + timedelta(
                    minutes=self.timeframe_to_minutes(self.timeframe) * 10))
            return dataframe

        # Lambo2
        (dataframe, conditions) = self.lambo2.populate_entry_trend(dataframe, conditions)

        # Elliot Waves
        (dataframe, conditions) = self.elliot.populate_entry_trend_v1(dataframe, conditions)
        (dataframe, conditions) = self.elliot.populate_entry_trend_v2(dataframe, conditions)
        (dataframe, conditions) = self.elliot.populate_entry_trend_cofi(dataframe, conditions)

        down_trend = (
            (dataframe['higher_tf_trend'] > 1)
        )
        dataframe.loc[down_trend, 'enter_long'] = 1

        dataframe = self.populate_entry_trend_sr(dataframe=dataframe, metadata=metadata)

        # Checking the distance to the support
        if metadata['pair'] in self.support_dict:
            s = self.support_dict[metadata['pair']]
            if s:
                # Calculating the nearest support level for each candle that is below the current price
                dataframe['nearest_support'] = dataframe['close'].apply(
                    lambda x: min([support for support in s if support <= x], default=x,
                                  key=lambda support: abs(x - support))
                )

                if 'nearest_support' in dataframe.columns:
                    # Calculation of the percentage difference between the price and the nearest support
                    dataframe['distance_to_support_pct'] = (
                                                                   dataframe['nearest_support'] - dataframe['close']) / \
                                                           dataframe['close'] * 100

                    # Generating buy signals based on support
                    buy_threshold = 0.1  # 0.1 %
                    dataframe.loc[
                        (dataframe['distance_to_support_pct'] >= 0) &
                        (dataframe['distance_to_support_pct'] <= buy_threshold),
                        'buy_support'
                    ] = 1

                    dataframe.loc[
                        (dataframe['distance_to_support_pct'] >= 0) &
                        (dataframe['distance_to_support_pct'] <= buy_threshold),
                        'enter_tag'
                    ] += 'sr_buy'

                    # Remove helper columns
                    dataframe.drop(['nearest_support', 'distance_to_support_pct'],
                                   axis=1, inplace=True)

        # Adding conditions for EMA and volume
        dataframe.loc[(dataframe['volume'] > 0) & (dataframe['ema_diff_buy_signal'].astype(int) > 0), 'buy_ema'] = 1
        dataframe.loc[
            (dataframe['volume'] > 0) & (dataframe['ema_diff_buy_signal'].astype(int) > 0), 'enter_tag'] += 'ema_dbs_'

        # Generating buy signals only if both conditions are met
        dataframe.loc[(dataframe['buy_support'] == 1) & (dataframe['buy_ema'] == 1) & (
                dataframe['rsi'] <= dataframe['weighted_rsi']), 'enter_long'] = 1

        # Remove helper columns
        if 'buy_support' in dataframe.columns:
            dataframe.drop(['buy_support'], axis=1, inplace=True)
        if 'buy_ema' in dataframe.columns:
            dataframe.drop(['buy_ema'], axis=1, inplace=True)

        dont_buy_conditions = [
            dataframe['pnd_volume_warn'] < 0.0,
            dataframe['btc_rsi_8_1h'] < 35.0,
            # Dont enter late
            # Given all positions are full and one is closed, we want to open on first buy signal
            (dataframe['enter_long'].shift(1) == 1 & (dataframe['sma_2'].shift(1) < dataframe['sma_2']))
        ]

        if conditions:
            final_condition = reduce(lambda x, y: x | y, conditions)
            dataframe.loc[final_condition, 'enter_long'] = 1
        if dont_buy_conditions:
            for condition in dont_buy_conditions:
                dataframe.loc[condition, 'enter_long'] = 0

        return dataframe

    def calculate_percentage_difference(self, original_price, current_price):
        percentage_diff = ((current_price - original_price) / original_price)
        return percentage_diff

    def pump_dump_protection(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        df36h = dataframe.copy().shift(432)
        df24h = dataframe.copy().shift(288)
        dataframe['volume_mean_short'] = dataframe['volume'].rolling(4).mean()
        dataframe['volume_mean_long'] = df24h['volume'].rolling(48).mean()
        dataframe['volume_mean_base'] = df36h['volume'].rolling(288).mean()
        dataframe['volume_change_percentage'] = (dataframe['volume_mean_long'] / dataframe['volume_mean_base'])
        dataframe['rsi_mean'] = dataframe['rsi'].rolling(48).mean()
        dataframe['pnd_volume_warn'] = np.where((dataframe['volume_mean_short'] / dataframe['volume_mean_long'] > 5.0),
                                                -1, 0)
        return dataframe

    def custom_exit(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                    current_profit: float, **kwargs):
        if current_profit < -0.15 and (current_time - trade.open_date_utc).days >= 60:
            return 'unclog'

    def order_price(self, free_amount, positions, dca_buys):
        total_dca_budget = free_amount - (positions + 1) * dca_buys
        return total_dca_budget / (positions * dca_buys)

    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, '1h') for pair in pairs]

        if self.config['stake_currency'] in ['USDT', 'BUSD', 'USDC', 'DAI', 'TUSD', 'PAX', 'USD', 'EUR', 'GBP']:
            btc_info_pair = f"BTC/{self.config['stake_currency']}"
            if (self.config['trading_mode'] == TradingMode.FUTURES):
                btc_info_pair = btc_info_pair + f":{self.config['stake_currency']}"
        else:
            btc_info_pair = "BTC/USDT"
            if (self.config['trading_mode'] == TradingMode.FUTURES):
                btc_info_pair = btc_info_pair + f":{self.config['stake_currency']}"

        informative_pairs.extend(
            ((btc_info_pair, self.timeframe), (btc_info_pair, self.inf_1h))
        )
        return informative_pairs

    def base_tf_btc_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['price_trend_long'] = (
                dataframe['close'].rolling(8).mean() / dataframe['close'].shift(8).rolling(144).mean())
        ignore_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
        dataframe.rename(columns=lambda s: f"btc_{s}" if s not in ignore_columns else s, inplace=True)
        return dataframe

    def info_tf_btc_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['rsi_8'] = ta.RSI(dataframe, timeperiod=8)
        ignore_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
        dataframe.rename(columns=lambda s: f"btc_{s}" if s not in ignore_columns else s, inplace=True)
        return dataframe

    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                              current_rate: float, current_profit: float, min_stake: float,
                              max_stake: float, **kwargs):

        if current_rate is None:
            return None

        # Update the current time to the current UTC time
        pct_threshold = self.timeframed_drops[self.timeframe]
        logging.info(f"Timeframe: {self.timeframe}, Pct Threshold: {pct_threshold}")
        current_time = datetime.utcnow()

        try:
            # Get an analyzed dataframe for the specified trading pair and timeframe
            dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
            df = dataframe.copy()
        except Exception as e:
            # Log an error if there's an issue retrieving the dataframe and exit the method
            if (log_level.value <= 3): logging.error(f"Error getting analyzed dataframe: {e}")
            return None

        if trade.pair in self.locked:
            return None

        last_candle = df.iloc[-1]

        # Initialize the lowest price for the trade pair if it's not already set
        if trade.pair not in self.lowest_prices:
            self.lowest_prices[trade.pair] = trade.open_rate

        if trade.pair not in self.price_drop_percentage:
            self.price_drop_percentage[trade.pair] = {"last_drop_time": current_time, "last_drop_rate": current_rate}

        # Update the lowest price if the current rate is lower
        if current_rate < self.lowest_prices[trade.pair]:
            self.lowest_prices[trade.pair] = current_rate

        # Calculate the percentage drop from the opening rate
        price_drop = (self.lowest_prices[trade.pair] - trade.open_rate) / trade.open_rate

        # Record the time if the price drop exceeds 5%
        if ((price_drop <= pct_threshold)
                and (self.price_drop_percentage[trade.pair].get("last_drop_time", current_time) != current_time)):

            if "last_drop_rate" in self.price_drop_percentage[trade.pair].keys():
                last = self.price_drop_percentage[trade.pair].get("last_drop_rate")
                if last is not None:
                    if current_rate < last:
                        self.price_drop_percentage[trade.pair]["last_drop_time"] = current_time
                        self.price_drop_percentage[trade.pair]["last_drop_rate"] = current_rate

            # If a 5% drop has been recorded and it's been more than 3 hours since then
            if self.price_drop_percentage[trade.pair].get("last_drop_time") > current_time:
                time_since_last_drop = current_time - self.price_drop_percentage[trade.pair]["last_drop_time"]
                if time_since_last_drop.total_seconds() / 3600 >= 3:
                    logging.info(f"Locking {trade.pair}")
                    self.lock_pair(trade.pair, until=datetime.now(timezone.utc) + timedelta(minutes=24 * 60))
                    self.locked.append(trade.pair)
                    return None  # Avoid further DCA

        last_buy_order = None
        count_of_buys = sum(order.ft_order_side == 'buy' and order.status == 'closed' for order in trade.orders)
        for order in reversed(trade.orders):
            if order.ft_order_side == 'buy' and order.status == 'closed':
                last_buy_order = order
                break

        # Record the time when a 5% drop was first noted
        if trade.pair not in self.price_drop_percentage:
            self.price_drop_percentage[trade.pair] = {"last_drop_time": None, "last_drop_rate": None}

        if self.max_safety_orders >= count_of_buys:
            # Calculate the percentage difference between the last buy order and the current rate
            pct_diff = self.calculate_percentage_difference(original_price=last_buy_order.price,
                                                            current_price=current_rate)
            # Check if the percentage difference is less than the threshold value
            if pct_diff < pct_threshold:
                if last_buy_order and current_rate < last_buy_order.price:
                    # Check RSI conditions for DCA
                    rsi_value = last_candle['rsi']  # Assuming RSI is part of the dataframe
                    w_rsi = last_candle['weighted_rsi']  # Assuming weighted RSI is part of the dataframe
                    if rsi_value <= w_rsi:
                        # Log information about the store
                        if (log_level.value <= 1): logging.info(
                            f'AP1 {trade.pair}, Profit: {current_profit}, Stake {trade.stake_amount}')
                        # Get the total stake amount in the wallet
                        total_stake_amount = self.wallets.get_total_stake_amount()
                        # Calculate the amount for the next bet using DCA (Dollar Cost Averaging)
                        calculated_dca_stake = self.calculate_dca_price(base_value=trade.stake_amount,
                                                                        decline=current_profit * 100,
                                                                        target_percent=1)
                        # Adjust the bet size if it is higher than the available balance
                        while calculated_dca_stake >= total_stake_amount:
                            calculated_dca_stake = calculated_dca_stake / 4
                        # Log information about the adjusted bet
                        if (log_level.value <= 1): logging.info(f'AP2 {trade.pair}, DCA: {calculated_dca_stake}')
                        # Return the adjusted bet size
                        return calculated_dca_stake
        # Return None if no conditions are met to adjust the trade position
        return None

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                            time_in_force: str, current_time: datetime, entry_tag: Optional[str],
                            side: str, **kwargs) -> bool:

        try:
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            df = dataframe.copy()
        except Exception as e:
            if (log_level.value <= 3): logging.error(f"Error getting analyzed dataframe: {e}")
            return None

        result = Trade.get_open_trade_count() < self.out_open_trades_limit
        return result

    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                           rate: float, time_in_force: str, exit_reason: str,
                           current_time: datetime, **kwargs) -> bool:
        exit_reason = f"{exit_reason}_{trade.enter_tag}"

        if 'unclog' in exit_reason or 'force' in exit_reason:
            # logging.info(f"CTE - FORCE or UNCLOG, EXIT")
            return True

        current_profit = trade.calc_profit_ratio(rate)
        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)

        if current_profit >= 0.005 and 'psl' in exit_reason:
            logging.info(f"CTE - PSL EXIT: {pair}, {current_profit}, {rate}, {exit_reason}, {amount}")
            return True

        # Checking if the current high is higher than the open of the last candle
        last_candle = dataframe.iloc[-1]

        # if last_candle['doji_candle'] > 1:
        #     return False

        if last_candle['high'] > last_candle['open']:
            # logging.info(f"CTE - Cena stále roste (high > open), HOLD")
            return False

        # Current EMA values
        ema_8_current = dataframe['ema_8'].iat[-1]
        ema_14_current = dataframe['ema_14'].iat[-1]

        # EMA values e previous candle
        ema_8_previous = dataframe['ema_8'].iat[-2]
        ema_14_previous = dataframe['ema_14'].iat[-2]

        # EMA difference calculation between current and previous candle
        diff_current = abs(ema_8_current - ema_14_current)
        diff_previous = abs(ema_8_previous - ema_14_previous)

        # Calculation of percentage change between diff_current and diff_previous
        diff_change_pct = (diff_previous - diff_current) / diff_previous

        if current_profit >= 0.0025:
            if ema_8_current <= ema_14_current and diff_change_pct >= 0.025:
                # logging.info(f"CTE - EMA 8 {ema_8_current} <= EMA 14 {ema_14_current} with decrease in difference >= 3%, EXIT")
                return True
            elif ema_8_current > ema_14_current and diff_current > diff_previous:
                # logging.info(f"CTE - EMA 8 {ema_8_current} > EMA 14 {ema_14_current} with increasing difference, HOLD")
                return False
            else:
                # logging.info(f"CTE - Conditions not met, EXIT")
                return True
        else:
            return False

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[:, 'exit_tag'] = ''

        dataframe.loc[
            (
                    (dataframe['close'] > dataframe['fib_618']) &
                    (dataframe['sma_50'].shift(1) > dataframe['sma_50'])
            ),
            'exit_tag'] = 'fib_618_sma_50'

        dataframe.loc[
            (
                    (dataframe['close'] > dataframe['fib_618']) &
                    (dataframe['sma_50'].shift(1) > dataframe['sma_50'])
            ),
            'exit_long'] = 1

        dataframe.loc[
            (
                    (dataframe['close'] < dataframe['stop_loss'].shift(1)) |
                    (dataframe['close'] > dataframe['take_profit'].shift(1))
            ),
            'exit_tag'] = 'psl'

        dataframe.loc[
            (
                    (dataframe['close'] < dataframe['stop_loss'].shift(1)) |
                    (dataframe['close'] > dataframe['take_profit'].shift(1))
            ),
            'exit_long'] = 1

        # dataframe.loc[(dataframe['five_max'] == 1), 'exit_long'] = 1
        # dataframe.loc[(dataframe['five_max'] == 1), 'exit_tag'] = 'five_max'

        dataframe.loc[:, 'exit_short'] = 0
        return dataframe

    def percentage_drop_indicator(self, dataframe, period, threshold=0.3):
        # Calculation of the highest price for the last period
        highest_high = dataframe['high'].rolling(period).max()
        # Calculating the percentage drop below the highest price
        percentage_drop = (highest_high - dataframe['close']) / highest_high * 100
        dataframe.loc[percentage_drop < threshold, 'percentage_drop_buy'] = 1
        dataframe.loc[percentage_drop > threshold, 'percentage_drop_buy'] = 0
        return dataframe

    def timeframe_to_minutes(self, timeframe):
        """Converts the timeframe to minutes."""
        if timeframe.endswith('m'):
            return int(timeframe[:-1])
        elif timeframe.endswith('h'):
            return int(timeframe[:-1]) * 60
        elif timeframe.endswith('d'):
            return int(timeframe[:-1]) * 1440
        else:
            raise ValueError("Unknown timeframe: {}".format(timeframe))
