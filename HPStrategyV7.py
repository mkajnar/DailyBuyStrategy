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


class HPStrategyV7(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = '15m'
    leverage_value = 3

    minimal_roi = {
        "0": 0.03
    }

    to_kill = []
    rebuy = {}
    last_sell = {}

    process_only_new_candles = True
    startup_candle_count = 50

    is_opt_sl = True
    sl1 = DecimalParameter(0.01, 0.50, default=0.35, space='sell', decimals=2, optimize=is_opt_sl)
    sl3 = DecimalParameter(0.01, 0.35, default=0.25, space='sell', decimals=2, optimize=is_opt_sl)
    sl5 = DecimalParameter(0.01, 0.25, default=0.2, space='sell', decimals=2, optimize=is_opt_sl)
    sl10 = DecimalParameter(0.01, 0.20, default=0.15, space='sell', decimals=2, optimize=is_opt_sl)
    stoplosses = {1: -sl1.value, 3: -sl3.value, 5: -sl5.value, 10: -sl10.value}

    is_partial_stoploss_used = True
    if is_partial_stoploss_used:
        partial_stoploss_koef = DecimalParameter(0.15, 1, default=0.3, space='sell', decimals=2,
                                                 optimize=is_partial_stoploss_used)
    else:
        partial_stoploss_koef = DecimalParameter(0.15, 1, default=1, space='sell', decimals=2,
                                                 optimize=is_partial_stoploss_used)

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
    max_dca_multiplier = multis[leverage_value]
    open_trade_limit = IntParameter(1, 20, default=20, space='buy', optimize=True)
    is_opt_position_adjustment = True

    position_adjustment_enable = True
    dca_threshold_pct = DecimalParameter(0.01, 0.15, default=0.125, decimals=3, space='buy',
                                         optimize=position_adjustment_enable)

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

    trailing_stop_positive = 0.001
    trailing_stop_positive_offset = 0.008
    stoploss = stoplosses[leverage_value]

    short_window = IntParameter(10, 20, default=5)
    long_window = IntParameter(40, 50, default=9)
    rsi_period = IntParameter(10, 20, default=8)
    rsi_low_threshold = IntParameter(20, 30, default=30)
    rsi_high_threshold = IntParameter(70, 80, default=70)


    fast_ewo = 100
    slow_ewo = 200

    use_exit_signal = True
    ignore_roi_if_entry_signal = False
    use_custom_stoploss = True
    exit_profit_offset = 0.001 * leverage_value
    exit_profit_only = False
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

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                        current_profit: float, **kwargs) -> float:
        """
        Adjusts the stop loss dynamically based on current conditions.
        Parameters:
            pair (str): The trading pair.
            trade (Trade): The trade object.
            current_time (datetime): The current time.
            current_rate (float): The current rate of the trading pair.
            current_profit (float): The current profit of the trade.
            **kwargs: Additional keyword arguments.
        Returns:
            float: The custom stop loss value.
        """
        # Check if the current profit exceeds the positive trailing stop offset
        if current_profit > self.trailing_stop_positive_offset:
            new_stop_loss = max(current_profit - self.trailing_stop_positive, self.stoploss)
            return new_stop_loss
        # Calculate the number of DCA entries
        dca_count = trade.nr_of_successful_entries - trade.nr_of_successful_exits + 1
        # Check if the trade has reached the maximum entry position adjustment and exceeded the maximum hold time
        if (dca_count >= self.max_entry_position_adjustment) and \
                (current_time - trade.open_date_utc > datetime.timedelta(hours=self.max_hold_time.value)):
            self.to_kill.append(trade.id)
            return 1
        # Adjust the stop loss if the current profit is positive and the local minimum is lower than the current rate
        if current_profit > 0:
            dataframe = self.dp.get_pair_dataframe(pair=pair, timeframe=self.timeframe)
            local_min = dataframe['low'].rolling(window=20).min().iloc[-1]
            if local_min < current_rate:
                stoploss_gap = (current_rate - local_min) / current_rate
                custom_stoploss_value = -abs(stoploss_gap)
                logging.info(
                    f"XXXXXXX CSL SETTING SL {custom_stoploss_value} for pair {pair} at rate {current_rate} - S4")
                return custom_stoploss_value
        # If none of the above conditions are met, return the default stop loss value
        return self.stoploss

    def calculate_heiken_ashi(self, dataframe):
        """
        Calculate Heiken Ashi candles based on the provided dataframe.
        Parameters:
            dataframe (pd.DataFrame): The input dataframe containing OHLC data.
        Returns:
            pd.DataFrame: A dataframe containing Heiken Ashi candle data.
        """
        # Check if the dataframe is empty
        if dataframe.empty:
            raise ValueError("DataFrame is empty")
        # Initialize a new dataframe for Heiken Ashi candles
        heiken_ashi = pd.DataFrame(index=dataframe.index)
        # Calculate Heiken Ashi close
        heiken_ashi['HA_Close'] = (dataframe['open'] + dataframe['high'] + dataframe['low'] + dataframe['close']) / 4
        # Calculate Heiken Ashi open (first row should be the same as close)
        heiken_ashi['HA_Open'] = heiken_ashi['HA_Close'].shift(1)
        heiken_ashi['HA_Open'].iloc[0] = heiken_ashi['HA_Close'].iloc[0]
        # Calculate Heiken Ashi high and low
        heiken_ashi['HA_High'] = heiken_ashi[['HA_Open', 'HA_Close']].join(dataframe[['high']], how='inner').max(axis=1)
        heiken_ashi['HA_Low'] = heiken_ashi[['HA_Open', 'HA_Close']].join(dataframe[['low']], how='inner').min(axis=1)
        # Apply rolling mean
        rolling_window = self.rolling_ha_treshold.value
        heiken_ashi['HA_Close'] = heiken_ashi['HA_Close'].rolling(window=rolling_window).mean()
        heiken_ashi['HA_Open'] = heiken_ashi['HA_Open'].rolling(window=rolling_window).mean()
        heiken_ashi['HA_High'] = heiken_ashi['HA_High'].rolling(window=rolling_window).mean()
        heiken_ashi['HA_Low'] = heiken_ashi['HA_Low'].rolling(window=rolling_window).mean()
        return heiken_ashi

    def should_already_sell(self, dataframe):
        """
        Determine if the strategy should sell based on Heiken Ashi candles.
        Parameters:
            dataframe (pd.DataFrame): The input dataframe containing OHLC data.
        Returns:
            bool: True if the strategy should sell, False otherwise.
        """
        # Calculate Heiken Ashi candles
        heiken_ashi = self.calculate_heiken_ashi(dataframe)
        # Extract the last candle
        last_candle = heiken_ashi.iloc[-1].squeeze()
        # Check if the Heiken Ashi close is less than the open
        if last_candle['HA_Close'] > last_candle['HA_Open']:
            return False  # No sell signal
        else:
            return True  # Sell signal

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
        return Trade.get_open_trade_count() < self.open_trade_limit.value

    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                           rate: float, time_in_force: str, exit_reason: str,
                           current_time: datetime, **kwargs) -> bool:
        """
        Confirm whether a trade should be exited.
        Parameters:
            pair (str): The trading pair.
            trade (Trade): The trade object.
            order_type (str): The type of order.
            amount (float): The amount of the trade.
            rate (float): The rate of the trade.
            time_in_force (str): The time in force of the order.
            exit_reason (str): The reason for the trade exit.
            current_time (datetime): The current time.
            **kwargs: Additional keyword arguments.
        Returns:
            bool: True if the trade should be exited, False otherwise.
        """
        # Get the analyzed dataframe for the pair and timeframe
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        # Calculate the profit ratio of the trade at the current rate
        profit_ratio = trade.calc_profit_ratio(rate)
        # Check if the profit ratio is negative and the trade ID is in the list to kill
        if (profit_ratio < 0) and (trade.id in self.to_kill):
            self.to_kill.remove(trade.id)
            return True
        # Check if the exit reason contains 'swing' or 'trailing'
        if 'swing' in exit_reason or 'trailing' in exit_reason:
            return profit_ratio > 0
        # Check if the exit reason contains 'roi' and the profit ratio is positive
        # or if the strategy should already sell based on Heiken Ashi
        if 'roi' in exit_reason:
            return profit_ratio > 0 or self.should_already_sell(dataframe=dataframe)
        # If none of the above conditions are met, return True
        return True

    def ewo(self, dataframe, ema_length=5, ema2_length=35):
        df = dataframe.copy()
        ema1 = ta.EMA(df, timeperiod=ema_length)
        ema2 = ta.EMA(df, timeperiod=ema2_length)
        emadif = (ema1 - ema2) / df['close'] * 100
        return emadif

    def identify_swing_trend(self, dataframe: DataFrame, lookback: int) -> DataFrame:
        """
        Identify swing trends in the provided dataframe.
        Parameters:
            dataframe (DataFrame): The input dataframe containing OHLC data.
            lookback (int): The number of periods to look back for swing trend identification.
        Returns:
            DataFrame: A dataframe containing swing trend identification.
        """
        # Initialize a new column for swing trend
        dataframe['trend'] = 0
        # Initialize variables for last swing high and low
        last_swing_high = last_swing_low = np.nan
        # Optimize: Get close prices outside the loop
        close_prices = dataframe['close']
        # Loop through the dataframe
        for i in range(lookback, len(dataframe) - lookback):
            # Optimize: Reduce data access by calculating max and min range once
            max_range = close_prices[i - lookback:i + lookback + 1].max()
            min_range = close_prices[i - lookback:i + lookback + 1].min()
            # Optimize: Reduce data access by storing current close price
            current_close = close_prices[i]
            # Check if the current close price is the max range
            if current_close == max_range:
                # Check if last swing high is NaN or current close is greater than last swing high
                if np.isnan(last_swing_high) or current_close > last_swing_high:
                    # Optimize: Use .loc for correct value assignment
                    dataframe.loc[i:, 'trend'] = 1
                    # Optimize: Store current close as last swing high
                    last_swing_high = current_close
            # Check if the current close price is the min range
            if current_close == min_range:
                # Check if last swing low is not NaN and current close is less than last swing low
                if not np.isnan(last_swing_low) and current_close < last_swing_low:
                    # Optimize: Use .loc for correct value assignment
                    dataframe.loc[i:, 'trend'] = 0
                    # Optimize: Store current close as last swing low
                    last_swing_low = current_close
        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Identifikace swing trendu
        dataframe = self.identify_swing_trend(dataframe, 5)
        # Swing high/low
        dataframe['swing_low'] = (dataframe['close'].shift(2) > dataframe['close'].shift(1)) & \
                                 (dataframe['close'].shift(1) < dataframe['close']).astype(int)
        dataframe['swing_high'] = (dataframe['close'].shift(2) < dataframe['close'].shift(1)) & \
                                  (dataframe['close'].shift(1) > dataframe['close']).astype(int)
        # CCI, RSI, ATR, SMA
        dataframe['cci'] = ta.CCI(dataframe, timeperiod=14)
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        dataframe['sma_15'] = ta.SMA(dataframe, timeperiod=15)
        # CTI
        dataframe['cti'] = pta.cti(dataframe["close"], length=20)
        # RSI Fast/Slow
        dataframe['rsi_fast'] = ta.RSI(dataframe, timeperiod=4)
        dataframe['rsi_slow'] = ta.RSI(dataframe, timeperiod=20)
        # STOCHF Fastk
        stoch_fast = ta.STOCHF(dataframe, 5, 3, 0, 3, 0)
        dataframe['fastk'] = stoch_fast['fastk']
        # EMA
        dataframe['ema_8'] = ta.EMA(dataframe, timeperiod=8)
        dataframe['ema_16'] = ta.EMA(dataframe, timeperiod=16)
        # STOCHF Fastd
        stoch_fast = ta.STOCHF(dataframe, 5, 5, 1, 5, 1)
        dataframe['fastd'] = stoch_fast['fastd']
        # Bollinger Bands
        bollinger2 = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband2'] = bollinger2['lower']
        dataframe['bb_middleband2'] = bollinger2['mid']
        dataframe['bb_upperband2'] = bollinger2['upper']
        dataframe['bb_width'] = (dataframe['bb_upperband2'] - dataframe['bb_lowerband2']) / dataframe['bb_middleband2']
        # Rolling mean volume
        dataframe['volume_mean_12'] = dataframe['volume'].rolling(12).mean().shift(1)
        dataframe['volume_mean_24'] = dataframe['volume'].rolling(24).mean().shift(1)
        # MFI
        dataframe['mfi'] = ta.MFI(dataframe, timeperiod=14)
        # Market Capitalization
        dataframe['cap'] = dataframe['volume'].shift(0) * dataframe['close'].shift(0)

        self.calculate_moving_averages(dataframe)
        self.calculate_rsi(dataframe)
        self.detect_swings(dataframe)

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

    # Metoda pro výpočet klouzavých průměrů
    def calculate_moving_averages(self, data):
        data['Short_MA'] = data['close'].rolling(window=self.short_window.value, min_periods=1).mean()
        data['Long_MA'] = data['close'].rolling(window=self.long_window.value, min_periods=1).mean()

    # Metoda pro výpočet RSI
    def calculate_rsi(self, data):
        data['RSI'] = ta.RSI(data['close'], timeperiod=self.rsi_period.value)

    # Metoda pro detekci swing-low a swing-high
    def detect_swings(self, data):
        data['Swing_Low'] = np.where((data['low'].shift(1) > data['low']) & (data['low'].shift(-1) > data['low']), True,
                                     False).astype(int)
        data['Swing_High'] = np.where((data['high'].shift(1) < data['high']) & (data['high'].shift(-1) < data['high']),
                                      True, False).astype(int)

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        # conditions = []
        # dataframe.loc[:, 'enter_tag'] = ''
        # buy_1 = (
        #         ((dataframe['rsi_slow'] < dataframe['rsi_slow'].shift(1)) &
        #          (dataframe['rsi_fast'] < self.buy_rsi_fast_32.value) &
        #          (dataframe['rsi'] > self.buy_rsi_32.value) &
        #          (dataframe['close'] < dataframe['sma_15'] * self.buy_sma15_32.value) &
        #          (dataframe['cti'] < self.buy_cti_32.value)) |
        #         (qtpylib.crossed_above(dataframe['rsi_fast'], dataframe['rsi_slow']))
        # )
        # conditions.append(buy_1)
        # dataframe.loc[buy_1, 'enter_tag'] += 'buy_1'
        # if conditions:
        #     dataframe.loc[
        #         reduce(lambda x, y: x | y, conditions),
        #         'enter_long'] = 1

        # dataframe.loc[
        #     (dataframe['swing_low'] > 0) & (dataframe['cci'] <= -100),
        #     ['enter_long', 'enter_tag']
        # ] = (1, 'swing_low')

        dataframe.loc[
            ((dataframe['Swing_Low'] > 0).rolling(2).sum() > 0) & (dataframe['cci'] <= -100),
            ['enter_long', 'enter_tag']
        ] = (1, 'swing_low')
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # dataframe.loc[(dataframe['cci'] > 100) &
        #               (dataframe['rsi'] > 80) &
        #               (dataframe['volume'] > 0), ['exit_long', 'exit_tag']] = (1, 'cci_sell')

        # dataframe.loc[(dataframe['swing_high'] > 0), ['exit_long', 'exit_tag']] = (1, 'swing_high')
        dataframe.loc[(dataframe['Swing_High'].rolling(2).sum() > 0), ['exit_long', 'exit_tag']] = (1, 'swing_high')
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

    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: Optional[float], max_stake: float,
                            leverage: float, entry_tag: Optional[str], side: str,
                            **kwargs) -> float:
        # Zajištění, že proposed_stake nebude nižší než min_stake
        if min_stake is not None and proposed_stake < min_stake:
            return min_stake
        else:
            return proposed_stake / self.max_dca_multiplier

    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                              current_rate: float, current_profit: float,
                              min_stake: Optional[float], max_stake: float,
                              current_entry_rate: float, current_exit_rate: float,
                              current_entry_profit: float, current_exit_profit: float,
                              **kwargs) -> Optional[float]:
        """
        Adjusts the trade position based on current conditions.

        Parameters:
            trade (Trade): The trade object.
            current_time (datetime): The current time.
            current_rate (float): The current rate of the trading pair.
            current_profit (float): The current profit of the trade.
            min_stake (Optional[float]): The minimum stake allowed.
            max_stake (float): The maximum stake allowed.
            current_entry_rate (float): The entry rate of the trade.
            current_exit_rate (float): The exit rate of the trade.
            current_entry_profit (float): The profit at entry.
            current_exit_profit (float): The profit at exit.
            **kwargs: Additional keyword arguments.

        Returns:
            Optional[float]: The adjusted stake amount or None.
        """

        try:
            dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
            # Extract the last and previous candles from the dataframe
            last_candle = dataframe.iloc[-1].squeeze()
            previous_candle = dataframe.iloc[-2].squeeze()

            if (last_candle['swing_low'] == 0
                    and previous_candle['swing_low'] == 0
                    and self.is_pair_locked(trade.pair)):
                return None
            else:
                self.unlock_pair(trade.pair)

            # Calculate the count of entries and select filled orders
            count_of_entries = trade.nr_of_successful_entries - trade.nr_of_successful_exits
            filled_entries = trade.select_filled_orders(trade.entry_side)

            # If there are filled entries, proceed
            if filled_entries:
                first = filled_entries[0]
                if first:
                    stake_amount = first.stake_amount
                    # Check if the trade pair is marked for rebuy
                    if trade.pair in self.rebuy:
                        if last_candle['swing_low'] > 0 or previous_candle['swing_low'] > 0:
                            s = self.rebuy.pop(trade.pair)
                            logging.info(f"***** ATP REBUY Increasing {trade.pair} position for {s} *****")
                            self.unlock_pair(trade.pair)
                            self.lock_pair(trade.pair, current_time + datetime.timedelta(minutes=10), reason='rebuy',
                                           side='*')
                            return abs(s)
                        else:
                            return None
                    else:
                        # Check if current profit is below partial stoploss
                        if current_profit < 0 and current_profit < self.stoploss * self.partial_stoploss_koef.value:
                            self.last_dca_timeframe[trade.id] = current_time
                            logging.info(
                                f"***** ATP LOSS Lowering {trade.pair} position for {-(trade.stake_amount / 2)} *****")
                            self.rebuy[trade.pair] = stake_amount / 3
                            self.unlock_pair(trade.pair)
                            self.lock_pair(trade.pair, current_time + datetime.timedelta(minutes=10),
                                           reason='partial_stoploss', side='*')
                            return -(stake_amount / 2)
                    # Check if current profit is above partial profit threshold and no successful exits have been made
                    if current_profit > (
                            self.dca_threshold_pct.value * count_of_entries) and trade.nr_of_successful_exits == 0:
                        self.unlock_pair(trade.pair)
                        self.lock_pair(trade.pair, current_time + datetime.timedelta(minutes=3),
                                       reason='partial_profit', side='*')
                        return -(stake_amount / 2)
                    # Check if enough time has passed since the last DCA
                    if trade.id in self.last_dca_timeframe:
                        td = current_time - self.last_dca_timeframe[trade.id]
                        if (td.total_seconds() / 60) < (
                                self.timeframe_to_minutes(self.timeframe) * self.candles_before_dca.value):
                            return None
                    # Check if maximum entry position adjustment reached or other conditions are met
                    if count_of_entries >= self.max_entry_position_adjustment or last_candle['close'] < previous_candle[
                        'close'] or current_profit > -self.dca_threshold_pct.value:
                        return None
                    # Adjust stake amount if current profit is below DCA threshold
                    if current_profit < -self.dca_threshold_pct.value * count_of_entries:
                        try:
                            self.last_dca_timeframe[trade.id] = current_time
                            stake_amount = stake_amount * (1 + (count_of_entries * 0.25))
                        except Exception as exception:
                            stake_amount = 5
                            pass
                        if last_candle['swing_low'] > 0 or previous_candle['swing_low'] > 0:
                            self.unlock_pair(trade.pair)
                            self.lock_pair(trade.pair, current_time + datetime.timedelta(minutes=3),
                                           reason='dca_applied',
                                           side='*')
                            return stake_amount
                        else:
                            return None
                return None
        except Exception as e:
            # logging.error(f"Error in adjust_trade_position: {e}")
            return None

    def timeframe_to_minutes(self, timeframe: str) -> int:
        if 'm' in timeframe:
            return int(timeframe.replace('m', ''))
        elif 'h' in timeframe:
            return int(timeframe.replace('h', '')) * 60
        elif 'd' in timeframe:
            return int(timeframe.replace('d', '')) * 1440
        else:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
