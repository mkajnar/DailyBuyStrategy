import logging
import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame
from typing import Optional, Union
from freqtrade.persistence import Trade, Order
from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,
                                IStrategy, IntParameter)
# --------------------------------
# Add your lib to import here
import datetime
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import pandas_ta as pta


# This class is a sample. Feel free to customize it.
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
    open_trade_limit = 6
    position_adjustment_enable = True
    dca_threshold_pct = DecimalParameter(0.01, 0.20, default=0.03 * leverage_value, decimals=2, space='buy',
                                         optimize=position_adjustment_enable)
    candles_before_dca = IntParameter(1, 10, default=5, space='buy', optimize=True)

    rolling_ha_treshold = IntParameter(3, 10, default=7, space='buy', optimize=True)
    trailing_stop = True
    trailing_only_offset_is_reached = True
    trailing_stop_positive = 0.003
    trailing_stop_positive_offset = 0.01
    stoploss = -0.05 * leverage_value
    use_exit_signal = True
    ignore_roi_if_entry_signal = True
    use_custom_stoploss = True
    # exit_profit_offset = 0.001 * leverage_value
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

    def calculate_heiken_ashi(self, dataframe):
        if dataframe.empty:
            raise ValueError("DataFrame je prázdný")
        heiken_ashi = pd.DataFrame(index=dataframe.index)
        heiken_ashi['HA_Close'] = (dataframe['open'] + dataframe['high'] + dataframe['low'] + dataframe['close']) / 4
        heiken_ashi['HA_Open'] = heiken_ashi['HA_Close'].shift(1)
        heiken_ashi['HA_Open'].iloc[0] = heiken_ashi['HA_Close'].iloc[0]
        heiken_ashi['HA_High'] = heiken_ashi[['HA_Open', 'HA_Close']].join(dataframe['high'], how='inner').max(axis=1)
        heiken_ashi['HA_Low'] = heiken_ashi[['HA_Open', 'HA_Close']].join(dataframe['low'], how='inner').min(axis=1)
        # Aplikace klouzavého průměru
        heiken_ashi['HA_Close'] = heiken_ashi['HA_Close'].rolling(window=self.rolling_ha_treshold.value).mean()
        heiken_ashi['HA_Open'] = heiken_ashi['HA_Open'].rolling(window=self.rolling_ha_treshold.value).mean()
        heiken_ashi['HA_High'] = heiken_ashi['HA_High'].rolling(window=self.rolling_ha_treshold.value).mean()
        heiken_ashi['HA_Low'] = heiken_ashi['HA_Low'].rolling(window=self.rolling_ha_treshold.value).mean()

        return heiken_ashi

    def should_already_sell(self, dataframe):
        heiken_ashi = self.calculate_heiken_ashi(dataframe)
        last_candle = heiken_ashi.iloc[-1]
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
        return (Trade.get_open_trade_count() < self.open_trade_limit)

    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                           rate: float, time_in_force: str, exit_reason: str,
                           current_time: datetime, **kwargs) -> bool:
        force_reasons = ['force_sell', 'force_exit']
        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        if 'trailing' in exit_reason:
            return rate > trade.open_rate + self.trailing_stop_positive_offset * self.leverage_value
        if exit_reason in force_reasons:
            return True
        should_already_sell = self.should_already_sell(dataframe)
        if should_already_sell:
            return True
        return False

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                        current_profit: float, **kwargs) -> float:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        atr = dataframe.iloc[-1]['atr']
        atr_multiplier = 3
        stop_loss_atr = atr * atr_multiplier
        stop_loss_percentage = -stop_loss_atr / current_rate
        return max(stop_loss_percentage, self.stoploss)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['cci'] = ta.CCI(dataframe, timeperiod=14)
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[(dataframe['cci'] < -100) &
                      (dataframe['rsi'] < 30) &
                      (dataframe['volume'] > 0), ['enter_long', 'enter_tag']] = (1, 'cci_buy')
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[(dataframe['cci'] > 100) &
                      (dataframe['rsi'] > 70) &
                      (dataframe['volume'] > 0), ['exit_long', 'exit_tag']] = (1, 'cci_sell')
        return dataframe

        # This is called when placing the initial order (opening trade)

    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: Optional[float], max_stake: float,
                            leverage: float, entry_tag: Optional[str], side: str,
                            **kwargs) -> float:
        return proposed_stake / self.max_dca_multiplier

    # Assuming you have added this attribute at the strategy class level
    # self.last_dca_timeframe = {}

    """ def adjust_trade_position(self, trade: Trade, current_time: datetime,
                              current_rate: float, current_profit: float,
                              min_stake: Optional[float], max_stake: float,
                              current_entry_rate: float, current_exit_rate: float,
                              current_entry_profit: float, current_exit_profit: float,
                              **kwargs) -> Optional[float]:

        # Ensure last_dca_time is timezone-aware, defaulting to UTC if not present
        last_dca_time = self.last_dca_timeframe.get(trade.id,
                                                    datetime.datetime.min.replace(tzinfo=datetime.timezone.utc))
        if current_time - last_dca_time < datetime.timedelta(minutes=self.timeframe_to_minutes(self.timeframe)):
            return None  # Skip DCA if already done in the current timeframe

        if current_profit > self.dca_threshold_pct.value and trade.nr_of_successful_exits == 0:
            # Take half of the profit at +5%
            return -(trade.stake_amount / 2)
        if current_profit > -self.dca_threshold_pct.value:
            return None

        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        # Only buy when not actively falling price.
        last_candle = dataframe.iloc[-1].squeeze()
        previous_candle = dataframe.iloc[-2].squeeze()
        if last_candle['close'] < previous_candle['close']:
            return None

        filled_entries = trade.select_filled_orders(trade.entry_side)
        count_of_entries = trade.nr_of_successful_entries
        try:
            stake_amount = filled_entries[0].stake_amount
            stake_amount = stake_amount * (1 + (count_of_entries * 0.25))
            # Update the last DCA timeframe for this trade
            self.last_dca_timeframe[trade.id] = current_time
            return stake_amount
        except Exception as exception:
            return None
        return None
    """

    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                              current_rate: float, current_profit: float,
                              min_stake: Optional[float], max_stake: float,
                              current_entry_rate: float, current_exit_rate: float,
                              current_entry_profit: float, current_exit_profit: float,
                              **kwargs) -> Optional[float]:

        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()
        previous_candle = dataframe.iloc[-2].squeeze()

        if (current_profit > self.dca_threshold_pct.value and
                trade.nr_of_successful_exits == 0):
            return -(trade.stake_amount / 2)

        if trade.id in self.last_dca_timeframe.keys():
            td = current_time - self.last_dca_timeframe[trade.id]
            if ((td.total_seconds() / 60)
                    < (self.timeframe_to_minutes(self.timeframe) * self.candles_before_dca.value)):
                return None

        filled_entries = trade.select_filled_orders(trade.entry_side)
        count_of_entries = trade.nr_of_successful_entries

        if ((count_of_entries >= self.max_entry_position_adjustment) or
                (last_candle['close'] < previous_candle['close']) or
                (current_profit > -self.dca_threshold_pct.value)):
            return None

        averaged_stake = (sum([entry.stake_amount for entry in filled_entries
                               if entry.stake_amount > 0]) / count_of_entries)
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
