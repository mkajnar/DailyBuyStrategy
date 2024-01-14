import datetime
import logging
from datetime import datetime
from datetime import timedelta, timezone
from functools import reduce
from typing import Optional
import numpy as np
import talib.abstract as ta
from pandas import DataFrame
from talib import CDLDOJI, MFI
import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.persistence import Trade
from freqtrade.strategy import DecimalParameter, IntParameter
from freqtrade.strategy.interface import IStrategy


def loginfo(param):
    # logging.info(param)
    pass
def logerror(param):
    logerror(param)
    pass
def logwarning(param):
    logwarning(param)
    pass


class STPyramideStrategy(IStrategy):
    INTERFACE_VERSION = 3
    position_adjustment_enable = True
    trailing_stop = True
    # out_open_trades_limit = 10
    trailing_stop_positive = 0.005
    trailing_stop_positive_offset = 0.01
    trailing_only_offset_is_reached = True
    ignore_roi_if_entry_signal = True
    exit_profit_only = False
    last_adjustment_time = {}
    order_openrates = {}
    minimal_roi = {
        "0": 0.10
    }
    stoploss = -0.05

    timeframed_drops = {
        '1m': -0.01,
        '5m': -0.05,
        '15m': -0.05,
        '30m': -0.075,
        '1h': -0.1,
        '2h': -0.1,
        '4h': -0.1,
        '8h': -0.1,
        '1d': -0.1
    }

    rsi = IntParameter(10, 60, default=30, space='buy', optimize=True)
    decrease_percentage = DecimalParameter(0.01, 0.1, default=0.05, space='buy', optimize=True)
    increase_percentage = DecimalParameter(0.01, 0.1, default=0.05, space='buy', optimize=True)
    stoploss1 = DecimalParameter(0.05, 2.0, default=0.25, space='buy', optimize=True)
    takeprofit1 = DecimalParameter(0.05, 2.0, default=1.5, space='buy', optimize=True)
    stoploss2 = DecimalParameter(0.05, 2.0, default=0.4, space='buy', optimize=True)
    takeprofit2 = DecimalParameter(0.05, 2.0, default=1.25, space='buy', optimize=True)
    profit_treshold = DecimalParameter(0.001, 0.01, default=0.005, space='sell', optimize=True)
    downtrend_max_candles = IntParameter(5, 30, default=10, space='buy', optimize=False)
    downtrend_pct_treshold = DecimalParameter(0.5, 5, default=0.75, space='buy', optimize=False)



    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = self.detect_hammer_doji(dataframe=dataframe)
        # EMA
        dataframe['ema_fast'] = ta.EMA(dataframe, timeperiod=5)
        dataframe['ema_slow'] = ta.EMA(dataframe, timeperiod=10)
        # RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe = self.prepare_psl(dataframe=dataframe)
        try:
            self.detect_trend(dataframe=dataframe)
            self.get_max_drawdown(dataframe=dataframe)
        except Exception as ex:
            logerror(str(ex))
        return dataframe

    def get_max_drawdown(self, dataframe):
        """
        Calculate the maximum drawdown of a given dataframe.

        Args:
            dataframe (pandas.DataFrame): The input dataframe containing the financial data.

        Returns:
            None

        Raises:
            None

        """
        t = self.downtrend_pct_treshold.value
        try:
            s = self.timeframed_drops.get(self.timeframe, '1h') * t
            dataframe['max_drawdown'] = s
            df = dataframe[-200:]
            cumulative_max = df['close'].cummax()
            if cumulative_max:
                drawdowns = (df['close'] - cumulative_max) / cumulative_max
                max_drawdown = drawdowns.min()
                if -max_drawdown > self.timeframed_drops.get(self.timeframe, '1h'):
                    dataframe['max_drawdown'] = s
                else:
                    dataframe['max_drawdown'] = -max_drawdown * t
            else:
                dataframe['max_drawdown'] = s
        except Exception as ex:
            # logerror(str(ex))
            pass
    def detect_trend(self, dataframe):
        """
        Sets the 'trend' column in the given dataframe to indicate whether the data shows an uptrend or a downtrend.

        Args:
            dataframe (pandas.DataFrame): The dataframe containing the data.

        Returns:
            None
        """
        dataframe['trend'] = 'downtrend'
        try:
            x = self.downtrend_max_candles.value
            aggregated_candle = {
                'open': dataframe['open'].iloc[-x],
                'high': dataframe['high'].iloc[-x:].max(),
                'low': dataframe['low'].iloc[-x:].min(),
                'close': dataframe['close'].iloc[-1]
            }
            if aggregated_candle['close'] > aggregated_candle['open']:
                dataframe['trend'] = 'uptrend'
            else:
                dataframe['trend'] = 'downtrend'
        except Exception as ex:
            # logerror(str(ex))
            pass

    def detect_hammer_doji(self, dataframe: DataFrame):
        """
        Detects Hammer and Doji candlesticks in the given DataFrame.
        Args:
            dataframe (DataFrame): The input DataFrame containing candlestick data.
        Returns:
            DataFrame: The input DataFrame with additional columns indicating the presence of Hammer and Doji candlesticks.
        """
        hammer = ta.CDLHAMMER(dataframe['open'], dataframe['high'], dataframe['low'], dataframe['close'])
        doji = ta.CDLDOJI(dataframe['open'], dataframe['high'], dataframe['low'], dataframe['close'])
        dataframe['is_hammer'] = (hammer > 0)
        dataframe['is_doji'] = (doji > 0)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                    (dataframe['ema_fast'] > dataframe['ema_slow']) &  # EMA Fast je nad EMA Slow
                    (dataframe['rsi'] < 70)  # RSI je pod 70
            ),
            'enter_long'] = 1
        return dataframe


    def timeframe_to_minutes(self, timeframe):
        """
        Convert the timeframe to minutes
        :param timeframe:
        :return:

        This code snippet defines a function called timeframe_to_minutes that
        takes in a timeframe as input. It checks the last character of the timeframe
        to determine its unit (minutes, hours, or days) and converts the numerical
        value to minutes accordingly. If the timeframe ends with 'm', it returns the
        numerical value as an integer. If it ends with 'h', it multiplies the numerical
        value by 60 and returns the result. If it ends with 'd', it multiplies the numerical
        value by 1440 (24 hours * 60 minutes) and returns the result.
        If the timeframe does not end with any of these units, it raises
        a ValueError with an appropriate error message.
        """
        if timeframe.endswith('m'):
            return int(timeframe[:-1])
        elif timeframe.endswith('h'):
            return int(timeframe[:-1]) * 60
        elif timeframe.endswith('d'):
            return int(timeframe[:-1]) * 1440
        else:
            raise ValueError("Unknown timeframe: {}".format(timeframe))

    def calculate_release_percentage(self, pair, current_rate, open_rate, stake_amount, pct_threshold,
                                     release_percentage):
        """
        Calculate the release percentage based on the provided parameters.
        Parameters:
            pair (str): The trading pair.
            current_rate (float): The current rate of the pair.
            open_rate (float): The rate at which the pair was opened.
            stake_amount (float): The amount of stake.
            pct_threshold (float): The percentage threshold.
            release_percentage (float): The release percentage.
        Returns:
            float: The amount to release based on the conditions specified.

        This code snippet defines a function called calculate_release_percentage.
        It takes several parameters including pair, current_rate, open_rate, stake_amount, pct_threshold, and release_percentage.
        The function calculates the price_loss by subtracting the current_rate from the open_rate and dividing it by the open_rate.
        If the price_loss is greater than the pct_threshold, it calculates the amount_to_release by multiplying the stake_amount by
        the negative value of release_percentage. It then checks if the absolute value of amount_to_release is less than or equal
        to stake_amount. If it is, it assigns amount_to_release to for_free, otherwise it assigns the negative value of stake
        amount to for_free. It logs a message indicating the adjustment made and returns for_free.
        If the price_loss is not greater than the pct_threshold, the function returns 0.
        """
        price_loss = (open_rate - current_rate) / open_rate
        if price_loss > pct_threshold:
            amount_to_release = stake_amount * -release_percentage
            for_free = amount_to_release if abs(amount_to_release) <= stake_amount else -stake_amount
            loginfo(f"Adjusting trade - free {for_free} from {pair}")
            return for_free
        return 0

    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                              current_rate: float, current_profit: float, min_stake: float,
                              max_stake: float, **kwargs):
        self.order_openrates[trade.pair] = trade.open_rate
        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        last_candle = dataframe.iloc[-1]
        pct_threshold = last_candle['max_drawdown']

        if current_time.tzinfo is not None and current_time.tzinfo.utcoffset(current_time) is not None:
            current_time = current_time.replace(tzinfo=None)

        last_adjustment = self.last_adjustment_time.get(trade.pair, datetime.min)
        if current_time - last_adjustment < timedelta(minutes=self.timeframe_to_minutes(self.timeframe)):
            return None

        if last_candle['trend'] == 'downtrend':
            total_stake_amount = self.wallets.get_total_stake_amount()

            if total_stake_amount <= 0:

                loginfo(f"Adjusting position down for redistribute {trade.pair}")

                return self.calculate_release_percentage(pair=trade.pair, current_rate=current_rate,
                                                         open_rate=trade.open_rate,
                                                         stake_amount=trade.stake_amount, pct_threshold=pct_threshold,
                                                         release_percentage=0.15)
            return None

        rsi = last_candle['rsi']
        if rsi < self.rsi.value:
            decrease_percentage = self.decrease_percentage.value
            increase_percentage = self.increase_percentage.value
            if (current_rate - last_candle['close']) / current_rate >= decrease_percentage:
                new_amount = min(trade.amount, max_stake)
                loginfo(f"Adjusting position for {trade.pair} profit to {new_amount}")
                self.last_adjustment_time[trade.pair] = current_time
                return new_amount
            if (last_candle['close'] - current_rate) / current_rate >= increase_percentage:
                new_amount = min(trade.amount, max_stake)
                loginfo(f"Adjusting position for {trade.pair} profit to {new_amount}")
                self.last_adjustment_time[trade.pair] = current_time
                return new_amount

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                            time_in_force: str, current_time: datetime, entry_tag: Optional[str],
                            side: str, **kwargs) -> bool:
        confirm = True
        open_rate = self.order_openrates.get(pair, None)
        if open_rate is not None:
            confirm = open_rate < rate

        # DO NOT LIMIT!!!!!!!!!!!!!!!!!!!!!!
        # confirm = confirm and (Trade.get_open_trade_count() <= self.out_open_trades_limit)
        # if confirm:
        #     loginfo(f"Confirmed trade for {pair}, {order_type}, {amount}, {rate}, {time_in_force}, {current_time}, {entry_tag}, {side}, {kwargs}")
        return confirm

    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                           rate: float, time_in_force: str, exit_reason: str,
                           current_time: datetime, **kwargs) -> bool:
        exit_reason = f"{exit_reason}_{trade.enter_tag}"
        if ('unclog' in exit_reason or 'force' in exit_reason
                or 'stoploss' == exit_reason or
                'stop-loss' == exit_reason or 'psl takeprofit' == exit_reason
                or 'psl stoploss' == exit_reason):
            return True

        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        last_candle = dataframe.iloc[-1]
        current_profit = trade.calc_profit_ratio(rate)

        if last_candle['ema_fast'] > last_candle['ema_slow']:
            return False
        t = True if current_profit > self.profit_treshold.value else False
        if t:
            loginfo(f"CTE - Profit > {self.profit_treshold.value}, EXIT")
        return t

    def custom_exit(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                    current_profit: float, **kwargs) -> bool:
        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        last_candle = dataframe.iloc[-1]
        current_profit = trade.calc_profit_ratio(current_rate)

        if last_candle['high'] > last_candle['open'] and (current_profit > self.profit_treshold.value):
            loginfo(f"CTE - Cena stále roste (high > open), Profit > {self.profit_treshold.value},  HOLD")
            return False
        return True

    def prepare_psl(self, dataframe: DataFrame) -> DataFrame:
        # Kontrola, jestli je dostatek dat
        if len(dataframe) < 10:  # Zvolte vhodný počet řádků dle potřeby
            logwarning("Nedostatečný počet řádků dat pro výpočet PSL.")
            # Nastavení stop_loss a take_profit v případě nedostatečných dat
            last_price = dataframe['close'].iloc[-1]
            dataframe['stop_loss'] = last_price * (1 - self.stoploss2.value)  # Příklad koeficientu pro stop_loss
            dataframe['take_profit'] = last_price * (1 + self.takeprofit2.value)  # Příklad koeficientu pro take_profit
            return dataframe

        atr = ta.ATR(dataframe, timeperiod=3)
        dataframe = self.detect_hammer_doji(dataframe)
        for i in range(1, len(dataframe)):
            row = dataframe.iloc[i]
            prev_row = dataframe.iloc[i - 1]
            if row['is_hammer'] or row['is_doji']:
                dataframe.at[i, 'stop_loss'] = prev_row['low'] - atr.iloc[i] * self.stoploss1.value
                dataframe.at[i, 'take_profit'] = prev_row['high'] + atr.iloc[i] * self.takeprofit1.value
            else:
                dataframe.at[i, 'stop_loss'] = prev_row['low'] - atr.iloc[i] * self.stoploss2.value
                dataframe.at[i, 'take_profit'] = prev_row['high'] + atr.iloc[i] * self.takeprofit2.value

        loginfo("Prepared PSL")
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Kontrola existence sloupce 'take_profit'
        if 'take_profit' in dataframe.columns:
            profit_cond = (
                    (dataframe['high'] > dataframe['take_profit']) |
                    (dataframe['close'] > dataframe['take_profit']) |
                    (dataframe['open'] > dataframe['take_profit'])
            )
            dataframe.loc[profit_cond, 'exit_tag'] = 'psl takeprofit'
            dataframe.loc[profit_cond, 'exit_long'] = 1
        else:
            logwarning("Sloupec 'take_profit' neexistuje v DataFrame.")

        # Kontrola existence sloupce 'stop_loss'
        if 'stop_loss' in dataframe.columns:
            stoploss_cond = (
                    (dataframe['low'] < dataframe['stop_loss']) |
                    (dataframe['close'] < dataframe['stop_loss']) |
                    (dataframe['open'] < dataframe['stop_loss'])
            )
            dataframe.loc[stoploss_cond, 'exit_tag'] = 'psl stoploss'
            dataframe.loc[stoploss_cond, 'exit_long'] = 1
        else:
            logwarning("Sloupec 'stop_loss' neexistuje v DataFrame.")

        return dataframe



