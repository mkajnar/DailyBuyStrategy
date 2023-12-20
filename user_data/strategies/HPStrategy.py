import datetime
import json
import logging
import math
import os
import traceback
from datetime import datetime
from datetime import timedelta, timezone
from functools import reduce
from typing import Optional
import numpy as np
import pandas as pd
import talib.abstract as ta
from pandas import DataFrame

import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.constants import Config
from freqtrade.persistence import Trade, Order, LocalTrade
from freqtrade.strategy import merge_informative_pair, DecimalParameter, IntParameter
from freqtrade.strategy.interface import IStrategy


def pct_change(a, b):
    return (b - a) / a


def load_sell_value_info(sell_value_info_file):
    logging.info("Loading sell value info")
    try:
        user_data_directory = os.path.join('user_data')
        with open(os.path.join(user_data_directory, sell_value_info_file), 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        return {}


def save_sell_value_info(sell_value_info_file, sell_value_info):
    logging.info("Saving sell value info")
    user_data_directory = os.path.join('user_data')
    with open(os.path.join(user_data_directory, sell_value_info_file), 'w') as file:
        json.dump(sell_value_info, file)


def EWO(dataframe, ema_length=5, ema2_length=3):
    df = dataframe.copy()
    ema1 = ta.EMA(df, timeperiod=ema_length)
    ema2 = ta.EMA(df, timeperiod=ema2_length)
    return (ema1 - ema2) / df['close'] * 100


class HPStrategy(IStrategy):
    INTERFACE_VERSION = 2

    max_safety_orders = 2
    lowest_prices = {}
    highest_prices = {}
    price_drop_percentage = {}
    pairs_close_to_high = []
    locked = []
    stoploss = -0.99

    out_open_trades_limit = 4
    is_optimize_cofi = False
    use_sell_signal = True
    sell_profit_only = True
    # sell_profit_offset = 0.015
    ignore_roi_if_buy_signal = False
    position_adjustment_enable = True
    order_time_in_force = {
        'buy': 'gtc',
        'sell': 'gtc'
    }

    timeframe = '1m'
    inf_1h = '1h'
    process_only_new_candles = True
    startup_candle_count = 400
    plot_config = {
        'main_plot': {
            'ma_buy': {'color': 'orange'},
            'ma_sell': {'color': 'orange'},
        },
    }

    buy_params = {
        "base_nb_candles_buy": 12,
        "rsi_buy": 58,
        "ewo_high": 3.001,
        "ewo_low": -10.289,
        "low_offset": 0.987,
        "lambo2_ema_14_factor": 0.981,
        "lambo2_enabled": True,
        "lambo2_rsi_14_limit": 39,
        "lambo2_rsi_4_limit": 44,
        "buy_adx": 20,
        "buy_fastd": 20,
        "buy_fastk": 22,
        "buy_ema_cofi": 0.98,
        "buy_ewo_high": 4.179
    }

    buy_ema_cofi = DecimalParameter(0.96, 0.98, default=0.97, optimize=is_optimize_cofi)
    buy_fastk = IntParameter(20, 30, default=20, optimize=is_optimize_cofi)
    buy_fastd = IntParameter(20, 30, default=20, optimize=is_optimize_cofi)
    buy_adx = IntParameter(20, 30, default=30, optimize=is_optimize_cofi)
    buy_ewo_high = DecimalParameter(2, 12, default=3.553, optimize=is_optimize_cofi)
    base_nb_candles_buy = IntParameter(8, 20, default=buy_params['base_nb_candles_buy'], space='buy', optimize=False)
    low_offset = DecimalParameter(0.975, 0.995, default=buy_params['low_offset'], space='buy', optimize=True)
    rsi_buy = IntParameter(30, 70, default=buy_params['rsi_buy'], space='buy', optimize=False)
    lambo2_ema_14_factor = DecimalParameter(0.8, 1.2, decimals=3, default=buy_params['lambo2_ema_14_factor'],
                                            space='buy', optimize=True)
    lambo2_rsi_4_limit = IntParameter(10, 60, default=buy_params['lambo2_rsi_4_limit'], space='buy', optimize=True)
    lambo2_rsi_14_limit = IntParameter(10, 60, default=buy_params['lambo2_rsi_14_limit'], space='buy', optimize=True)

    fast_ewo = 50
    slow_ewo = 200
    ewo_low = DecimalParameter(-20.0, -7.0, default=buy_params['ewo_low'], space='buy', optimize=True)
    ewo_high = DecimalParameter(3.0, 5, default=buy_params['ewo_high'], space='buy', optimize=True)

    trailing_stop = True
    trailing_stop_positive = 0.002
    trailing_stop_positive_offset = 0.008
    trailing_only_offset_is_reached = True

    minimal_roi = {
        "0": 0.15,
        "30": 0.10,
        "60": 0.05,
        "90": 0.03,
        "120": 0.01,
        "240": 0
    }

    order_types = {
        'entry': 'market',
        'exit': 'market',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    sell_params = {
        "base_nb_candles_sell": 22,
        "high_offset": 1.014,
        "high_offset_2": 1.01
    }

    base_nb_candles_sell = IntParameter(8, 20, default=sell_params['base_nb_candles_sell'], space='sell',
                                        optimize=False)
    high_offset = DecimalParameter(1.000, 1.010, default=sell_params['high_offset'], space='sell', optimize=True)
    high_offset_2 = DecimalParameter(1.000, 1.010, default=sell_params['high_offset_2'], space='sell', optimize=True)

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
        return "HPStrategy v1.1.0 "

    def custom_sell(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                    current_profit: float, **kwargs):
        if current_profit < -0.05 and (current_time - trade.open_date_utc).days >= 7:
            return 'unclog'

    def order_price(self, free_amount, positions, dca_buys):
        total_dca_budget = free_amount - (positions + 1) * dca_buys
        return total_dca_budget / (positions * dca_buys)

    # def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
    #                         proposed_stake: float, min_stake: Optional[float], max_stake: float,
    #                         leverage: float, entry_tag: Optional[str], side: str, **kwargs) -> float:
    #     min_trade_size = 3
    #     if proposed_stake < min_trade_size:
    #         return 0
    #     adjusted_price = self.order_price(free_amount=self.wallets.get_available_stake_amount(),
    #                                       positions=3,
    #                                       dca_buys=self.max_safety_orders)
    #     if adjusted_price < min_trade_size:
    #         return 0
    #     logging.info(f"Adjusting entry price to {adjusted_price}")
    #     return adjusted_price

    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, '1h') for pair in pairs]

        if self.config['stake_currency'] in ['USDT', 'BUSD', 'USDC', 'DAI', 'TUSD', 'PAX', 'USD', 'EUR', 'GBP']:
            btc_info_pair = f"BTC/{self.config['stake_currency']}"
        else:
            btc_info_pair = "BTC/USDT"

        informative_pairs.extend(
            ((btc_info_pair, self.timeframe), (btc_info_pair, self.inf_1h))
        )
        return informative_pairs

    def analyze_price_movements(self, dataframe, metadata, window=50):
        pair = metadata['pair']
        low = dataframe['low'].rolling(window=window).min()
        high = dataframe['high'].rolling(window=window).max()
        current_price = dataframe['close'].iloc[-1]
        mid_price = (low + high) / 2
        price_to_mid_ratio = ((current_price - mid_price) / (high - mid_price)).iloc[-1]

        self.pairs_close_to_high = list(set(self.pairs_close_to_high))

        if price_to_mid_ratio > 0.5:
            if pair not in self.pairs_close_to_high:
                self.pairs_close_to_high.append(pair)
                if pair in self.locked:
                    self.locked.remove(pair)
        elif pair in self.pairs_close_to_high:
            self.pairs_close_to_high.remove(pair)
            if pair not in self.locked:
                logging.info(f"Locking {pair}")
                self.lock_pair(pair, until=datetime.now(timezone.utc) + timedelta(minutes=5))
                self.locked.append(pair)

        user_data_directory = os.path.join('user_data')
        if not os.path.exists(user_data_directory):
            os.makedirs(user_data_directory)
        with open(os.path.join(user_data_directory, 'high_moving_pairs.json'), 'w') as f:
            json.dump(self.pairs_close_to_high, f, indent=4)

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

    def save_dictionaries_to_disk(self):
        try:
            user_data_directory = os.path.join('user_data')
            if not os.path.exists(user_data_directory):
                os.makedirs(user_data_directory)
            with open(os.path.join(user_data_directory, 'lowest_prices.json'), 'w') as file:
                json.dump(self.lowest_prices, file, indent=4)
            with open(os.path.join(user_data_directory, 'highest_prices.json'), 'w') as file:
                json.dump(self.highest_prices, file, indent=4)
            with open(os.path.join(user_data_directory, 'price_drop_percentage.json'), 'w') as file:
                json.dump(self.price_drop_percentage, file, indent=4)
        except Exception as ex:
            logging.error(str(ex))

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # logging.info("Populating indicators")
        dataframe['price_history'] = dataframe['close'].shift(1)
        data_last_bbars = dataframe[-30:].copy()
        low_min = dataframe['low'].rolling(window=14).min()
        high_max = dataframe['high'].rolling(window=14).max()
        dataframe['stoch_k'] = 100 * (dataframe['close'] - low_min) / (high_max - low_min)
        dataframe['stoch_d'] = dataframe['stoch_k'].rolling(window=3).mean()

        pair = metadata['pair']
        if self.config['stake_currency'] in ['USDT', 'BUSD']:
            btc_info_pair = f"BTC/{self.config['stake_currency']}"
        else:
            btc_info_pair = "BTC/USDT"

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

        for val in self.base_nb_candles_buy.range:
            dataframe[f'ma_buy_{val}'] = ta.EMA(dataframe, timeperiod=val)
        for val in self.base_nb_candles_sell.range:
            dataframe[f'ma_sell_{val}'] = ta.EMA(dataframe, timeperiod=val)

        dataframe['hma_50'] = qtpylib.hull_moving_average(dataframe['close'], window=50)
        dataframe['sma_9'] = ta.SMA(dataframe, timeperiod=9)

        # Elliot
        dataframe['EWO'] = EWO(dataframe, self.fast_ewo, self.slow_ewo)

        # RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['rsi_fast'] = ta.RSI(dataframe, timeperiod=4)
        dataframe['rsi_slow'] = ta.RSI(dataframe, timeperiod=20)

        # lambo2
        dataframe['ema_14'] = ta.EMA(dataframe, timeperiod=14)
        dataframe['rsi_4'] = ta.RSI(dataframe, timeperiod=4)
        dataframe['rsi_14'] = ta.RSI(dataframe, timeperiod=14)

        # # Pump strength
        # dataframe['zema_30'] = ftt.zema(dataframe, period=30)
        # dataframe['zema_200'] = ftt.zema(dataframe, period=200)
        # dataframe['pump_strength'] = (dataframe['zema_30'] - dataframe['zema_200']) / dataframe['zema_30']

        # Cofi
        stoch_fast = ta.STOCHF(dataframe, 5, 3, 0, 3, 0)
        dataframe['fastd'] = stoch_fast['fastd']
        dataframe['fastk'] = stoch_fast['fastk']
        dataframe['adx'] = ta.ADX(dataframe)
        dataframe['ema_8'] = ta.EMA(dataframe, timeperiod=8)

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

        dataframe['ema_diff_buy_signal'] = ema_buy_signal | crossover_up | close_to_crossover_up

        dataframe['ema_diff_sell_signal'] = ((dataframe['ema_pct_diff'] > 0) &
                                             (dataframe['prev_ema_pct_diff'] > 0) &
                                             (dataframe['ema_pct_diff'].abs() < dataframe['prev_ema_pct_diff'].abs()))

        dataframe = self.pump_dump_protection(dataframe, metadata)

        # Bullish Divergence
        # Ujistěte se, že výpočet používá pouze historická data
        low_min = dataframe['low'].rolling(window=14).min()
        rsi_min = dataframe['rsi'].rolling(window=14).min()
        bullish_div = (low_min.shift(1) > low_min) & (rsi_min.shift(1) < rsi_min)
        dataframe['bullish_divergence'] = bullish_div.astype(int)

        # Fractals
        # Upravte tak, aby se nezahrnovala budoucí data
        dataframe['fractal_top'] = (dataframe['high'] > dataframe['high'].shift(2)) & \
                                   (dataframe['high'] > dataframe['high'].shift(1)) & \
                                   (dataframe['high'] > dataframe['high']) & \
                                   (dataframe['high'] > dataframe['high'].shift(-1))
        dataframe['fractal_bottom'] = (dataframe['low'] < dataframe['low'].shift(2)) & \
                                      (dataframe['low'] < dataframe['low'].shift(1)) & \
                                      (dataframe['low'] < dataframe['low']) & \
                                      (dataframe['low'] < dataframe['low'].shift(-1))

        dataframe['turnaround_signal'] = (bullish_div) & (dataframe['fractal_bottom'])
        dataframe['rolling_max'] = dataframe['high'].cummax()
        dataframe['drawdown'] = (dataframe['rolling_max'] - dataframe['low']) / dataframe['rolling_max']
        dataframe['below_90_percent_drawdown'] = dataframe['drawdown'] >= 0.90

        # MACD a Volatility Factor
        # MACD výpočet zůstává nezměněn
        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']

        # Výpočet volatility (použití rolling standard deviation)
        dataframe['volatility'] = dataframe['close'].rolling(window=14).std()

        # Výpočet volatility pomocí ATR nebo standardní odchylky
        dataframe['volatility'] = dataframe['close'].rolling(window=14).std()

        # Normalizace volatility
        min_volatility = dataframe['volatility'].rolling(window=14).min()
        max_volatility = dataframe['volatility'].rolling(window=14).max()
        dataframe['volatility_factor'] = (dataframe['volatility'] - min_volatility) / \
                                         (max_volatility - min_volatility)

        # Přizpůsobení MACD na základě volatility
        dataframe['macd_adjusted'] = dataframe['macd'] * (1 - dataframe['volatility_factor'])
        dataframe['macdsignal_adjusted'] = dataframe['macdsignal'] * (1 + dataframe['volatility_factor'])

        dataframe = self.percentage_drop_indicator(dataframe, 9, threshold=0.21)
        # dataframe.drop_duplicates()
        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        self.analyze_price_movements(dataframe=dataframe, metadata=metadata, window=200)
        better_pair = metadata['pair'] not in self.pairs_close_to_high

        conditions = []
        dataframe.loc[:, 'buy_tag'] = ''
        dataframe.loc[:, 'sell_tag'] = ''

        lambo2 = (
            # bool(self.lambo2_enabled.value) &
            # (dataframe['pump_warning'] == 0) &
                (dataframe['close'] < (dataframe['ema_14'] * self.lambo2_ema_14_factor.value)) &
                (dataframe['rsi_4'] < int(self.lambo2_rsi_4_limit.value)) &
                (dataframe['rsi_14'] < int(self.lambo2_rsi_14_limit.value))
        )
        dataframe.loc[lambo2, 'buy_tag'] += 'lambo2_'
        conditions.append(lambo2)

        buy1ewo = (
                (dataframe['rsi_fast'] < 35) &
                (dataframe['close'] < (dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'] * self.low_offset.value)) &
                (dataframe['EWO'] > self.ewo_high.value) &
                (dataframe['rsi'] < self.rsi_buy.value) &
                (dataframe['volume'] > 0) &
                (dataframe['close'] < (
                        dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset.value))
        )
        dataframe.loc[buy1ewo, 'buy_tag'] += 'buy1eworsi_'
        conditions.append(buy1ewo)

        buy2ewo = (
                (dataframe['rsi_fast'] < 35) &
                (dataframe['close'] < (dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'] * self.low_offset.value)) &
                (dataframe['EWO'] < self.ewo_low.value) &
                (dataframe['volume'] > 0) &
                (dataframe['close'] < (
                        dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset.value))
        )
        dataframe.loc[buy2ewo, 'buy_tag'] += 'buy2ewo_'
        conditions.append(buy2ewo)

        is_cofi = (
                (dataframe['open'] < dataframe['ema_8'] * self.buy_ema_cofi.value) &
                (qtpylib.crossed_above(dataframe['fastk'], dataframe['fastd'])) &
                (dataframe['fastk'] < self.buy_fastk.value) &
                (dataframe['fastd'] < self.buy_fastd.value) &
                (dataframe['adx'] > self.buy_adx.value) &
                (dataframe['EWO'] > self.buy_ewo_high.value)
        )
        dataframe.loc[is_cofi, 'buy_tag'] += 'cofi_'
        conditions.append(is_cofi)

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions) & better_pair,
                'buy'
            ] = 1

        dont_buy_conditions = [
            dataframe['pnd_volume_warn'] < 0.0,
            dataframe['btc_rsi_8_1h'] < 35.0
        ]

        if conditions:
            final_condition = reduce(lambda x, y: x | y, conditions)
            dataframe.loc[final_condition, 'buy'] = 1
        if dont_buy_conditions:
            for condition in dont_buy_conditions:
                dataframe.loc[condition, 'buy'] = 0
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if conditions := [
            (dataframe['close'] > dataframe['hma_50'])
            & (
                    dataframe['close']
                    > (
                            dataframe[f'ma_sell_{self.base_nb_candles_sell.value}']
                            * self.high_offset_2.value
                    )
            )
            & (dataframe['rsi'] > 50)
            & (dataframe['volume'] > 0)
            & (dataframe['rsi_fast'] > dataframe['rsi_slow'])
            | (dataframe['close'] < dataframe['hma_50'])
            & (
                    dataframe['close']
                    > (
                            dataframe[f'ma_sell_{self.base_nb_candles_sell.value}']
                            * self.high_offset.value
                    )
            )
            & (dataframe['volume'] > 0)
            & (dataframe['rsi_fast'] > dataframe['rsi_slow'])
        ]:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions),
                'sell'
            ] = 1
        return dataframe

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                            time_in_force: str, current_time: datetime, entry_tag: Optional[str],
                            side: str, **kwargs) -> bool:
        """
        Confirms the entry of a trade based on the number of open trades.
        Args:
            pair: The trading pair.
            order_type: The type of order.
            amount: The amount of the trade.
            rate: The rate of the trade.
            time_in_force: The time in force of the order.
            current_time: The current time.
            entry_tag: The entry tag of the trade.
            side: The side of the trade.
            **kwargs: Additional keyword arguments.
        Returns:
            bool: True if the entry is confirmed, False otherwise.
        """

        # logging.info(f"Count opened trades is {Trade.get_open_trade_count()}")
        result = Trade.get_open_trade_count() < self.out_open_trades_limit
        # if result:
        # logging.info(f"Count of opened trades is lower than {self.out_open_trades_limit}, OK")
        # logging.info(f"Pair {pair} - entry confirm: Amount {amount}, Rate {rate}, Side {side}, Entry tag {entry_tag}")
        # else:
        # logging.info(f"Count of opened trades is higher than {self.out_open_trades_limit}, BAD")
        return result

    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                           rate: float, time_in_force: str, sell_reason: str,
                           current_time: datetime, **kwargs) -> bool:
        """
        Confirms the exit of a trade based on the sell reason and current profit.
        Args:
            pair: The trading pair.
            trade: The trade object.
            order_type: The type of order.
            amount: The amount of the trade.
            rate: The rate of the trade.
            time_in_force: The time in force of the order.
            sell_reason: The reason for selling.
            current_time: The current time.
            **kwargs: Additional keyword arguments.
        Returns:
            bool: True if the exit is confirmed, False otherwise.
        """
        sell_reason = f"{sell_reason}_" + trade.buy_tag
        current_profit = trade.calc_profit_ratio(rate)
        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        return (
                current_profit >= 0.01  # nechceme jít do ztráty
                or 'unclog' in sell_reason
                or 'force' in sell_reason
        )

    def percentage_drop_indicator(self, dataframe, period, threshold=0.3):
        # Vypočet nejvyšší ceny za poslední období
        highest_high = dataframe['high'].rolling(period).max()
        # Vypočet procentuálního poklesu pod nejvyšší cenou
        percentage_drop = (highest_high - dataframe['close']) / highest_high * 100
        dataframe.loc[percentage_drop < threshold, 'percentage_drop_buy'] = 1
        dataframe.loc[percentage_drop > threshold, 'percentage_drop_buy'] = 0
        return dataframe


class HPStrategyDCA(HPStrategy):
    initial_safety_order_trigger = -0.018
    safety_order_step_scale = 1.2
    safety_order_volume_scale = 1.4
    drawdown_limit = -3
    average_dropdown = {}
    buy_params = {
        "dca_min_rsi": 35,
    }

    timeframes_in_minutes = {
        '1m': 1,
        '5m': 5,
        '15m': 15,
        '30m': 30,
        '1h': 60,
        '4h': 240,
        '1d': 1440
    }

    buy_params.update(HPStrategy.buy_params)
    dca_min_rsi = IntParameter(35, 75, default=buy_params['dca_min_rsi'], space='buy', optimize=True)

    def version(self) -> str:
        return f"{super().version()} DCA "

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = super().populate_indicators(dataframe, metadata)
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        return dataframe

    def calculate_volatility(self, dataframe: DataFrame, pair: str, timeframe: str) -> float:
        # logging.info("Calculating volatility")
        timeframes_in_minutes = {
            '1m': 1,
            '5m': 5,
            '15m': 15,
            '30m': 30,
            '1h': 60,
            '4h': 240,
            '1d': 1440
        }

        interval_in_minutes = timeframes_in_minutes.get(timeframe)

        if interval_in_minutes is None:
            raise ValueError("Neplatný timeframe. Prosím, zadejte jeden z podporovaných timeframe.")
        periods = int(24 * 60 / interval_in_minutes)
        dataframe['pct_change'] = dataframe['close'].pct_change()
        return dataframe['pct_change'].tail(periods).abs().mean() * 100

    def dynamic_stake_adjustment(self, stake, volatility):
        return stake * 0.8 if volatility > 0.05 else stake

    def calculate_drawdown(self, current_price, last_order_price):
        return (current_price - last_order_price) / last_order_price * 100

    def calculate_dca_amount(self, current_price, target_profit, average_buy_price, total_investment):
        target_sell_price = average_buy_price * (1 + target_profit)
        required_price_rise = target_sell_price / current_price
        return total_investment * (required_price_rise - 1)

    def check_buy_conditions(self, lambo2_ema_14_factor, lambo2_rsi_4_limit, lambo2_rsi_14_limit, base_nb_candles_buy,
                             low_offset,
                             ewo_high, rsi_buy, base_nb_candles_sell, high_offset, ewo_low, buy_ema_cofi, buy_fastk,
                             buy_fastd, buy_adx, buy_ewo_high, last_candle, previous_candle):
        lambo2 = (
                (last_candle['close'] < (last_candle['ema_14'] * lambo2_ema_14_factor.value)) &
                (last_candle['rsi_4'] < int(lambo2_rsi_4_limit.value)) &
                (last_candle['rsi_14'] < int(lambo2_rsi_14_limit.value))
        )
        conditions = [lambo2]
        buy1ewo = (
                (last_candle['rsi_fast'] < 35) &
                (last_candle['close'] < (
                        last_candle[f'ma_buy_{base_nb_candles_buy.value}'] * low_offset.value)) &
                (last_candle['EWO'] > ewo_high.value) &
                (last_candle['rsi'] < rsi_buy.value) &
                (last_candle['volume'] > 0) &
                (last_candle['close'] < (
                        last_candle[f'ma_sell_{base_nb_candles_sell.value}'] * high_offset.value))
        )
        conditions.append(buy1ewo)
        buy2ewo = (
                (last_candle['rsi_fast'] < 35) &
                (last_candle['close'] < (
                        last_candle[f'ma_buy_{base_nb_candles_buy.value}'] * low_offset.value)) &
                (last_candle['EWO'] < ewo_low.value) &
                (last_candle['volume'] > 0) &
                (last_candle['close'] < (
                        last_candle[f'ma_sell_{base_nb_candles_sell.value}'] * high_offset.value))
        )
        conditions.append(buy2ewo)
        crossed_above_fastk_fastd = (previous_candle['fastk'] < previous_candle['fastd']) and (
                last_candle['fastk'] > last_candle['fastd'])
        is_cofi = (
                (last_candle['open'] < last_candle['ema_8'] * buy_ema_cofi.value) &
                crossed_above_fastk_fastd &
                (last_candle['fastk'] < buy_fastk.value) &
                (last_candle['fastd'] < buy_fastd.value) &
                (last_candle['adx'] > buy_adx.value) &
                (last_candle['EWO'] > buy_ewo_high.value)
        )
        conditions.append(is_cofi)
        return any(conditions)

    # def calculate_median_drop(self, dataframe, num_candles, pair):
    #     if len(dataframe) < num_candles:
    #         return None
    #     dataframe['max_price'] = dataframe['high'].rolling(window=num_candles).max()
    #     dataframe['percent_drop'] = (dataframe['max_price'] - dataframe['close']) / dataframe['max_price'] * 100
    #     median_drop = dataframe['percent_drop'].rolling(window=num_candles).median().iloc[-1]
    #     return int(round(median_drop))

    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                              current_rate: float, current_profit: float, min_stake: float,
                              max_stake: float, **kwargs):

        try:
            dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
            df = dataframe.copy()
        except Exception as e:
            logging.error(f"Error getting analyzed dataframe: {e}")
            return None

        # average = self.calculate_median_drop(dataframe=df, num_candles=20, pair=trade.pair)
        volatility = self.calculate_volatility(df, trade.pair, self.timeframe)
        adjusted_min_stake = self.dynamic_stake_adjustment(min_stake, volatility)
        adjusted_max_stake = self.dynamic_stake_adjustment(max_stake, volatility)

        last_candle = df.iloc[-1].squeeze()
        previous_candle = df.iloc[-2].squeeze()
        if last_candle['close'] < previous_candle['close']:
            return None

        # Přidáme kontrolu na základě percentage_drop
        highest_high = df['high'].rolling(9).max()
        percentage_drop = (highest_high - df['close']) / highest_high * 100

        dt = percentage_drop.tail(30)
        if dt.is_monotonic_increasing:
            logging.info(
                f"Percentage drop se pro {trade.pair} stále zvětšuje, DCA se neprovádí.")
            return None

        current_candle_index = df.index[-1]

        if last_buy_order := next(
                (
                        order
                        for order in sorted(
                    trade.orders, key=lambda x: x.order_date, reverse=True
                )
                        if order.ft_order_side == 'buy' and order.status == 'closed'
                ),
                None,
        ):
            last_buy_candle = dataframe.loc[dataframe['date'] == last_buy_order.order_date]
            if not last_buy_candle.empty:
                last_buy_candle_index = last_buy_candle.index[0]
                if current_candle_index == last_buy_candle_index:
                    return None

        if not self.check_buy_conditions(self.lambo2_ema_14_factor, self.lambo2_rsi_4_limit, self.lambo2_rsi_14_limit,
                                         self.base_nb_candles_buy, self.low_offset, self.ewo_high, self.rsi_buy,
                                         self.base_nb_candles_sell, self.high_offset, self.ewo_low, self.buy_ema_cofi,
                                         self.buy_fastk, self.buy_fastd, self.buy_adx, self.buy_ewo_high, last_candle,
                                         previous_candle):
            return None

        count_of_buys = sum(order.ft_order_side == 'buy' and order.status == 'closed' for order in trade.orders)

        if self.max_safety_orders >= count_of_buys >= 1:
            last_order_price = trade.open_rate
            if last_buy_order := next(
                    (
                            order
                            for order in sorted(
                        trade.orders, key=lambda x: x.order_date, reverse=True
                    )
                            if order.ft_order_side == 'buy'
                    ),
                    None,
            ):
                last_order_price = last_buy_order.price or last_buy_order.average

            drawdown = self.calculate_drawdown(current_rate, last_order_price) if last_order_price else 0

            if drawdown <= -3:
                try:
                    stake_amount = self.wallets.get_trade_stake_amount(trade.pair, None)
                    stake_amount = min(stake_amount * math.pow(self.safety_order_volume_scale, (count_of_buys - 1)),
                                       adjusted_max_stake)
                    if stake_amount < adjusted_min_stake:
                        return None

                    try:
                        price_change_rate = (last_candle['close'] - previous_candle['close']) / previous_candle['close']
                        if price_change_rate < -0.02:
                            adjusted_stake = stake_amount * 1.5
                        elif price_change_rate > 0.02:
                            adjusted_stake = stake_amount * 0.75
                        else:
                            adjusted_stake = stake_amount
                    except:
                        adjusted_stake = stake_amount

                    return adjusted_stake

                except Exception as exception:
                    logging.error(f"Error adjusting trade position: {exception}")
                    return None

        return None


class HPStrategyTF(HPStrategyDCA):
    def version(self) -> str:
        return f"{super().version()} TF "

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = super().populate_indicators(dataframe, metadata)

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
        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = super().populate_buy_trend(dataframe, metadata)
        down_trend = (
            (dataframe['higher_tf_trend'] > 1)
        )
        dataframe.loc[down_trend, 'buy'] = 1
        return dataframe


class HPStrategyTFJPA(HPStrategyTF):
    trailing_stop = True
    trailing_only_offset_is_reached = True
    trailing_stop_positive = 0.007
    trailing_stop_positive_offset = 0.015
    ignore_roi_if_buy_signal = True

    def version(self) -> str:
        return f"{super().version()} JPA "

    def calculate_dca_price(self, base_value, decline, target_percent):
        result = (((base_value / 100) * abs(decline)) / target_percent) * 100
        return result

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[:, 'buy_tag'] = ''
        dataframe.loc[:, 'buy'] = 0
        dataframe.loc[(dataframe['volume'] > 0) & (
                dataframe['ema_diff_buy_signal'].astype(int) > 0), 'buy_tag'] += 'ema_diff_buy_signal'
        dataframe.loc[(dataframe['volume'] > 0) & (dataframe['ema_diff_buy_signal'].astype(int) > 0), 'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[:, 'sell_tag'] = ''
        dataframe.loc[:, 'sell'] = 0
        dataframe.loc[(dataframe['volume'] > 0) & (
                dataframe['ema_diff_sell_signal'].astype(int) > 0), 'sell_tag'] += 'ema_diff_sell_signal'
        dataframe.loc[(dataframe['volume'] > 0) & (dataframe['ema_diff_sell_signal'].astype(int) > 0), 'sell'] = 1
        return dataframe

    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                              current_rate: float, current_profit: float, min_stake: float,
                              max_stake: float, **kwargs):

        # logging.info('AP1')

        try:
            dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
            df = dataframe.copy()
        except Exception as e:
            logging.error(f"Error getting analyzed dataframe: {e}")
            return None

        try:
            dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
            df = dataframe.copy()
        except Exception as e:
            logging.error(f"Error getting analyzed dataframe: {e}")
            return None

        # average = self.calculate_median_drop(dataframe=df, num_candles=20, pair=trade.pair)

        # Přidáme kontrolu na základě percentage_drop
        highest_high = df['high'].rolling(9).max()
        percentage_drop = (highest_high - df['close']) / highest_high * 100
        dt = percentage_drop.tail(30)
        if dt.is_monotonic_increasing:
            logging.info(
                f"Percentage drop se pro {trade.pair} stále zvětšuje, DCA se neprovádí.")
            return None

        last_candle = df.iloc[-1].squeeze()
        previous_candle = df.iloc[-2].squeeze()
        if last_candle['close'] <= previous_candle['close']:
            return None

        # logging.info('AP2')

        count_of_buys = sum(order.ft_order_side == 'buy' and order.status == 'closed' for order in trade.orders)
        if self.max_safety_orders > count_of_buys:
            # logging.info('AP3')

            # last_order_price = trade.open_rate
            # if last_buy_order := next(
            #         (
            #                 order
            #                 for order in sorted(
            #             trade.orders, key=lambda x: x.order_date, reverse=True
            #         )
            #                 if order.ft_order_side == 'buy'
            #         ),
            #         None,
            # ):
            #     last_order_price = last_buy_order.price or last_buy_order.average
            #
            # drawdown = self.calculate_drawdown(current_rate, last_order_price) if last_order_price else 0
            #
            # if drawdown >= -3:
            #     return None
            #
            pct = current_profit * 100
            pct_treshold = 2
            if pct <= -pct_treshold and last_candle['ema_diff_buy_signal'] == 1:

                logging.info(f'AP4 {trade.pair}, Profit: {current_profit}')
                stake_amount = self.wallets.get_trade_stake_amount(trade.pair, None)
                total_stake_amount = self.wallets.get_total_stake_amount()
                calculated_dca_stake = self.calculate_dca_price(base_value=stake_amount,
                                                                decline=current_profit * 100,
                                                                target_percent=1)
                if calculated_dca_stake > total_stake_amount:
                    logging.info(f'AP5 {trade.pair}, DCA: {calculated_dca_stake} is more than {stake_amount}')
                    return None
                logging.info(f'AP6 {trade.pair}, DCA: {calculated_dca_stake}')
                return calculated_dca_stake
        return None
