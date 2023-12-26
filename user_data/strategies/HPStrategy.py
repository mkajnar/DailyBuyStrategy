import datetime
import json
import logging
import math
import os
from datetime import datetime
from datetime import timedelta, timezone
from functools import reduce
from typing import Optional, List

import numpy as np
import talib.abstract as ta
from pandas import DataFrame
from technical.indicators import ichimoku

import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.persistence import Trade, LocalTrade, Order
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

    max_safety_orders = 3
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

        ichi = ichimoku(dataframe)
        dataframe['senkou_span_a'] = ichi['senkou_span_a']
        dataframe['senkou_span_b'] = ichi['senkou_span_b']

        # Vytvoření vah pro vážený průměr
        weights = np.linspace(1, 0, 300)  # Váhy od 1 (nejnovější) do 0 (nejstarší)
        weights /= weights.sum()  # Normalizace vah tak, aby jejich součet byl 1

        # Výpočet váženého průměru RSI pro posledních 300 svící
        dataframe['weighted_rsi'] = dataframe['rsi'].rolling(window=300).apply(
            lambda x: np.sum(weights * x[-300:]), raw=False
        )

        # Přidání signálu 'jstkr'
        # Vytváří 1, když je součet 'macd' a 'macd_signal' záporný a 'rsi' <= 30
        dataframe['jstkr'] = ((dataframe['macd'] + dataframe['macdsignal'] < -0.01) & (dataframe['rsi'] <= 17)).astype(int)
        dataframe['jstkr_2'] = ((abs(dataframe['macd'] - dataframe['macdsignal']) / dataframe['macd'].abs() > 0.2) &(dataframe['rsi'] < 25)).astype('int')

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
        # logging.info(f"Count opened trades is {Trade.get_open_trade_count()}")

        try:
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            df = dataframe.copy()
        except Exception as e:
            logging.error(f"Error getting analyzed dataframe: {e}")
            return None

        last_candle = df.iloc[-1].squeeze()

        result = Trade.get_open_trade_count() < self.out_open_trades_limit
        return result

    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                           rate: float, time_in_force: str, sell_reason: str,
                           current_time: datetime, **kwargs) -> bool:
        sell_reason = f"{sell_reason}_" + trade.buy_tag
        current_profit = trade.calc_profit_ratio(rate)
        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)

        # Aktuální hodnoty EMA
        ema_8_current = dataframe['ema_8'].iat[-1]
        ema_14_current = dataframe['ema_14'].iat[-1]

        # Hodnoty EMA předchozí svíčky
        ema_8_previous = dataframe['ema_8'].iat[-2]
        ema_14_previous = dataframe['ema_14'].iat[-2]

        # Výpočet rozdílu EMA mezi aktuální a předchozí svíčkou
        diff_current = abs(ema_8_current - ema_14_current)
        diff_previous = abs(ema_8_previous - ema_14_previous)

        # Výpočet procentní změny mezi diff_current a diff_previous
        diff_change_pct = (diff_previous - diff_current) / diff_previous

        if 'unclog' in sell_reason or 'force' in sell_reason:
            logging.info(f"CTE - FORCE or UNCLOG, EXIT")
            return True
        elif current_profit >= 0.0025:
            if ema_8_current <= ema_14_current and diff_change_pct >= 0.025:
                logging.info(
                    f"CTE - EMA 8 {ema_8_current} <= EMA 14 {ema_14_current} with decrease in difference >= 3%, EXIT")
                return True
            elif ema_8_current > ema_14_current and diff_current > diff_previous:
                logging.info(f"CTE - EMA 8 {ema_8_current} > EMA 14 {ema_14_current} with increasing difference, HOLD")
                return False
            else:
                logging.info(f"CTE - Conditions not met, EXIT")
                return True
        else:
            return False

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
    support_dict = {}
    resistance_dict = {}

    def version(self) -> str:
        return f"{super().version()} JPA "

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
            logging.error(str(ex))

    def calculate_dynamic_clusters(self, values, max_clusters):
        """
        Dynamicky vypočítá průměrované shluky z daného seznamu hodnot.

        Args:
        values (list): Seznam hodnot pro shlukování.
        max_clusters (int): Maximální počet shluků, který se má vytvořit.

        Returns:
        list: Seznam průměrných hodnot pro každý vytvořený shluk.
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

        threshold = 0.3  # Počáteční prahová hodnota
        while True:
            clusters = cluster_values(threshold)
            if len(clusters) <= max_clusters:
                break
            threshold += 0.3

        # Výpočet průměrů pro každý shluk
        cluster_averages = [round(sum(cluster) / len(cluster), 2) for cluster in clusters]
        return cluster_averages

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = super().populate_indicators(dataframe, metadata)
        self.calculate_support_resistance_dicts(metadata['pair'], dataframe)
        # logging.info(f"Support dict: {json.dumps(self.support_dict)}")
        return dataframe

    def calculate_dca_price(self, base_value, decline, target_percent):
        return (((base_value / 100) * abs(decline)) / target_percent) * 100

    def populate_buy_trend_sr(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Kontrola vzdálenosti k supportu a resistenci
        if metadata['pair'] in self.support_dict and metadata['pair'] in self.resistance_dict:
            supports = self.support_dict[metadata['pair']]
            resistances = self.resistance_dict[metadata['pair']]

            if supports and resistances:
                # Vypočítání nejbližší úrovně supportu a resistence pro každou svíčku
                dataframe['nearest_support'] = dataframe['close'].apply(
                    lambda x: min([support for support in supports if support <= x], default=x,
                                  key=lambda support: abs(x - support))
                )
                dataframe['nearest_resistance'] = dataframe['close'].apply(
                    lambda x: min([resistance for resistance in resistances if resistance >= x], default=x,
                                  key=lambda resistance: abs(x - resistance))
                )

                # Vypočítání procentního rozdílu mezi cenou a nejbližším supportem/resistencí
                dataframe['distance_to_support_pct'] = (
                                                               dataframe['nearest_support'] - dataframe['close']) / \
                                                       dataframe['close'] * 100
                dataframe['distance_to_resistance_pct'] = (
                                                                  dataframe['nearest_resistance'] - dataframe[
                                                              'close']) / dataframe['close'] * 100

                # Vygenerování nákupních signálů na základě supportu a resistence
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
                    'buy_tag'
                ] += 'sr_buy_mid'

                # Odebrání pomocných sloupců
                dataframe.drop(
                    ['nearest_support', 'nearest_resistance', 'distance_to_support_pct', 'distance_to_resistance_pct'],
                    axis=1, inplace=True)

        # Přidání podmínek pro EMA a objem
        dataframe.loc[(dataframe['volume'] > 0) & (dataframe['ema_diff_buy_signal'].astype(int) > 0), 'buy_ema'] = 1
        dataframe.loc[
            (dataframe['volume'] > 0) & (dataframe['ema_diff_buy_signal'].astype(int) > 0), 'buy_tag'] += 'ema_dbs_'

        # Generování nákupních signálů pouze pokud jsou splněny obě podmínky
        dataframe.loc[(dataframe['buy_signal'] == 1) & (dataframe['buy_ema'] == 1) & (
                dataframe['rsi'] <= dataframe['weighted_rsi']), 'buy'] = 1

        # Odebrání pomocných sloupců
        if 'buy_support' in dataframe.columns:
            dataframe.drop(['buy_support'], axis=1, inplace=True)
        if 'buy_ema' in dataframe.columns:
            dataframe.drop(['buy_ema'], axis=1, inplace=True)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[:, 'buy_tag'] = ''
        dataframe = super().populate_buy_trend(dataframe, metadata)
        dataframe = self.populate_buy_trend_sr(dataframe=dataframe, metadata=metadata)

        # Kontrola vzdálenosti k supportu
        if metadata['pair'] in self.support_dict:
            s = self.support_dict[metadata['pair']]
            if s:
                # Vypočítání nejbližší úrovně supportu pro každou svíčku, která je pod aktuální cenou
                dataframe['nearest_support'] = dataframe['close'].apply(
                    lambda x: min([support for support in s if support <= x], default=x,
                                  key=lambda support: abs(x - support))
                )

                if 'nearest_support' in dataframe.columns:
                    # Vypočítání procentního rozdílu mezi cenou a nejbližším supportem
                    dataframe['distance_to_support_pct'] = (
                                                                   dataframe['nearest_support'] - dataframe['close']) / \
                                                           dataframe['close'] * 100

                    # Vygenerování nákupních signálů na základě supportu
                    buy_threshold = 0.1  # 0.1 %
                    dataframe.loc[
                        (dataframe['distance_to_support_pct'] >= 0) &
                        (dataframe['distance_to_support_pct'] <= buy_threshold),
                        'buy_support'
                    ] = 1

                    dataframe.loc[
                        (dataframe['distance_to_support_pct'] >= 0) &
                        (dataframe['distance_to_support_pct'] <= buy_threshold),
                        'buy_tag'
                    ] += 'sr_buy'

                    # Odebrání pomocných sloupců
                    dataframe.drop(['nearest_support', 'distance_to_support_pct'],
                                   axis=1, inplace=True)

        # Přidání podmínek pro EMA a objem
        dataframe.loc[(dataframe['volume'] > 0) & (dataframe['ema_diff_buy_signal'].astype(int) > 0), 'buy_ema'] = 1
        dataframe.loc[
            (dataframe['volume'] > 0) & (dataframe['ema_diff_buy_signal'].astype(int) > 0), 'buy_tag'] += 'ema_dbs_'

        # Generování nákupních signálů pouze pokud jsou splněny obě podmínky
        dataframe.loc[(dataframe['buy_support'] == 1) & (dataframe['buy_ema'] == 1) & (
                dataframe['rsi'] <= dataframe['weighted_rsi']), 'buy'] = 1

        # Odebrání pomocných sloupců
        if 'buy_support' in dataframe.columns:
            dataframe.drop(['buy_support'], axis=1, inplace=True)
        if 'buy_ema' in dataframe.columns:
            dataframe.drop(['buy_ema'], axis=1, inplace=True)

        return dataframe

    def calculate_percentage_difference(self, original_price, current_price):
        percentage_diff = ((current_price - original_price) / original_price)
        return percentage_diff

    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                              current_rate: float, current_profit: float, min_stake: float,
                              max_stake: float, **kwargs):
        # Aktualizace aktuálního času na současný čas UTC
        current_time = datetime.utcnow()  # Datový typ: datetime

        try:
            # Získání analyzovaného dataframe pro daný obchodní pár a časový rámec
            dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
            df = dataframe.copy()  # Datový typ: pandas DataFrame
        except Exception as e:
            # Logování chyby při získávání dataframe a ukončení metody
            logging.error(f"Error getting analyzed dataframe: {e}")
            return None

        # Kontrola, zda obchodní pár má definovanou podporu
        if trade.pair in self.support_dict:
            # Získání seznamu podpor pro daný obchodní pár
            s = self.support_dict[trade.pair]  # Datový typ: list

            # Výpočet nejbližší podpory pro každou cenu uzavření v dataframe
            df['nearest_support'] = df['close'].apply(
                lambda x: min([support for support in s if support <= x], default=x,
                              key=lambda support: abs(x - support))
            )

            if 'nearest_support' in df.columns:

                # Získání poslední svíčky (candle) z dataframe
                last_candle = df.iloc[-1]  # Datový typ: pandas Series

                if 'nearest_support' in last_candle:
                    nearest_support = last_candle['nearest_support']  # Datový typ: float
                    # Výpočet procentní vzdálenosti k nejbližší podpoře
                    distance_to_support_pct = abs(
                        (nearest_support - current_rate) / current_rate)  # Datový typ: float, jednotka: %
                    # Kontrola, zda je aktuální kurz blízko nebo pod nejbližší podporou
                    if (0 <= distance_to_support_pct <= 0.01) or (current_rate < nearest_support):
                        # Počítání uzavřených nákupních příkazů
                        count_of_buys = sum(order.ft_order_side == 'buy' and order.status == 'closed' for order in
                                            trade.orders)  # Datový typ: int
                        # Zjištění času posledního nákupu
                        last_buy_time = max(
                            [order.order_date for order in trade.orders if order.ft_order_side == 'buy'],
                            default=trade.open_date_utc)
                        last_buy_time = last_buy_time.replace(
                            tzinfo=None)  # Odstranění časové zóny, Datový typ: datetime
                        # Výpočet intervalu svíčky (candle) v minutách
                        candle_interval = self.timeframe_to_minutes(self.timeframe)  # Datový typ: int, jednotka: minuty
                        # Výpočet času od posledního nákupu v minutách
                        time_since_last_buy = (
                                                      current_time - last_buy_time).total_seconds() / 60  # Datový typ: float, jednotka: minuty
                        # Výpočet počtu svíček, které musí uplynout před dalším nákupem
                        candles = 60 + (30 * (count_of_buys - 1))  # Datový typ: int
                        # Kontrola, zda uplynul dostatečný čas od posledního nákupu
                        if time_since_last_buy < candles * candle_interval:
                            return None
                        # Kontrola, zda počet bezpečnostních příkazů (safety orders) není překročen
                        if self.max_safety_orders >= count_of_buys:
                            # Hledání posledního uzavřeného nákupního příkazu
                            last_buy_order = None
                            for order in reversed(trade.orders):
                                if order.ft_order_side == 'buy' and order.status == 'closed':
                                    last_buy_order = order
                                    break
                            # Definice prahové hodnoty pro další nákup
                            pct_threshold = -0.03  # Datový typ: float, jednotka: %
                            # Výpočet procentní rozdílu mezi posledním nákupním příkazem a aktuálním kurzem
                            pct_diff = self.calculate_percentage_difference(original_price=last_buy_order.price,
                                                                            current_price=current_rate)  # Datový typ: float, jednotka: %
                            # Kontrola, zda je procentní rozdíl menší než prahová hodnota
                            if pct_diff <= pct_threshold:
                                if last_buy_order and current_rate < last_buy_order.price:
                                    # Kontrola RSI podmínky pro DCA
                                    rsi_value = last_candle['rsi']  # Předpokládá se, že RSI je součástí dataframe
                                    w_rsi = last_candle[
                                        'weighted_rsi']  # Předpokládá se, že Weighted RSI je součástí dataframe

                                    if rsi_value <= w_rsi:
                                        # Logování informací o obchodu
                                        logging.info(
                                            f'AP1 {trade.pair}, Profit: {current_profit}, Stake {trade.stake_amount}')

                                        # Získání celkové částky sázky v peněžence
                                        total_stake_amount = self.wallets.get_total_stake_amount()  # Datový typ: float

                                        # Výpočet částky pro další sázku pomocí DCA (Dollar Cost Averaging)
                                        calculated_dca_stake = self.calculate_dca_price(base_value=trade.stake_amount,
                                                                                        decline=current_profit * 100,
                                                                                        target_percent=1)  # Datový typ: float
                                        # Upravení velikosti sázky, pokud je vyšší než dostupný zůstatek
                                        while calculated_dca_stake >= total_stake_amount:
                                            calculated_dca_stake = calculated_dca_stake / 4  # Datový typ: float
                                        # Logování informací o upravené sázce
                                        logging.info(f'AP2 {trade.pair}, DCA: {calculated_dca_stake}')
                                        # Vrácení upravené velikosti sázky
                                        return calculated_dca_stake
            # Vrácení None, pokud nejsou splněny podmínky pro upravení obchodní pozice
            return None

    def timeframe_to_minutes(self, timeframe):
        """Převede timeframe na minuty."""
        if timeframe.endswith('m'):
            return int(timeframe[:-1])
        elif timeframe.endswith('h'):
            return int(timeframe[:-1]) * 60
        elif timeframe.endswith('d'):
            return int(timeframe[:-1]) * 1440
        else:
            raise ValueError("Neznámý timeframe: {}".format(timeframe))


class HPStrategyJSTKR(HPStrategyTFJPA):
    def version(self) -> str:
        return f"{super().version()} JSTKR "

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = super().populate_buy_trend(dataframe, metadata)
        dataframe.loc[(dataframe['jstkr_2'] == 1), 'buy'] = 1
        dataframe.loc[(dataframe['jstkr_2'] == 1), 'buy_tag'] += 'jstkr_2_'
        return dataframe

    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                              current_rate: float, current_profit: float, min_stake: float,
                              max_stake: float, **kwargs):
        stake_adjusted = super().adjust_trade_position(trade, current_time, current_rate, current_profit, min_stake,
                                                       max_stake, **kwargs)
        if stake_adjusted:
            return stake_adjusted
        else:
            try:
                # Získání analyzovaného dataframe pro daný obchodní pár a časový rámec...
                dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
                df = dataframe.copy()  # Datový typ: pandas DataFrame
            except Exception as e:
                # Logování chyby při získávání dataframe a ukončení metody
                logging.error(f"Error getting analyzed dataframe: {e}")
                return None
            last_candle = df.iloc[-1]
            if last_candle['jstkr_2'] > 0:
                # Logování informací o obchodu
                logging.info(f'AP1 {trade.pair}, Profit: {current_profit}, Stake {trade.stake_amount}')
                # Získání celkové částky sázky v peněžence
                total_stake_amount = self.wallets.get_total_stake_amount()  # Datový typ: float
                # Výpočet částky pro další sázku pomocí DCA (Dollar Cost Averaging)
                calculated_dca_stake = self.calculate_dca_price(base_value=trade.stake_amount,
                                                                decline=current_profit * 100,
                                                                target_percent=1)  # Datový typ: float
                # Upravení velikosti sázky, pokud je vyšší než dostupný zůstatek
                while calculated_dca_stake >= total_stake_amount:
                    calculated_dca_stake = calculated_dca_stake / 4  # Datový typ: float
                # Logování informací o upravené sázce
                logging.info(f'AP2 {trade.pair}, DCA: {calculated_dca_stake}')
                # Vrácení upravené velikosti sázky
                return calculated_dca_stake
        return None
