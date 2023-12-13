# --- Do not remove these libs ---

import datetime
import json
import logging
import math
import os
from datetime import datetime, timedelta, timezone
from functools import reduce
from typing import Optional

import numpy as np
import pandas as pd
# --------------------------------
import talib.abstract as ta
from pandas import DataFrame

import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.persistence import Trade
from freqtrade.strategy import merge_informative_pair, DecimalParameter, IntParameter
from freqtrade.strategy.interface import IStrategy

logger = logging.getLogger(__name__)


def EWO(dataframe, ema_length=5, ema2_length=3):
    df = dataframe.copy()
    ema1 = ta.EMA(df, timeperiod=ema_length)
    ema2 = ta.EMA(df, timeperiod=ema2_length)
    return (ema1 - ema2) / df['close'] * 100


def pct_change(a, b):
    return (b - a) / a


class HPStrategy(IStrategy):
    INTERFACE_VERSION = 2

    """
    # ROI table:
    minimal_roi = {
    "0": 0.08,
        "20": 0.04,
        "40": 0.032,
        "87": 0.016,
        "201": 0,
        "202": -1
    }
    """

    # Buy hyperspace params:
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

    # Sell hyperspace params:
    sell_params = {
        "base_nb_candles_sell": 22,
        "high_offset": 1.014,
        "high_offset_2": 1.01
    }

    order_types = {
        'entry': 'market',
        'exit': 'market',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    @property
    def protections(self):
        return [
            {
                "method": "CooldownPeriod",
                "stop_duration_candles": 5
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

    # ROI table:
    minimal_roi = {
        "0": 0.99,
    }

    lowest_prices = {}
    highest_prices = {}
    price_drop_percentage = {}
    pairs_close_to_high = []
    locked = []

    # Stoploss:
    stoploss = -0.99

    # SMAOffset
    base_nb_candles_buy = IntParameter(8, 20, default=buy_params['base_nb_candles_buy'], space='buy', optimize=False)
    base_nb_candles_sell = IntParameter(8, 20, default=sell_params['base_nb_candles_sell'], space='sell',
                                        optimize=False)
    low_offset = DecimalParameter(0.975, 0.995, default=buy_params['low_offset'], space='buy', optimize=True)
    high_offset = DecimalParameter(1.000, 1.010, default=sell_params['high_offset'], space='sell', optimize=True)
    high_offset_2 = DecimalParameter(1.000, 1.010, default=sell_params['high_offset_2'], space='sell', optimize=True)

    # lambo2
    lambo2_ema_14_factor = DecimalParameter(0.8, 1.2, decimals=3, default=buy_params['lambo2_ema_14_factor'],
                                            space='buy', optimize=True)
    lambo2_rsi_4_limit = IntParameter(10, 60, default=buy_params['lambo2_rsi_4_limit'], space='buy', optimize=True)
    lambo2_rsi_14_limit = IntParameter(10, 60, default=buy_params['lambo2_rsi_14_limit'], space='buy', optimize=True)

    # Protection
    fast_ewo = 50
    slow_ewo = 200

    ewo_low = DecimalParameter(-20.0, -7.0, default=buy_params['ewo_low'], space='buy', optimize=True)
    ewo_high = DecimalParameter(3.0, 5, default=buy_params['ewo_high'], space='buy', optimize=True)
    rsi_buy = IntParameter(30, 70, default=buy_params['rsi_buy'], space='buy', optimize=False)

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.001
    trailing_stop_positive_offset = 0.012
    trailing_only_offset_is_reached = True

    # cofi
    is_optimize_cofi = False
    buy_ema_cofi = DecimalParameter(0.96, 0.98, default=0.97, optimize=is_optimize_cofi)
    buy_fastk = IntParameter(20, 30, default=20, optimize=is_optimize_cofi)
    buy_fastd = IntParameter(20, 30, default=20, optimize=is_optimize_cofi)
    buy_adx = IntParameter(20, 30, default=30, optimize=is_optimize_cofi)
    buy_ewo_high = DecimalParameter(2, 12, default=3.553, optimize=is_optimize_cofi)

    # Sell signal
    use_sell_signal = True
    sell_profit_only = True
    sell_profit_offset = 0.01
    ignore_roi_if_buy_signal = False

    # Adjusting
    position_adjustment_enable = True
    ## Optional order time in force.
    order_time_in_force = {
        'buy': 'gtc',
        'sell': 'gtc'
    }

    # Optimal timeframe for the strategy
    timeframe = '5m'
    inf_1h = '1h'

    process_only_new_candles = True
    startup_candle_count = 400

    plot_config = {
        'main_plot': {
            'ma_buy': {'color': 'orange'},
            'ma_sell': {'color': 'orange'},
        },
    }

    def version(self) -> str:
        return f"{super().version()} HPStrategy "

    def custom_sell(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                    current_profit: float, **kwargs):
        # Sell any positions at a loss if they are held for more than 7 days.
        if current_profit < -0.04 and (current_time - trade.open_date_utc).days >= 4:
            return 'unclog'

    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: Optional[float], max_stake: float,
                            leverage: float, entry_tag: Optional[str], side: str, **kwargs) -> float:
        min_trade_size = 3
        if proposed_stake < min_trade_size:
            return 0  # Nedostatečná sázka pro obchod
        max_stake_for_three_trades = max_stake / 3
        if max_stake_for_three_trades < min_trade_size:
            return 0  # Nedostatek prostředků pro obchodování
        return min(proposed_stake, max_stake_for_three_trades)

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

        df36h = dataframe.copy().shift(432)  # TODO FIXME: This assumes 5m timeframe
        df24h = dataframe.copy().shift(288)  # TODO FIXME: This assumes 5m timeframe

        dataframe['volume_mean_short'] = dataframe['volume'].rolling(4).mean()
        dataframe['volume_mean_long'] = df24h['volume'].rolling(48).mean()
        dataframe['volume_mean_base'] = df36h['volume'].rolling(288).mean()

        dataframe['volume_change_percentage'] = (dataframe['volume_mean_long'] / dataframe['volume_mean_base'])

        dataframe['rsi_mean'] = dataframe['rsi'].rolling(48).mean()

        dataframe['pnd_volume_warn'] = np.where((dataframe['volume_mean_short'] / dataframe['volume_mean_long'] > 5.0),
                                                -1, 0)

        return dataframe

    def base_tf_btc_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Indicators
        # -----------------------------------------------------------------------------------------
        dataframe['price_trend_long'] = (
                dataframe['close'].rolling(8).mean() / dataframe['close'].shift(8).rolling(144).mean())

        # Add prefix
        # -----------------------------------------------------------------------------------------
        ignore_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
        dataframe.rename(columns=lambda s: f"btc_{s}" if s not in ignore_columns else s, inplace=True)

        return dataframe

    def info_tf_btc_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Indicators
        # -----------------------------------------------------------------------------------------
        dataframe['rsi_8'] = ta.RSI(dataframe, timeperiod=8)

        # Add prefix
        # -----------------------------------------------------------------------------------------
        ignore_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
        dataframe.rename(columns=lambda s: f"btc_{s}" if s not in ignore_columns else s, inplace=True)

        return dataframe

    def save_dictionaries_to_disk(self):
        try:
            # Cesta k adresáři, kde chcete slovníky uložit
            user_data_directory = os.path.join('user_data')

            # Ujistěte se, že adresář existuje
            if not os.path.exists(user_data_directory):
                os.makedirs(user_data_directory)

            # Uložení slovníku lowest_prices
            with open(os.path.join(user_data_directory, 'lowest_prices.json'), 'w') as file:
                json.dump(self.lowest_prices, file, indent=4)

            # Uložení slovníku highest_prices
            with open(os.path.join(user_data_directory, 'highest_prices.json'), 'w') as file:
                json.dump(self.highest_prices, file, indent=4)

            # Uložení slovníku price_drop_percentage
            with open(os.path.join(user_data_directory, 'price_drop_percentage.json'), 'w') as file:
                json.dump(self.price_drop_percentage, file, indent=4)

        except Exception as ex:
            logging.error(str(ex))

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe['price_history'] = dataframe['close'].shift(1)
        data_last_bbars = dataframe[-30:].copy()
        # Výpočet %K
        low_min = dataframe['low'].rolling(window=14).min()
        high_max = dataframe['high'].rolling(window=14).max()
        dataframe['stoch_k'] = 100 * (dataframe['close'] - low_min) / (high_max - low_min)
        # Výpočet %D
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

        # Calculate all ma_buy values
        for val in self.base_nb_candles_buy.range:
            dataframe[f'ma_buy_{val}'] = ta.EMA(dataframe, timeperiod=val)

        # Calculate all ma_sell values
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

        dataframe = self.pump_dump_protection(dataframe, metadata)

        # Najdeme lokální minima pro cenu a RSI
        low_min = dataframe['low'].rolling(window=14, center=True).apply(lambda x: np.argmin(x) == 7, raw=True)
        rsi_min = dataframe['rsi'].rolling(window=14, center=True).apply(lambda x: np.argmin(x) == 7, raw=True)
        # Identifikujeme bullish divergenci (cena dělá nová minima, ale RSI ne)
        bullish_div = (low_min.notna()) & (rsi_min.shift() > rsi_min)
        dataframe['bullish_divergence'] = bullish_div.astype(int)

        # Fractals
        # Definice fraktalu pro vrchol a dno
        dataframe['fractal_top'] = (dataframe['high'] > dataframe['high'].shift(2)) & \
                                   (dataframe['high'] > dataframe['high'].shift(1)) & \
                                   (dataframe['high'] > dataframe['high'].shift(-1)) & \
                                   (dataframe['high'] > dataframe['high'].shift(-2))
        dataframe['fractal_bottom'] = (dataframe['low'] < dataframe['low'].shift(2)) & \
                                      (dataframe['low'] < dataframe['low'].shift(1)) & \
                                      (dataframe['low'] < dataframe['low'].shift(-1)) & \
                                      (dataframe['low'] < dataframe['low'].shift(-2))
        # Označení možného bodu obratu
        dataframe['turnaround_signal'] = (bullish_div) & (dataframe['fractal_bottom'])
        # Výpočet 'rolling maximum' - nejvyšší ceny dosažené do daného momentu
        dataframe['rolling_max'] = dataframe['high'].cummax()
        # Výpočet propadu ceny v procentech z 'rolling maximum'
        dataframe['drawdown'] = (dataframe['rolling_max'] - dataframe['low']) / dataframe['rolling_max']
        # Identifikace bodů, kde je propad ceny 90% nebo více
        dataframe['below_90_percent_drawdown'] = dataframe['drawdown'] >= 0.90

        dataframe.drop_duplicates()

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        self.analyze_price_movements(dataframe=dataframe, metadata=metadata, window=200)
        better_pair = metadata['pair'] not in self.pairs_close_to_high

        dataframe.loc[:, 'buy_tag'] = ''

        lambo2 = (
            # bool(self.lambo2_enabled.value) &
            # (dataframe['pump_warning'] == 0) &
                (dataframe['close'] < (dataframe['ema_14'] * self.lambo2_ema_14_factor.value)) &
                (dataframe['rsi_4'] < int(self.lambo2_rsi_4_limit.value)) &
                (dataframe['rsi_14'] < int(self.lambo2_rsi_14_limit.value))
        )
        dataframe.loc[lambo2, 'buy_tag'] += 'lambo2_'
        conditions = [lambo2]
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
            dataframe['btc_rsi_8_1h'] < 35.0,
        ]

        # Kombinace nové podmínky s ostatními existujícími podmínkami
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

    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                           rate: float, time_in_force: str, sell_reason: str,
                           current_time: datetime, **kwargs) -> bool:
        sell_reason = f"{sell_reason}_" + trade.buy_tag

        current_profit = trade.calc_profit_ratio(rate)
        return (
                current_profit >= self.sell_profit_offset
                or 'unclog' in sell_reason
                or 'force' in sell_reason
        )


class HPStrategyDCA(HPStrategy):
    initial_safety_order_trigger = -0.018
    max_safety_orders = 3
    safety_order_step_scale = 1.2
    safety_order_volume_scale = 1.4

    drawdown_limit = -2

    buy_params = {
        "dca_min_rsi": 35,
    }

    # append buy_params of parent class
    buy_params.update(HPStrategy.buy_params)

    dca_min_rsi = IntParameter(35, 75, default=buy_params['dca_min_rsi'], space='buy', optimize=True)

    def version(self) -> str:
        return f"{super().version()} DCA "

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = super().populate_indicators(dataframe, metadata)
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        return dataframe

    def calculate_volatility(self, dataframe: DataFrame, pair: str, timeframe: str) -> float:
        """
        Vypočítá volatilitu pro zadaný měnový pár a vrátí průměrnou absolutní hodnotu procentuální změny
        za posledních 24 hodin, závisí na časovém rámci (timeframe), v procentech.
        """
        # Převedení timeframe na počet minut
        timeframes_in_minutes = {
            '1m': 1,
            '5m': 5,
            '15m': 15,
            '30m': 30,
            '1h': 60,
            '4h': 240,
            '1d': 1440
        }

        # Získání počtu minut pro jeden interval
        interval_in_minutes = timeframes_in_minutes.get(timeframe)

        if interval_in_minutes is None:
            raise ValueError("Neplatný timeframe. Prosím, zadejte jeden z podporovaných timeframe.")

        # Počet period pro pokrytí 24 hodin
        periods = int(24 * 60 / interval_in_minutes)

        # Výpočet procentuální změny
        dataframe['pct_change'] = dataframe['close'].pct_change()

        return dataframe['pct_change'].tail(periods).abs().mean() * 100

    def dynamic_stake_adjustment(self, stake, volatility):
        """
        Dynamically adjust the stake based on the market's volatility.
        """
        return stake * 0.8 if volatility > 0.05 else stake

    def check_buy_conditions(self, last_candle, previous_candle):
        """
        Check if the buy conditions are met for the given candle.
        """
        # Lambo2 condition
        lambo2 = (
                (last_candle['close'] < (last_candle['ema_14'] * self.lambo2_ema_14_factor.value)) &
                (last_candle['rsi_4'] < int(self.lambo2_rsi_4_limit.value)) &
                (last_candle['rsi_14'] < int(self.lambo2_rsi_14_limit.value))
        )
        conditions = [lambo2]
        # Buy condition based on EWO, RSI, and moving averages
        buy1ewo = (
                (last_candle['rsi_fast'] < 35) &
                (last_candle['close'] < (
                        last_candle[f'ma_buy_{self.base_nb_candles_buy.value}'] * self.low_offset.value)) &
                (last_candle['EWO'] > self.ewo_high.value) &
                (last_candle['rsi'] < self.rsi_buy.value) &
                (last_candle['volume'] > 0) &
                (last_candle['close'] < (
                        last_candle[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset.value))
        )
        conditions.append(buy1ewo)

        buy2ewo = (
                (last_candle['rsi_fast'] < 35) &
                (last_candle['close'] < (
                        last_candle[f'ma_buy_{self.base_nb_candles_buy.value}'] * self.low_offset.value)) &
                (last_candle['EWO'] < self.ewo_low.value) &
                (last_candle['volume'] > 0) &
                (last_candle['close'] < (
                        last_candle[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset.value))
        )
        conditions.append(buy2ewo)

        # Manuální kontrola překřížení pro Cofi podmínku
        crossed_above_fastk_fastd = (previous_candle['fastk'] < previous_candle['fastd']) and (
                last_candle['fastk'] > last_candle['fastd'])
        is_cofi = (
                (last_candle['open'] < last_candle['ema_8'] * self.buy_ema_cofi.value) &
                crossed_above_fastk_fastd &
                (last_candle['fastk'] < self.buy_fastk.value) &
                (last_candle['fastd'] < self.buy_fastd.value) &
                (last_candle['adx'] > self.buy_adx.value) &
                (last_candle['EWO'] > self.buy_ewo_high.value)
        )
        conditions.append(is_cofi)

        # Final decision: Check if any of the conditions are True
        return any(conditions)

    def calculate_drawdown(self, current_price, last_order_price):
        return (current_price - last_order_price) / last_order_price * 100

    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                              current_rate: float, current_profit: float, min_stake: float,
                              max_stake: float, **kwargs):
        try:
            dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
            df = dataframe.copy()
        except Exception as e:
            return None

        volatility = self.calculate_volatility(df, trade.pair, self.timeframe)
        adjusted_min_stake = self.dynamic_stake_adjustment(min_stake, volatility)
        adjusted_max_stake = self.dynamic_stake_adjustment(max_stake, volatility)

        last_candle = df.iloc[-1].squeeze()
        previous_candle = df.iloc[-2].squeeze()
        if last_candle['close'] < previous_candle['close']:
            return None

        if not self.check_buy_conditions(last_candle, previous_candle):
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

            if drawdown <= self.drawdown_limit:
                try:
                    stake_amount = self.wallets.get_trade_stake_amount(trade.pair, None)
                    stake_amount = min(stake_amount * math.pow(self.safety_order_volume_scale, (count_of_buys - 1)),
                                       adjusted_max_stake)
                    if stake_amount < adjusted_min_stake:
                        return None

                    try:
                        # Výpočet rychlosti cenového pohybu
                        price_change_rate = (last_candle['close'] - previous_candle['close']) / previous_candle['close']
                        # Nastavení dynamických hranic sázky
                        if price_change_rate < -0.02:  # Pokud cena klesá o více než 2%
                            adjusted_stake = stake_amount * 1.5  # Zvyšte sázku o 50%
                        elif price_change_rate > 0.02:  # Pokud cena roste o více než 2%
                            adjusted_stake = stake_amount * 0.75  # Snížte sázku o 25%
                        else:
                            adjusted_stake = stake_amount  # Ponechte minimální sázku
                    except:
                        adjusted_stake = stake_amount
                    return adjusted_stake

                except Exception as exception:
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
