import json
import logging
import os
from datetime import datetime
from functools import reduce
from typing import Optional

import numpy as np
import pandas as pd
import talib
import talib.abstract as ta
from pandas import DataFrame
from technical.indicators import ichimoku

import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.persistence import Trade
from freqtrade.strategy import merge_informative_pair, DecimalParameter, IntParameter
from freqtrade.strategy.interface import IStrategy


class HPStrategyTFJPAConfirmV3(IStrategy):
    INTERFACE_VERSION = 3
    can_short = False
    support_dict = {}
    resistance_dict = {}
    lowest_prices = {}
    highest_prices = {}
    price_drop_percentage = {}
    pairs_close_to_high = []
    locked = []
    stoploss = -0.99

    use_exit_signal = True
    exit_profit_only = True
    ignore_roi_if_entry_signal = False
    position_adjustment_enable = True
    order_time_in_force = {
        'entry': 'gtc',
        'exit': 'gtc'
    }

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

    # jstrk_adjust = True

    buy_params = {
        "buy_adx": 21,
        "buy_ema_cofi": 0.962,
        "buy_ewo_high": 4.568,
        "buy_fastd": 21,
        "buy_fastk": 20,
        "candles_before": 20,
        "candles_dca_multiplier": 7,
        "dca_order_divider": 6,
        "dca_wallet_divider": 4,
        "distance_to_support_treshold": 0.043,
        "max_safety_orders": 3,
        "pct_drop_treshold": 0.013,
        "rsi_buy": 69,
        "base_nb_candles_buy": 12,  # value loaded from strategy
        "ewo_high": 3.001,  # value loaded from strategy
        "ewo_low": -10.289,  # value loaded from strategy
        "lambo2_ema_14_factor": 0.981,  # value loaded from strategy
        "lambo2_rsi_14_limit": 39,  # value loaded from strategy
        "lambo2_rsi_4_limit": 44,  # value loaded from strategy
        "low_offset": 0.987,  # value loaded from strategy
        "open_trade_limit": 5,  # value loaded from strategy
        "stoch_treshold": 25,  # value loaded from strategy
    }

    # Sell hyperspace params:
    sell_params = {
        "base_nb_candles_sell": 13,  # value loaded from strategy
        "high_offset": 1.002,  # value loaded from strategy
        "high_offset_2": 1.0,  # value loaded from strategy
    }

    # ROI table:  # value loaded from strategy
    minimal_roi = {
        "0": 0.109,
        "17": 0.035,
        "51": 0.011,
        "104": 0
    }

    is_optimize_dca = True
    is_optimize_sr = True
    is_optimize_cofi = True

    stoch_treshold = IntParameter(20, 40, default=buy_params['stoch_treshold'], space='buy', optimize=False)

    distance_to_support_treshold = DecimalParameter(0.01, 0.05, default=buy_params['distance_to_support_treshold'],
                                                    space='buy', optimize=is_optimize_sr)
    pct_drop_treshold = DecimalParameter(0.01, 0.05, default=buy_params['pct_drop_treshold'], space='buy',
                                         optimize=is_optimize_dca)
    candles_before = IntParameter(10, 20, default=buy_params['candles_before'], space='buy',
                                  optimize=is_optimize_dca)
    candles_dca_multiplier = IntParameter(1, 30, default=buy_params['candles_dca_multiplier'], space='buy',
                                          optimize=is_optimize_dca)
    open_trade_limit = IntParameter(1, 10, default=buy_params['open_trade_limit'], space='buy', optimize=False)

    dca_wallet_divider = IntParameter(open_trade_limit.value - 1, 10, default=buy_params['dca_wallet_divider'],
                                      space='buy', optimize=is_optimize_dca)

    dca_order_divider = IntParameter(2, 10, default=buy_params['dca_order_divider'], space='buy',
                                     optimize=is_optimize_dca)

    max_safety_orders = IntParameter(1, 10, default=buy_params['max_safety_orders'], space='buy',
                                     optimize=is_optimize_dca)

    buy_ema_cofi = DecimalParameter(0.96, 0.98, default=0.97, optimize=is_optimize_cofi)
    buy_fastk = IntParameter(20, 30, default=20, optimize=is_optimize_cofi)
    buy_fastd = IntParameter(20, 30, default=20, optimize=is_optimize_cofi)
    buy_adx = IntParameter(20, 30, default=30, optimize=is_optimize_cofi)
    buy_ewo_high = DecimalParameter(2, 12, default=3.553, optimize=is_optimize_cofi)
    base_nb_candles_buy = IntParameter(8, 20, default=buy_params['base_nb_candles_buy'], space='buy', optimize=False)
    low_offset = DecimalParameter(0.975, 0.995, default=buy_params['low_offset'], space='buy', optimize=False)

    rsi_buy = IntParameter(30, 70, default=buy_params['rsi_buy'], space='buy', optimize=True)

    lambo2_ema_14_factor = DecimalParameter(0.8, 1.2, decimals=3, default=buy_params['lambo2_ema_14_factor'],
                                            space='buy', optimize=False)
    lambo2_rsi_4_limit = IntParameter(10, 60, default=buy_params['lambo2_rsi_4_limit'], space='buy', optimize=False)
    lambo2_rsi_14_limit = IntParameter(10, 60, default=buy_params['lambo2_rsi_14_limit'], space='buy', optimize=False)

    fast_ewo = 50
    slow_ewo = 200
    ewo_low = DecimalParameter(-20.0, -7.0, default=buy_params['ewo_low'], space='buy', optimize=False)
    ewo_high = DecimalParameter(3.0, 5, default=buy_params['ewo_high'], space='buy', optimize=False)

    trailing_stop = True
    trailing_stop_positive = 0.206
    trailing_stop_positive_offset = 0.278
    trailing_only_offset_is_reached = True

    max_open_trades = 25
    amend_last_stake_amount = True

    start = 0.02
    increment = 0.02
    maximum = 0.2

    order_types = {
        'entry': 'market',
        'exit': 'market',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    base_nb_candles_sell = IntParameter(8, 20, default=sell_params['base_nb_candles_sell'], space='sell', optimize=True)
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
        return f"{super().version()} TFJPAConfirmV3 "

    def create_static_config(self):
        try:
            # Načtení aktuální konfigurace
            cf = self.config['config_files'][0]
            cfnew = cf.replace('.json', '_static.json')
            if not os.path.exists(cfnew):
                current_config = json.loads(open(cf, 'r').read())
                # Získání whitelistu měn (příklad, přizpůsobte dle vaší strategie)
                current_whitelist = self.dp.current_whitelist()
                # Nahrazení whitelistu v konfiguraci statickým seznamem
                current_config['exchange']['pair_whitelist'] = current_whitelist
                stp = [{"method": "StaticPairList", "number_assets": len(current_whitelist)}]
                current_config['pairlists'] = stp
                current_config['timeframe'] = self.timeframe
                # Uložení upravené konfigurace do souboru
                with open(cfnew, 'w') as config_file:
                    json.dump(current_config, config_file, indent=4)
        except Exception as e:
            print(e)

    def pivot_points(self, high, low, period=10):
        pivot_high = high.rolling(window=2 * period + 1, center=True).max()
        pivot_low = low.rolling(window=2 * period + 1, center=True).min()
        return high == pivot_high, low == pivot_low

    def calculate_support_resistance(self, df, period=10):
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

    def percentage_drop_indicator(self, dataframe, period, threshold=0.3):
        # Výpočet nejvyšší ceny za poslední období
        highest_high = dataframe['high'].rolling(period).max()
        # Vypočet procentuálního poklesu pod nejvyšší cenou
        percentage_drop = (highest_high - dataframe['close']) / highest_high * 100
        dataframe.loc[percentage_drop < threshold, 'percentage_drop_buy'] = 1
        dataframe.loc[percentage_drop > threshold, 'percentage_drop_buy'] = 0
        return dataframe

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

    def normalize_macd_value(self, value, min_val, max_val):
        if isinstance(value, tuple) or isinstance(min_val, tuple) or isinstance(max_val, tuple):
            return None
        # Normalization of value to a range between 1 and 10
        normalized = ((value - min_val) / (max_val - min_val)) * 9 + 1
        return normalized

    def red_candle_diff(self, dataframe: DataFrame) -> float:
        dataframe['rozdil_cervene_svicky'] = 0.0
        # urceni jesti je svicka cervena a pokud ano, urceni jejiho rozdilu mezi open a closed
        mask = dataframe['open'] > dataframe['close']
        dataframe.loc[mask, 'rozdil_cervene_svicky'] = dataframe['open'] - dataframe['close']
        return dataframe['rozdil_cervene_svicky'].astype(float)

    def normalize_to_0_100(self, dataframe: DataFrame):

        result = []
        window_size = 3  # Velikost časového okna 12 hodin

        for i in range(len(dataframe)):
            start_index = max(0, i - window_size + 1)  # Index začátku časového okna
            end_index = i + 1  # Index konce časového okna

            # Získání hodnot v časovém okně
            window_values = dataframe[start_index:end_index]

            # Najdi nejvyšší a nejnižší hodnotu v časovém okně
            max_value = max(window_values)
            min_value = min(window_values)

            # Výpočet normalizované hodnoty pro aktuální hodnotu MACD v časovém okně
            normalized_value = ((dataframe[i] - min_value) / (
                    max_value - min_value)) * 100000 if max_value != min_value else 0
            result.append(normalized_value)

        return result

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        self.create_static_config()

        # Výpočet DOJI
        try:
            open = dataframe['open']
            high = dataframe['high']
            low = dataframe['low']
            close = dataframe['close']
            # TA-LIB Doji Scanner
            doji = talib.CDLDOJI(open, high, low, close)
            doji_star = talib.CDLDRAGONFLYDOJI(open, high, low, close)
            # Create a Doji Column, replace Doji points with with an candle close price
            dataframe['doji'] = doji
            dataframe['doji'] = np.where(dataframe['doji'] > 0, close, 0)
            dataframe['doji_star'] = doji_star
            dataframe['doji_star'] = np.where(dataframe['doji_star'] > 0, close, 0)
            # Create new Complete_Doji column, merges both columns together and finds maximas
            dataframe['complete_doji'] = np.where(dataframe['doji'] > 0, dataframe['doji'], dataframe['doji_star'])
            complete_doji = dataframe['complete_doji']
            dataframe['complete_doji_idx'] = np.where(dataframe['complete_doji'] > 0, complete_doji.index, 0)
            # TA-LIB Trend line scanner
            trend_line = talib.HT_TRENDLINE(close)
            # Replace first 63 rows, 'NaN' values with 0s, create a new Trendline Column
            trend_line.fillna(0, inplace=True)
            dataframe['trend_line'] = trend_line
            # TA-LIB RSI Scanner
            rsi = talib.RSI(close, timeperiod=7)
            # Fill 'NaN' values with 0s
            dataframe['rsi'] = rsi.fillna(0)
            # Create new columns for RSI70 and RSI30 and fill with rsi values if Above/Below
            dataframe['rsi70'] = np.where(dataframe['rsi'] >= 70, rsi, 0)
            dataframe['rsi30'] = np.where(dataframe['rsi'] <= 30, rsi, 0)
            # Create Above/Below trend columns and insert Doji prices if Above/Below trend line
            dataframe['above_trend'] = np.where(dataframe['complete_doji'] > dataframe['trend_line'], dataframe.complete_doji, 0)
            dataframe['below_trend'] = np.where(dataframe['complete_doji'] < dataframe['trend_line'], dataframe.complete_doji, 0)
            dataframe['five_max'] = np.where(dataframe['above_trend'] > 0, 1, 0)
            dataframe['five_min'] = np.where(dataframe['below_trend'] > 0, 1, 0)
            dataframe.merge(dataframe, on='date')
        except Exception as e:
            logging.error(f"Error getting analyzed dataframe: {e}")
            pass

        # Výpočet Stochastic
        dataframe['price_history'] = dataframe['close'].shift(1)
        low_min = dataframe['low'].rolling(window=14).min()
        high_max = dataframe['high'].rolling(window=14).max()
        dataframe['stoch_k'] = 100 * (dataframe['close'] - low_min) / (high_max - low_min)
        dataframe['stoch_d'] = dataframe['stoch_k'].rolling(window=3).mean()

        # Výpočet Fibonacciho retracement úrovní
        dataframe['high_max'] = dataframe['high'].rolling(window=30).max()  # posledních 30 svíček
        dataframe['low_min'] = dataframe['low'].rolling(window=30).min()

        # Výpočet Fibonacciho úrovní
        diff = dataframe['high_max'] - dataframe['low_min']
        dataframe['fib_236'] = dataframe['high_max'] - 0.236 * diff
        dataframe['fib_382'] = dataframe['high_max'] - 0.382 * diff
        dataframe['fib_500'] = dataframe['high_max'] - 0.500 * diff
        dataframe['fib_618'] = dataframe['high_max'] - 0.618 * diff
        dataframe['fib_786'] = dataframe['high_max'] - 0.786 * diff

        if self.config['stake_currency'] in ['USDT', 'BUSD']:
            btc_info_pair = f"BTC/{self.config['stake_currency']}"
        else:
            btc_info_pair = "BTC/USDT"

        btc_info_tf = self.dp.get_pair_dataframe(btc_info_pair, self.inf_1h)
        btc_info_tf = self.info_tf_btc_indicators(btc_info_tf)
        dataframe = merge_informative_pair(dataframe, btc_info_tf, self.timeframe, self.inf_1h, ffill=True)
        drop_columns = [f"{s}_{self.inf_1h}" for s in ['date', 'open', 'high', 'low', 'close', 'volume']]
        dataframe.drop(columns=dataframe.columns.intersection(drop_columns), inplace=True)

        btc_base_tf = self.dp.get_pair_dataframe(btc_info_pair, self.timeframe)
        btc_base_tf = self.base_tf_btc_indicators(btc_base_tf)
        dataframe = merge_informative_pair(dataframe, btc_base_tf, self.timeframe, self.timeframe, ffill=True)
        drop_columns = [f"{s}_{self.timeframe}" for s in ['date', 'open', 'high', 'low', 'close', 'volume']]
        dataframe.drop(columns=dataframe.columns.intersection(drop_columns), inplace=True)

        for val in self.base_nb_candles_buy.range:
            dataframe[f'ma_buy_{val}'] = ta.EMA(dataframe, timeperiod=val)
        for val in self.base_nb_candles_sell.range:
            dataframe[f'ma_sell_{val}'] = ta.EMA(dataframe, timeperiod=val)

        dataframe['hma_50'] = qtpylib.hull_moving_average(dataframe['close'], window=50)
        dataframe['sma_9'] = ta.SMA(dataframe, timeperiod=9)
        dataframe['sma_50'] = ta.SMA(dataframe, timeperiod=50)

        # Elliot
        dataframe['EWO'] = EWO(dataframe, self.fast_ewo, self.slow_ewo)

        # RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['rsi_fast'] = ta.RSI(dataframe, timeperiod=4)
        dataframe['rsi_slow'] = ta.RSI(dataframe, timeperiod=20)

        # lambo2
        dataframe['ema_14'] = ta.EMA(dataframe, timeperiod=14)
        dataframe['rsi_4'] = ta.RSI(dataframe, timeperiod=4)
        dataframe['rsi_8'] = ta.RSI(dataframe, timeperiod=8)
        dataframe['rsi_14'] = ta.RSI(dataframe, timeperiod=14)

        # Cofi
        stoch_fast = ta.STOCHF(dataframe, 5, 3, 0, 3, 0)
        dataframe['fastd'] = stoch_fast['fastd']
        dataframe['fastk'] = stoch_fast['fastk']
        dataframe['adx'] = ta.ADX(dataframe)
        dataframe['ema_8'] = ta.EMA(dataframe, timeperiod=8)
        dataframe['ema_21'] = ta.EMA(dataframe, timeperiod=21)
        dataframe['ema_50'] = ta.EMA(dataframe, timeperiod=50)

        # ADX
        dataframe['adx'] = ta.ADX(dataframe)

        # Plus Directional Indicator / Movement
        dataframe['plus_dm'] = ta.PLUS_DM(dataframe)
        dataframe['plus_di'] = ta.PLUS_DI(dataframe)

        # # Minus Directional Indicator / Movement
        dataframe['minus_dm'] = ta.MINUS_DM(dataframe)
        dataframe['minus_di'] = ta.MINUS_DI(dataframe)

        # Aroon, Aroon Oscillator
        aroon = ta.AROON(dataframe)
        dataframe['aroonup'] = aroon['aroonup']
        dataframe['aroondown'] = aroon['aroondown']
        dataframe['aroonosc'] = ta.AROONOSC(dataframe)

        # Awesome Oscillator
        dataframe['ao'] = qtpylib.awesome_oscillator(dataframe)

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

        dataframe = self.pump_dump_protection(dataframe)

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

        dataframe['turnaround_signal'] = bullish_div & (dataframe['fractal_bottom'])
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

        dataframe['sar'] = ta.SAR(dataframe, start=self.start, increment=self.increment, maximum=self.maximum)
        dataframe['sar_buy'] = (dataframe['sar'] < dataframe['low']).astype(int)
        dataframe['sar_sell'] = (dataframe['sar'] > dataframe['high']).astype(int)

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

        # Zjištění hladin supportů a resistancí
        self.calculate_support_resistance_dicts(metadata['pair'], dataframe)

        # Přidání RSI a EMA indikátorů
        dataframe['ema'] = ta.EMA(dataframe, timeperiod=5)
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        # Detekce bullish svíčkových vzorů
        dataframe['bullish_engulfing'] = ta.CDLENGULFING(dataframe['open'], dataframe['high'], dataframe['low'],
                                                         dataframe['close']) > 0
        dataframe['hammer'] = ta.CDLHAMMER(dataframe['open'], dataframe['high'], dataframe['low'],
                                           dataframe['close']) > 0
        # Přidání SAR
        dataframe['sar'] = ta.SAR(dataframe, start=self.start, increment=self.increment, maximum=self.maximum)
        dataframe['sar_buy'] = (dataframe['sar'] < dataframe['low']).astype(int)
        dataframe['sar_sell'] = (dataframe['sar'] > dataframe['high']).astype(int)

        # Identifikace úrovní podpory a odporu
        dataframe['support'] = dataframe['close'].rolling(window=20).min()
        dataframe['resistance'] = dataframe['close'].rolling(window=20).max()

        # stochastic_cond = (
        #         (dataframe['stoch_k'] <= self.stoch_treshold.value) &
        #         (dataframe['stoch_d'] <= self.stoch_treshold.value) &
        #         (dataframe['stoch_k'] > dataframe['stoch_d'])
        # )
        # dataframe.loc[stochastic_cond, 'buy_tag'] = 'buy_stoch_kd'
        # dataframe.loc[stochastic_cond, 'buy'] = 1

        # if self.jstrk_adjust:
        #     dataframe['macdnasobek'] = abs(dataframe['macd'] * dataframe['macdsignal']) / 2
        #     dataframe['macdprocenta'] = abs(dataframe['macd']) / abs(dataframe['macdsignal']) * 100
        #     dataframe['macdproc'] = (abs(abs(dataframe['macdsignal']) - abs(dataframe['macd'])) / (
        #             0.5 * abs(dataframe['macdsignal']) - abs(dataframe['macd'])) * 100)
        #     dataframe['macd_procenta_rozdil'] = (
        #             abs(abs(dataframe['macdsignal']) - abs(dataframe['macd'])) / (dataframe['macdsignal'].abs()))
        #     dataframe['macd_procenta_soucet'] = (
        #             abs(abs(dataframe['macdsignal']) + abs(dataframe['macd'])) / (dataframe['macdsignal'].abs()))
        #     dataframe['macd_soucet/2'] = ((abs(dataframe['macdsignal']) + abs(dataframe['macd'])) / 2).abs()
        #     # Normalizace rozdílu MACD a MACD Signál do rozsahu 0-100
        #     macd_values = dataframe['macd']
        #     macd_signal_values = dataframe['macdsignal']
        #     diff = np.array(macd_values) - np.array(macd_signal_values)
        #     normalized_diff = ((diff - np.min(diff)) / (np.max(diff) - np.min(diff))) * 100
        #     # Přidejte normalizované hodnoty do DataFrame
        #     dataframe['normalized_macd_diff'] = normalized_diff.astype(int)
        #     # Zavolejte funkci s daty 'macd_values' a přidejte výsledek do DataFrame
        #     normalized_macd = self.normalize_to_0_100(macd_values)
        #     dataframe['normalized_macd'] = normalized_macd
        #     dataframe['rozdil_cervene_svicky'] = self.red_candle_diff(dataframe)
        #     # vytvorime meritko s min max hodnotami rozdilu mezi rozdily svicek dataframe['rozdil_cervene_svicky'] za 36 hod
        #     dataframe['rozdil_cervene_svicky'] = dataframe['rozdil_cervene_svicky'].astype(float)
        #     # Vytvoření časového okna 36 hodin a určení minima a maxima
        #     dataframe['min_36h'] = dataframe['rozdil_cervene_svicky'].rolling(window=14).min()
        #     dataframe['max_36h'] = dataframe['rozdil_cervene_svicky'].rolling(window=14).max()
        #     # dataframe['min_36h'] = urceni_rozdilu_svicky(dataframe, timeperiod=1).min()
        #     # dataframe['max_36h'] = urceni_rozdilu_svicky(dataframe, timeperiod=1).max()
        #     # Vytvoření sloupce pro procentuální poměr
        #     dataframe['percentualni_pomer'] = (dataframe['rozdil_cervene_svicky'] - dataframe['min_36h']) / (
        #             dataframe['max_36h'] - dataframe['min_36h']) * 100
        #     # normalized value + rsi_mean * rsi_slow * weighted_rsi
        #     dataframe['norm_macd-rsi'] = ((dataframe['normalized_macd'] + (
        #             (dataframe['rsi_mean'] * 2) * dataframe['rsi_slow'] * dataframe['weighted_rsi'])) / dataframe[
        #                                       'macd']) / 1000000
        #
        #     ranges = [
        #         ((-0.00000001, -0.000000099), (0.00000001, 0.000000099)),
        #         ((-0.0000001, -0.00000099), (0.0000001, 0.00000099)),
        #         ((-0.000001, -0.0000099), (0.000001, 0.0000099)),
        #         ((-0.00001, -0.000099), (0.00001, 0.000099)),
        #         ((-0.0001, -0.00099), (0.0001, 0.00099)),
        #         ((-0.001, -0.0099), (0.001, 0.0099)),
        #         ((-0.01, -0.099), (0.01, 0.099)),
        #         ((-0.1, 0.99), (0.1, 0.99)),
        #         ((1, 1000), (1, 1000))
        #     ]
        #
        #     # check the length of 'macd' column in the dataframe
        #     if 'macd' in dataframe and not dataframe['macd'].empty:
        #         values = dataframe['macd']
        #
        #         # check lenght of 'values' against 'ranges'
        #         if len(values) != len(ranges):
        #             print("Error: Lenghth of 'values' doesnt match the length of 'ranges'")
        #         else:
        #             # Normalize values for each graph to a range between 1 and 10
        #             normalized_values_macd = []
        #             for val, (min_val, max_val) in zip(values, ranges):
        #                 normalized_macd = self.normalize_macd_value(val, min_val, max_val)
        #                 if normalized_macd is not None:
        #                     normalized_values_macd.append(normalized_macd)
        #
        #             # Add normalized MACD values to the existing DataFrame
        #             dataframe['norm_macd'] = normalized_macd.astype(float)
        #             dataframe['normalized_values_macd'] = normalized_values_macd.astype(float)
        #     else:
        #         print("Error: 'macd' column is either missing or empty in the DataFrame.")
        #
        #     # posun celeho macd grafu do kladnych cisel
        #     dataframe['macd+5'] = dataframe['macd'] + 5
        #     dataframe['macdsignal+5'] = dataframe['macdsignal'] + 5
        #     dataframe['macd+5'] = dataframe['macd+5'].astype(float)
        #     dataframe['macd+5min36h'] = dataframe['macd+5'].rolling(window=1000).min()
        #     dataframe['macd+5max36h'] = dataframe['macd+5'].rolling(window=1000).max()
        #     dataframe['macd+5percent'] = ((10 * dataframe['macd+5'] - 10 * dataframe['macd+5min36h']) / (
        #             10 * dataframe['macd+5max36h'] - 10 * dataframe['macd+5min36h'])) * 10
        #
        #     # Přidání signálu 'jstkr'
        #     # Vytváří 1, když je součet 'macd' a 'macd_signal' záporný a 'rsi' <= 30
        #     dataframe['jstkr'] = (
        #             (dataframe['macd'] + dataframe['percentualni_pomer'] > 98) & (dataframe['rsi_fast'] <= 3)).astype(
        #         int)
        #     # Výpočet macd_diff_percent podle dané rovnice
        #     dataframe['macd_diff_percent'] = ((abs(dataframe['macdsignal'] - dataframe['macd'])) / (
        #             0.5 * abs(dataframe['macdsignal'] + dataframe['macd']))) * 100
        #     dataframe['macdsoucetid'] = (dataframe['macd'] + dataframe['macdsignal']) / 2
        #     # Příklad hodnot MACD a MACD Signal
        #     macd_values = dataframe['macd']  # Nahraďte skutečnými hodnotami MACD
        #     macd_signal_values = dataframe['macdsignal']  # Nahraďte skutečnými hodnotami MACD Signal
        #     # Vypočítání rozdílu mezi MACD a MACD Signal
        #     diff = np.array(macd_values) - np.array(macd_signal_values)
        #     # Normalizace rozdílu do rozsahu 0-100
        #     normalized_diff = ((diff - np.min(diff)) / (np.max(diff) - np.min(diff))) * 100
        #     dataframe['normalized_diff'] = normalized_diff.astype(int)
        #     dataframe['rsi_mean'] = dataframe['rsi_mean'].astype(float)
        #     dataframe['rsi_mean_fast'] = ((dataframe['rsi_mean'] + dataframe['rsi'] + dataframe[
        #         'weighted_rsi']) / 4) + 5 * \
        #                                  dataframe['macd+5percent']
        #     dataframe['jstkr_2'] = ((dataframe['percentualni_pomer'] < 25.1) &
        #                             (dataframe['percentualni_pomer'] > 0) &
        #                             (dataframe['rsi_mean'] < 60) &
        #                             (dataframe['rsi_mean_fast'] > 25) &
        #                             (dataframe['rsi_mean_fast'] < 60) &
        #                             (dataframe['macd+5percent'] < 0.01) &
        #                             (dataframe['rsi_fast'] > 15)).astype(float)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        mka_conditions = []
        dataframe.loc[:, 'enter_tag'] = ''

        # Define the modified Fibonacci condition with directional check
        fib_cond = (
                (dataframe['close'] <= dataframe['fib_618']) &
                (dataframe['close'] >= dataframe['fib_618'] * 0.99) &
                (dataframe['sma_50'].shift(1) < dataframe['sma_50'])
        )

        dataframe.loc[fib_cond, 'enter_tag'] += 'fib_sma_0618_'
        mka_conditions.append(fib_cond)

        conditions = []
        stochastic_cond = (
                (dataframe['stoch_k'] <= self.stoch_treshold.value) &
                (dataframe['stoch_d'] <= self.stoch_treshold.value) &
                (dataframe['stoch_k'] > dataframe['stoch_d'])
        )
        dataframe.loc[stochastic_cond, 'enter_tag'] += 'stoch_kd_'
        conditions.append(stochastic_cond)

        lambo2 = (
            # bool(self.lambo2_enabled.value) &
            # (dataframe['pump_warning'] == 0) &
                (dataframe['close'] < (dataframe['ema_14'] * self.lambo2_ema_14_factor.value)) &
                (dataframe['rsi_4'] < int(self.lambo2_rsi_4_limit.value)) &
                (dataframe['rsi_14'] < int(self.lambo2_rsi_14_limit.value))
        )
        dataframe.loc[lambo2, 'enter_tag'] += 'lambo2_'
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
        dataframe.loc[buy1ewo, 'enter_tag'] += 'buy1eworsi_'
        conditions.append(buy1ewo)

        buy2ewo = (
                (dataframe['rsi_fast'] < 35) &
                (dataframe['close'] < (dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'] * self.low_offset.value)) &
                (dataframe['EWO'] < self.ewo_low.value) &
                (dataframe['volume'] > 0) &
                (dataframe['close'] < (
                        dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset.value))
        )
        dataframe.loc[buy2ewo, 'enter_tag'] += 'buy2ewo_'
        conditions.append(buy2ewo)

        is_cofi = (
                (dataframe['open'] < dataframe['ema_8'] * self.buy_ema_cofi.value) &
                (qtpylib.crossed_above(dataframe['fastk'], dataframe['fastd'])) &
                (dataframe['fastk'] < self.buy_fastk.value) &
                (dataframe['fastd'] < self.buy_fastd.value) &
                (dataframe['adx'] > self.buy_adx.value) &
                (dataframe['EWO'] > self.buy_ewo_high.value)
        )
        dataframe.loc[is_cofi, 'enter_tag'] += 'cofi_'
        conditions.append(is_cofi)

        # """ Přidání potvrzení nákupního signálu """
        cond_sar = self.confirm_by_sar(dataframe)
        cond_candles = self.confirm_by_candles(dataframe)
        dataframe.loc[cond_sar, 'enter_tag'] += 'sar_'
        dataframe.loc[cond_candles, 'enter_tag'] += 'candles_'
        mka_conditions.append(cond_sar)
        mka_conditions.append(cond_candles)

        # jstkr_conditions = []
        # if self.jstrk_adjust:
        #     if 'jstkr_2' in dataframe.columns:
        #         jstkr_cond = (dataframe['jstkr_2'] == 1)
        #         jstkr_conditions.append(jstkr_cond)
        #         dataframe.loc[jstkr_cond, 'enter_tag'] += 'jstkr_2_'
        #     else:
        #         jstkr_conditions.append(False)
        #         pass

        # Processing final conditions
        if mka_conditions:
            try:
                final_condition_mka = reduce(lambda x, y: x & y, mka_conditions)
                final_condition_orig = reduce(lambda x, y: x | y, conditions)
                # final_condition_jstkr = reduce(lambda x, y: x | y, jstkr_conditions)
                final_condition = reduce(lambda x, y: x | y,
                                         [final_condition_mka, final_condition_orig])
                dataframe.loc[final_condition, 'enter_long'] = 1
            except Exception as e:
                logging.error(f"Error in final condition: {e}")
                pass  # Replace with logging or error handling

        dont_buy_conditions = [
            dataframe['pnd_volume_warn'] < 0.0,
            dataframe['btc_rsi_8_1h'] < 35.0
        ]

        for condition in dont_buy_conditions:
            dataframe.loc[condition, 'enter_long'] = 0

        dataframe.loc[(dataframe['five_min'] == 1), 'enter_long'] = 1
        dataframe.loc[(dataframe['five_min'] == 1), 'enter_tag'] = 'five_min'

        dataframe.loc[:, 'enter_short'] = 0

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[:, 'exit_tag'] = ''

        dataframe.loc[
            (
                    (dataframe['close'] > dataframe['fib_618']) &
                    (dataframe['sma_50'].shift(1) > dataframe['sma_50'])
            ),
            'exit_long'] = 1

        dataframe.loc[(dataframe['five_max'] == 1), 'exit_long'] = 1
        dataframe.loc[(dataframe['five_max'] == 1), 'exit_tag'] = 'five_max'

        dataframe.loc[:, 'exit_short'] = 0
        return dataframe

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                            time_in_force: str, current_time: datetime, entry_tag: Optional[str],
                            side: str, **kwargs) -> bool:
        if 'force' in entry_tag:
            return True
        try:
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            df = dataframe.copy()
            last_candle = df.iloc[-1].squeeze()
            cond_candles = self.confirm_by_candles(last_candle)
            # logging.info(f"{pair} has cond_candles status: {cond_candles}")
            cond_sar = self.confirm_by_sar(last_candle)
            # logging.info(f"{pair} has cond_sar status: {cond_sar}")
            result = (Trade.get_open_trade_count() < self.open_trade_limit.value) and (cond_candles or cond_sar)
            return result
        except Exception as e:
            logging.error(f"Error getting analyzed dataframe: {e}")

        return False

    def custom_exit(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                    current_profit: float, **kwargs):
        if current_profit < -0.05 and (current_time - trade.open_date_utc).days >= 30:
            return 'unclog'

    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                           rate: float, time_in_force: str, exit_reason: str,
                           current_time: datetime, **kwargs) -> bool:
        exit_reason = f"{exit_reason}_{trade.enter_tag}"
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

        if 'unclog' in exit_reason or 'force' in exit_reason:
            # logging.info(f"CTE - FORCE or UNCLOG, EXIT")
            return True

        # Kontrola, zda je aktuální high vyšší než open poslední svíčky
        last_candle = dataframe.iloc[-1]
        if last_candle['high'] > last_candle['open']:
            # logging.info(f"CTE - Cena stále roste (high > open), HOLD")
            return False

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

    def confirm_by_sar(self, data_dict):
        """ Based on TA indicators, populates the buy signal for the given dataframe """
        cond = (data_dict['sar_buy'] > 0)
        return cond

    def confirm_by_candles(self, data_dict):
        """ Based on TA indicators, populates the buy signal for the given dataframe """
        cond = ((data_dict['rsi'] < 30) &
                (data_dict['close'] > data_dict['ema']) &
                (data_dict['bullish_engulfing'] | data_dict['hammer']) &
                (data_dict['low'] < data_dict['support']) | (data_dict['high'] > data_dict['resistance']))
        return cond

    def base_tf_btc_indicators(self, dataframe: DataFrame) -> DataFrame:
        dataframe['price_trend_long'] = (
                dataframe['close'].rolling(8).mean() / dataframe['close'].shift(8).rolling(144).mean())
        ignore_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
        dataframe.rename(columns=lambda s: f"btc_{s}" if s not in ignore_columns else s, inplace=True)
        return dataframe

    def info_tf_btc_indicators(self, dataframe: DataFrame) -> DataFrame:
        dataframe['rsi_8'] = ta.RSI(dataframe, timeperiod=8)
        ignore_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
        dataframe.rename(columns=lambda s: f"btc_{s}" if s not in ignore_columns else s, inplace=True)
        return dataframe

    def pump_dump_protection(self, dataframe: DataFrame) -> DataFrame:
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

    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                              current_rate: float, current_profit: float, min_stake: float,
                              max_stake: float, **kwargs):

        current_time = datetime.utcnow()  # Datový typ: datetime

        try:
            dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
            df = dataframe.copy()  # Datový typ: pandas DataFrame
        except Exception as e:
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

            # Injekce signálu JSTRK
            if 'nearest_support' in df.columns:  # or self.jstrk_adjust:
                # Získání poslední svíčky (candle) z dataframe
                last_candle = df.iloc[-1]  # Datový typ: pandas Series
                # Injekce signálu JSTRK
                if 'nearest_support' in last_candle:  # or self.jstrk_adjust:
                    nearest_support = last_candle['nearest_support']  # Datový typ: float
                    # Výpočet procentní vzdálenosti k nejbližší podpoře
                    distance_to_support_pct = abs(
                        (nearest_support - current_rate) / current_rate)  # Datový typ: float, jednotka: %
                    # Kontrola, zda je aktuální kurz blízko nebo pod nejbližší podporou
                    # Injekce signálu JSTRK
                    if (0 <= distance_to_support_pct <= self.distance_to_support_treshold.value) or (
                            current_rate < nearest_support):  # or self.jstrk_adjust:
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
                        candles = self.candles_before.value + (
                                self.candles_dca_multiplier.value * (count_of_buys - 1))  # Datový typ: int
                        # Kontrola, zda uplynul dostatečný čas od posledního nákupu
                        if time_since_last_buy < candles * candle_interval:
                            return None
                        # Kontrola, zda počet bezpečnostních příkazů (safety orders) není překročen
                        if self.max_safety_orders.value >= count_of_buys:
                            # Hledání posledního uzavřeného nákupního příkazu
                            last_buy_order = None
                            for order in reversed(trade.orders):
                                if order.ft_order_side == 'buy' and order.status == 'closed':
                                    last_buy_order = order
                                    break

                            # Výpočet procentní rozdílu mezi posledním nákupním příkazem a aktuálním kurzem
                            pct_diff = self.calculate_percentage_difference(original_price=last_buy_order.price,
                                                                            current_price=current_rate)  # Datový typ: float, jednotka: %
                            # Kontrola, zda je procentní rozdíl menší než prahová hodnota

                            if last_candle['five_min'] == 0:
                                return None

                            if pct_diff <= -self.pct_drop_treshold.value:
                                if last_buy_order and current_rate < last_buy_order.price:
                                    # Kontrola RSI podmínky pro DCA
                                    rsi_value = last_candle['rsi']  # Předpokládá se, že RSI je součástí dataframe
                                    w_rsi = last_candle[
                                        'weighted_rsi']  # Předpokládá se, že Weighted RSI je součástí dataframe

                                    if rsi_value <= w_rsi:
                                        # Extra kontrola JSTRK - nebo bez něj
                                        # if (not self.jstrk_adjust
                                        #         or (self.jstrk_adjust and last_candle['jstkr_2'] == 1)):
                                        # Logování informací o obchodu
                                        logging.info(
                                            f'AP1 {trade.pair}, Profit: {current_profit}, Stake {trade.stake_amount}')

                                        # Získání celkové částky sázky v peněžence
                                        total_stake_amount = self.wallets.get_total_stake_amount()
                                        # pokud bude signál ok, nemusíme tolik omezovat DCA
                                        # / self.dca_wallet_divider.value

                                        # Výpočet částky pro další sázku pomocí DCA (Dollar Cost Averaging)
                                        calculated_dca_stake = self.calculate_dca_price(
                                            base_value=trade.stake_amount,
                                            decline=current_profit * 100,
                                            target_percent=1)  # Datový typ: float
                                        # Upravení velikosti sázky, pokud je vyšší než dostupný zůstatek
                                        while calculated_dca_stake >= total_stake_amount:
                                            calculated_dca_stake = calculated_dca_stake / self.dca_order_divider.value  # Datový typ: float
                                        # Logování informací o upravené sázce
                                        logging.info(f'AP2 {trade.pair}, DCA: {calculated_dca_stake}')
                                        # Vrácení upravené velikosti sázky
                                        return calculated_dca_stake
            # Vrácení None, pokud nejsou splněny podmínky pro upravení obchodní pozice
            return None

    def calculate_percentage_difference(self, original_price, current_price):
        percentage_diff = ((current_price - original_price) / original_price)
        return percentage_diff

    def calculate_dca_price(self, base_value, decline, target_percent):
        return (((base_value / 100) * abs(decline)) / target_percent) * 100

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

    def prepare_doji(self, df):
        df = df.set_index(df.date)
        df = df.drop_duplicates(keep=False)
        df['date'] = pd.to_datetime(df['date']).tolist()
        df = df.iloc[:200]
        open = df['open']
        high = df['high']
        low = df['low']
        close = df['close']
        # TA-LIB Doji Scanner
        doji = talib.CDLDOJI(open, high, low, close)
        doji_star = talib.CDLDRAGONFLYDOJI(open, high, low, close)
        # Create a Doji Column, replace Doji points with with an candle close price
        df['doji'] = doji
        df['doji'] = np.where(df['doji'] > 0, close, 0)
        df['doji_star'] = doji_star
        df['doji_star'] = np.where(df['doji_star'] > 0, close, 0)
        # Create new Complete_Doji column, merges both columns together and finds maximas
        df['complete_doji'] = np.where(df['doji'] > 0, df['doji'], df['doji_star'])
        complete_doji = df['complete_doji']
        df['complete_doji_idx'] = np.where(df['complete_doji'] > 0, complete_doji.index, 0)
        # TA-LIB Trend line scanner
        trend_line = talib.HT_TRENDLINE(close)
        # Replace first 63 rows, 'NaN' values with 0s, create a new Trendline Column
        trend_line.fillna(0, inplace=True)
        df['trend_line'] = trend_line
        # TA-LIB RSI Scanner
        rsi = talib.RSI(close, timeperiod=7)
        # Fill 'NaN' values with 0s
        df['rsi'] = rsi.fillna(0)
        # Create new columns for RSI70 and RSI30 and fill with rsi values if Above/Below
        df['rsi70'] = np.where(df['rsi'] >= 70, rsi, 0)
        df['rsi30'] = np.where(df['rsi'] <= 30, rsi, 0)
        # Create Above/Below trend columns and insert Doji prices if Above/Below trend line
        df['above_trend'] = np.where(df['complete_doji'] > df['trend_line'], df.complete_doji, 0)
        df['below_trend'] = np.where(df['complete_doji'] < df['trend_line'], df.complete_doji, 0)
        df['five_max'] = np.where(df['above_trend'] > 0, 1, 0)
        df['five_min'] = np.where(df['below_trend'] > 0, 1, 0)
        pass


def EWO(dataframe, ema_length=5, ema2_length=3):
    df = dataframe.copy()
    ema1 = ta.EMA(df, timeperiod=ema_length)
    ema2 = ta.EMA(df, timeperiod=ema2_length)
    return (ema1 - ema2) / df['close'] * 100
