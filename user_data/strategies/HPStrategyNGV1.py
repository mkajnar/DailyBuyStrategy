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


class HPStrategyNGV1(IStrategy):
    timeframe = '1m'
    inf_1h = '1h'

    support_dict = {}
    resistance_dict = {}

    trade_limit = 4
    max_safety_orders = 3

    use_sell_signal = True
    sell_profit_only = True
    ignore_roi_if_buy_signal = False
    position_adjustment_enable = True
    process_only_new_candles = True
    startup_candle_count = 400

    order_time_in_force = {
        'buy': 'gtc',
        'sell': 'gtc'
    }
    plot_config = {
        'main_plot': {
            'ma_buy': {'color': 'orange'},
            'ma_sell': {'color': 'orange'},
        },
    }

    # SAR
    start = 0.02
    increment = 0.02
    maximum = 0.2

    trailing_stop = True
    trailing_stop_positive = 0.001
    trailing_stop_positive_offset = 0.005
    trailing_only_offset_is_reached = True
    stoploss = -0.99

    minimal_roi = {
        "0": 0.03,
        "30": 0.02,
        "60": 0.01,
        "90": 0.005,
        "120": 0
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

    low_offset = DecimalParameter(0.975, 0.995, default=buy_params['low_offset'], space='buy', optimize=True)
    high_offset = DecimalParameter(1.000, 1.010, default=sell_params['high_offset'], space='sell', optimize=True)
    high_offset_2 = DecimalParameter(1.000, 1.010, default=sell_params['high_offset_2'], space='sell', optimize=True)

    base_nb_candles_buy = IntParameter(8, 20, default=buy_params['base_nb_candles_buy'], space='buy', optimize=False)
    base_nb_candles_sell = IntParameter(8, 20, default=sell_params['base_nb_candles_sell'], space='sell',
                                        optimize=False)

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
        return "HPStrategyNGV1"

    def pivot_points(self, high, low, period=10):
        pivot_high = high.rolling(window=2 * period + 1, center=True).max()
        pivot_low = low.rolling(window=2 * period + 1, center=True).min()
        return high == pivot_high, low == pivot_low

    def calculate_support_resistance(self, df, period=10, loopback=290):
        high_pivot, low_pivot = self.pivot_points(df['high'], df['low'], period)
        df['resistance'] = df['high'][high_pivot]
        df['support'] = df['low'][low_pivot]
        return df

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

    def calculate_support_resistance_dicts(self, pair: str, df: DataFrame):
        try:
            df = self.calculate_support_resistance(df)
            self.support_dict[pair] = self.calculate_dynamic_clusters(df['support'].dropna().tolist(), 4)
            self.resistance_dict[pair] = self.calculate_dynamic_clusters(df['resistance'].dropna().tolist(), 4)
        except Exception as ex:
            logging.error(str(ex))

    def custom_sell(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                    current_profit: float, **kwargs):
        if current_profit < -0.05 and (current_time - trade.open_date_utc).days >= 7:
            return 'unclog'

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

    def calculate_percentage_difference(self, original_price, current_price):
        percentage_diff = ((current_price - original_price) / original_price)
        return percentage_diff

    def calculate_dca_price(self, base_value, decline, target_percent):
        return (((base_value / 100) * abs(decline)) / target_percent) * 100

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

    def info_tf_btc_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['rsi_8'] = ta.RSI(dataframe, timeperiod=8)
        ignore_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
        dataframe.rename(columns=lambda s: f"btc_{s}" if s not in ignore_columns else s, inplace=True)
        return dataframe

    def base_tf_btc_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['price_trend_long'] = (
                dataframe['close'].rolling(8).mean() / dataframe['close'].shift(8).rolling(144).mean())
        ignore_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
        dataframe.rename(columns=lambda s: f"btc_{s}" if s not in ignore_columns else s, inplace=True)
        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        # RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['rsi_fast'] = ta.RSI(dataframe, timeperiod=4)
        dataframe['rsi_slow'] = ta.RSI(dataframe, timeperiod=20)

        # SAR
        dataframe['sar'] = ta.SAR(dataframe, start=self.start, increment=self.increment, maximum=self.maximum)
        dataframe['sar_buy'] = (dataframe['sar'] < dataframe['low']).astype(int)
        dataframe['sar_sell'] = (dataframe['sar'] > dataframe['high']).astype(int)

        # EMA
        dataframe['ema_5'] = ta.EMA(dataframe, timeperiod=5)
        dataframe['ema_8'] = ta.EMA(dataframe, timeperiod=8)
        dataframe['ema_14'] = ta.EMA(dataframe, timeperiod=14)

        dataframe['ema_5'] = ta.EMA(dataframe['close'], timeperiod=5)

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

        # HMA
        dataframe['hma_50'] = qtpylib.hull_moving_average(dataframe['close'], window=50)

        # Vytvoření vah pro vážený průměr
        weights = np.linspace(1, 0, 300)  # Váhy od 1 (nejnovější) do 0 (nejstarší)
        weights /= weights.sum()  # Normalizace vah tak, aby jejich součet byl 1

        # Výpočet váženého průměru RSI pro posledních 300 svící
        dataframe['weighted_rsi'] = dataframe['rsi'].rolling(window=300).apply(
            lambda x: np.sum(weights * x[-300:]), raw=False
        )

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

        # Detekce bullish svíčkových vzorů
        dataframe['bullish_engulfing'] = ta.CDLENGULFING(dataframe['open'], dataframe['high'], dataframe['low'],
                                                         dataframe['close']) > 0
        dataframe['hammer'] = ta.CDLHAMMER(dataframe['open'], dataframe['high'], dataframe['low'],
                                           dataframe['close']) > 0
        # MACD
        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']

        dataframe['rebuy_signal'] = ((dataframe['ema_diff_buy_signal'].astype(int) > 0)
                                     & (dataframe['sar_sell'] == 1)).astype(int)

        # Přidání signálu 'jstkr'
        # Vytváří 1, když je součet 'macd' a 'macd_signal' záporný a 'rsi' <= 30
        dataframe['jstkr'] = ((dataframe['macd'] + dataframe['macdsignal'] < -0.01) & (dataframe['rsi'] <= 17)).astype(
            int)
        dataframe['jstkr_2'] = ((abs(dataframe['macd'] - dataframe['macdsignal']) / dataframe['macd'].abs() > 0.2) & (
                dataframe['rsi'] <= 25)).astype('int')
        dataframe['jstkr_3'] = ((abs(dataframe['macd'] - dataframe['macdsignal']) / dataframe['macd'].abs() > 0.04) & (
                dataframe['rsi_fast'] <= 10)).astype('int')

        # přidání výchozích buy/sell sloupců kvůli funkcionalitě ostatních metod
        if 'sell' not in dataframe.columns:
            dataframe['sell'] = 0
        if 'sell_tag' not in dataframe.columns:
            dataframe['sell_tag'] = ''
        if 'buy' not in dataframe.columns:
            dataframe['buy'] = 0
        if 'buy_tag' not in dataframe.columns:
            dataframe['buy_tag'] = ''
        self.calculate_support_resistance_dicts(metadata['pair'], dataframe)
        return dataframe

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                            time_in_force: str, current_time: datetime, entry_tag: Optional[str],
                            side: str, **kwargs) -> bool:
        try:
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            df = dataframe.copy()
        except Exception as e:
            logging.error(f"Error getting analyzed dataframe: {e}")
            return None

        # Získání aktuální svíčky
        last_candle = df.iloc[-1].squeeze()
        #
        # # Podmínky pro potvrzení nákupního signálu
        # cond_candles = self.confirm_by_candles(last_candle)
        # cond_sar = self.confirm_by_sar(last_candle)
        #
        # # Příprava výsledku
        # result = ((Trade.get_open_trade_count() < self.trade_limit) and (cond_candles or cond_sar))

        # Příprava výsledku
        result = ((Trade.get_open_trade_count() < self.trade_limit)
                  & (last_candle['ema_diff_buy_signal'] == 1)
                  & (last_candle['sar_sell'] == 1))
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

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

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
                dataframe['distance_to_support_pct'] = (dataframe['nearest_support'] - dataframe['close']) / dataframe[
                    'close'] * 100
                dataframe['distance_to_resistance_pct'] = (dataframe['nearest_resistance'] - dataframe['close']) / \
                                                          dataframe['close'] * 100

                # Vygenerování nákupních signálů na základě supportu a resistence
                buy_threshold = 0.1  # 0.1 %
                dataframe.loc[
                    (dataframe['distance_to_support_pct'] >= 0) &
                    (dataframe['distance_to_support_pct'] <= buy_threshold) &
                    (dataframe['distance_to_resistance_pct'] >= buy_threshold),
                    'buy_signal'
                ] = 1

        # SAR and PATTERNS
        cond_sar = self.confirm_by_sar(dataframe)
        cond_candles = self.confirm_by_candles(dataframe)
        # conditions = ((dataframe['buy_signal'] == 1)
        #               & (dataframe['volume'] > 0)
        #               & (dataframe['ema_diff_buy_signal'].astype(int) > 0)
        #               & (dataframe['sar_sell'] == 1))
        conditions = ((dataframe['buy_signal'] == 1)
                      & (dataframe['volume'] > 0))

        # Generování nákupních signálů
        # dataframe.loc[((conditions & cond_candles) | cond_sar), 'buy'] = 1
        dataframe.loc[conditions, 'buy_tag'] += 'sr_buy_'
        dataframe.loc[conditions, 'buy'] = 1

        # Odebrání pomocných sloupců
        for c in ['nearest_support', 'nearest_resistance', 'distance_to_support_pct', 'distance_to_resistance_pct']:
            if c in dataframe.columns:
                dataframe.drop([c], axis=1, inplace=True)

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if conditions := [
            (dataframe['close'] > dataframe['hma_50'])
            & (dataframe['close'] > (
                    dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset_2.value))
            & (dataframe['rsi'] > 50)
            & (dataframe['volume'] > 0)
            & (dataframe['rsi_fast'] > dataframe['rsi_slow'])
            | (dataframe['close'] < dataframe['hma_50'])
            & (dataframe['close'] > (dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset.value))
            & (dataframe['volume'] > 0)
            & (dataframe['rsi_fast'] > dataframe['rsi_slow'])
        ]:
            dataframe.loc[reduce(lambda x, y: x | y, conditions), 'sell'] = 1
        return dataframe

    def confirm_by_sar(self, data_dict):
        """ Based on TA indicators, populates the buy signal for the given dataframe """
        cond = (data_dict['sar_buy'] > 0)
        return cond

    def confirm_by_candles(self, data_dict):
        """ Based on TA indicators, populates the buy signal for the given dataframe """
        cond = ((data_dict['rsi'] <= data_dict['weighted_rsi'])
                & (data_dict['close'] > data_dict['ema_5'])
                & (data_dict['bullish_engulfing'] | data_dict['hammer']))
        return cond

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

                if last_candle['rebuy_signal'] == 0:
                    return None

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
                                # Kontrola ceny, aby se nekupovalo za vyšší než poslední nákup
                                if last_buy_order and current_rate < last_buy_order.price:
                                    # Kontrola RSI podmínky pro DCA
                                    rsi_value = last_candle['rsi']  # Předpokládá se, že RSI je součástí dataframe
                                    w_rsi = last_candle['weighted_rsi']
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
