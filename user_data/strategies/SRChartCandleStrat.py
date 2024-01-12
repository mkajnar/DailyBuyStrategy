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


class Elliot:
    base_nb_candles_sell = 22
    base_nb_candles_buy = 12
    ewo_low = 10.289
    ewo_high = 3.001
    fast_ewo = 50
    slow_ewo = 200
    low_offset = 0.987
    rsi_buy = 58
    high_offset = 1.014
    buy_ema_cofi = 0.97
    buy_fastk = 20
    buy_fastd = 20
    buy_adx = 30
    buy_ewo_high = 3.55

    def use_hyperopts(self, base_nb_candles_sell, base_nb_candles_buy, low_offset, ewo_high, ewo_low, rsi_buy,
                      high_offset, buy_ema_cofi, buy_fastk, buy_fastd, buy_adx, buy_ewo_high):
        self.base_nb_candles_sell = base_nb_candles_sell
        self.base_nb_candles_buy = base_nb_candles_buy
        self.low_offset = low_offset
        self.ewo_high = ewo_high
        self.ewo_low = ewo_low
        self.rsi_buy = rsi_buy
        self.high_offset = high_offset
        self.buy_ema_cofi = buy_ema_cofi
        self.buy_fastk = buy_fastk
        self.buy_fastd = buy_fastd
        self.buy_adx = buy_adx
        self.buy_ewo_high = buy_ewo_high

    def populate_indicators(self, dataframe: DataFrame):
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['rsi_fast'] = ta.RSI(dataframe, timeperiod=4)
        dataframe['EWO'] = self.__EWO(dataframe, self.fast_ewo, self.slow_ewo)
        for val in self.base_nb_candles_buy.range:
            dataframe[f'ma_buy_{val}'] = ta.EMA(dataframe, timeperiod=val)
        for val in self.base_nb_candles_sell.range:
            dataframe[f'ma_sell_{val}'] = ta.EMA(dataframe, timeperiod=val)
        return dataframe

    def populate_entry_trend_v1(self, dataframe: DataFrame, conditions: list):
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
        return (dataframe, conditions)

    def populate_entry_trend_v2(self, dataframe: DataFrame, conditions: list):
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
        return (dataframe, conditions)

    def populate_entry_trend_cofi(self, dataframe: DataFrame, conditions: list):
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
        return (dataframe, conditions)

    def __EWO(self, dataframe: DataFrame, ema_length=5, ema2_length=3):
        df = dataframe.copy()
        ema1 = ta.EMA(df, timeperiod=ema_length)
        ema2 = ta.EMA(df, timeperiod=ema2_length)
        return (ema1 - ema2) / df['close'] * 100


class SRChartCandleStrat(IStrategy):
    INTERFACE_VERSION = 3
    max_safety_orders = 3
    lowest_prices = {}
    highest_prices = {}
    price_drop_percentage = {}
    last_dca = {}
    pairs_close_to_high = []
    support_dict = {}
    resistance_dict = {}
    out_open_trades_limit = 10
    stoploss = -0.2

    trailing_stop = True
    trailing_stop_positive = 0.003
    trailing_stop_positive_offset = 0.008
    trailing_only_offset_is_reached = False

    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = True
    position_adjustment_enable = True
    order_time_in_force = {
        'entry': 'gtc',
        'exit': 'gtc'
    }

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
    timeframed_drops = {
        '1m': -0.01,
        '5m': -0.05,
        '15m': -0.05,
        '30m': -0.075,
        '1h': -0.1
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

    elliot = Elliot()

    downtrend_max_candles = IntParameter(5, 30, default=10, space='buy', optimize=False)
    downtrend_pct_treshold = DecimalParameter(0.5, 5, default=0.75, space='buy', optimize=False)
    mfi_buy_treshold = IntParameter(1, 15, default=5, space='buy', optimize=True)
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

    @property
    def protections(self):
        return [
            {
                "method": "CooldownPeriod",
                "stop_duration_candles": 1
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
        return "SRChartCandleStrat v1.0"

    # custom_info = {}
    #
    # def bot_loop_start(self, current_time: datetime, **kwargs) -> None:
    #
    #     for pair in list(self.custom_info):
    #         if "unlock_me" in self.custom_info[pair]:
    #             message = f"Found reverse position signal - unlocking {pair}"
    #             self.dp.send_msg(message)
    #             self.unlock_pair(pair)
    #             del self.custom_info[pair]

    def detect_hammer_doji(self, dataframe: DataFrame):
        # Přidání logiky pro detekci Hammer a Doji svíček
        hammer = ta.CDLHAMMER(dataframe['open'], dataframe['high'], dataframe['low'], dataframe['close'])
        doji = ta.CDLDOJI(dataframe['open'], dataframe['high'], dataframe['low'], dataframe['close'])
        dataframe['is_hammer'] = (hammer > 0)
        dataframe['is_doji'] = (doji > 0)
        return dataframe

    def dynamic_stop_loss_take_profit(self, dataframe: DataFrame) -> DataFrame:
        atr = ta.ATR(dataframe, timeperiod=14)
        dataframe = self.detect_hammer_doji(dataframe)
        for i in range(1, len(dataframe)):
            row = dataframe.iloc[i]
            prev_row = dataframe.iloc[i - 1]
            if row['is_hammer'] or row['is_doji']:
                dataframe.at[i, 'stop_loss'] = prev_row['low'] - atr.iloc[i] * 0.5
                dataframe.at[i, 'take_profit'] = prev_row['high'] + atr.iloc[i] * 3
            else:
                dataframe.at[i, 'stop_loss'] = prev_row['low'] - atr.iloc[i] * 0.8
                dataframe.at[i, 'take_profit'] = prev_row['high'] + atr.iloc[i] * 2.5
        return dataframe

    def calculate_dca_price(self, base_value, decline, target_percent):
        return (((base_value / 100) * abs(decline)) / target_percent) * 100

    def calculate_release_percentage(self, pair, current_rate, open_rate, stake_amount, pct_threshold,
                                     release_percentage):
        price_loss = (open_rate - current_rate) / open_rate
        if price_loss > pct_threshold:
            amount_to_release = stake_amount * -release_percentage
            for_free = amount_to_release if abs(amount_to_release) <= stake_amount else -stake_amount
            logging.info(f"Adjusting trade - free {for_free} from {pair}")
            return for_free
        return 0

    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                              current_rate: float, current_profit: float, min_stake: float,
                              max_stake: float, **kwargs):

        if current_rate is None or self.is_pair_locked(pair=trade.pair):
            logging.info(f"Skipping adjust_trade_position for {trade.pair} - pair is locked")
            return None

        current_time = datetime.utcnow()
        try:
            # Get an analyzed dataframe for the specified trading pair and timeframe
            dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
            df = dataframe.copy()
        except Exception as e:
            # Log an error if there's an issue retrieving the dataframe and exit the method
            logging.error(f"Error getting analyzed dataframe: {e}")
            return None

        last_candle = df.iloc[-1]

        if last_candle['trend'] == 'downtrend':
            total_stake_amount = self.wallets.get_total_stake_amount()
            pct_threshold = last_candle['max_drawdown']

            if total_stake_amount <= 0:
                return self.calculate_release_percentage(pair=trade.pair, current_rate=current_rate,
                                                         open_rate=trade.open_rate,
                                                         stake_amount=trade.stake_amount, pct_threshold=pct_threshold,
                                                         release_percentage=0.15)
            return None

        count_of_buys = sum(order.ft_order_side == 'buy' and order.status == 'closed' for order in trade.orders)
        if count_of_buys <= self.max_safety_orders:
            last_buy_order = None
            for order in reversed(trade.orders):
                if order.ft_order_side == 'buy' and order.status == 'closed':
                    last_buy_order = order
                    break
            pct_diff = self.calculate_percentage_difference(original_price=last_buy_order.price,
                                                            current_price=current_rate)
            logging.info(f"Price Drop: {pct_diff}, pct_threshold: {pct_threshold}")

            if pct_diff <= pct_threshold:
                if last_buy_order and current_rate < last_buy_order.price:
                    # rsi_value = last_candle['rsi']
                    # w_rsi = last_candle['weighted_rsi']
                    # if rsi_value <= w_rsi:
                    logging.info(f'AP1 {trade.pair}, Profit: {current_profit}, Stake {trade.stake_amount}')
                    total_stake_amount = self.wallets.get_total_stake_amount()
                    calculated_dca_stake = self.calculate_dca_price(base_value=trade.stake_amount,
                                                                    decline=current_profit * 100,
                                                                    target_percent=1)
                    while calculated_dca_stake >= total_stake_amount:
                        calculated_dca_stake = calculated_dca_stake / 4
                    logging.info(f'AP2 {trade.pair}, DCA: {calculated_dca_stake}')
                    last_dca = self.last_dca.get(trade.pair, None)
                    if ((last_dca is None)
                            or (last_dca + timedelta(
                                minutes=self.timeframe_to_minutes(self.timeframe) * 60) <= datetime.now())):
                        self.last_dca[trade.pair] = datetime.now()
                        return calculated_dca_stake
                    else:
                        return 0
        return 0

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        self.prepare(dataframe=dataframe)
        self.dynamic_stop_loss_take_profit(dataframe=dataframe)
        self.prepare_adx(dataframe=dataframe)
        self.prepare_rsi(dataframe=dataframe)
        self.prepare_stochastic(dataframe=dataframe)
        self.prepare_ema_diff_buy_signal(dataframe=dataframe)
        self.prepare_sma(dataframe=dataframe)
        self.prepare_ewo(dataframe=dataframe)
        self.prepare_doji(dataframe=dataframe)
        self.prepare_fibs(dataframe=dataframe)
        self.prepare_mfi(dataframe=dataframe)

        self.calculate_support_resistance_dicts(metadata['pair'], dataframe)
        dataframe = self.elliot.populate_indicators(dataframe=dataframe)
        self.detect_trend(dataframe=dataframe)
        return dataframe
        pass

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []

        if dataframe is not None:

            last_candle = dataframe.iloc[-1]
            # if last_candle['doji_candle'] > 0 and last_candle['lock_pair'] > 0:
            #     logging.info(f"Doji detected on {metadata['pair']} with lock")
            #     if not self.is_pair_locked(pair=metadata['pair']):
            #         self.lock_pair(pair=metadata['pair'], until=datetime.now(timezone.utc) + timedelta(
            #             minutes=self.timeframe_to_minutes(self.timeframe) * 14), reason='DOJI_LOCK')
            #         return dataframe
            #
            # if last_candle['doji_candle'] > 0 and last_candle['unlock_pair'] > 0:
            #     logging.info(f"Doji detected on {metadata['pair']} with unlock")
            #     if self.is_pair_locked(pair=metadata['pair']):
            #         self.unlock_pair(pair=metadata['pair'])

            dataframe.loc[
                (dataframe['mfi_buy'].rolling(window=self.mfi_buy_treshold.value).min() > 0), 'enter_tag'] += 'mfi_buy_'
            dataframe.loc[
                (dataframe['mfi_buy'].rolling(window=self.mfi_buy_treshold.value).min() > 0), 'enter_long'] = 1

            # Elliot Waves
            # (dataframe, conditions) = self.elliot.populate_entry_trend_v1(dataframe, conditions)
            (dataframe, conditions) = self.elliot.populate_entry_trend_v2(dataframe, conditions)
            (dataframe, conditions) = self.elliot.populate_entry_trend_cofi(dataframe, conditions)

            # Checking the distance to the nearest resistance
            # dataframe = self.populate_entry_trend_sr(dataframe=dataframe, metadata=metadata)

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
                        dataframe['distance_to_support_pct'] = (dataframe['nearest_support'] - dataframe['close']) / \
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
                (dataframe['volume'] > 0) & (
                        dataframe['ema_diff_buy_signal'].astype(int) > 0), 'enter_tag'] += 'ema_dbs_'

            # Generating buy signals only if both conditions are met
            dataframe.loc[(dataframe['buy_support'] == 1) & (dataframe['buy_ema'] == 1) & (
                    dataframe['rsi'] <= dataframe['weighted_rsi']), 'enter_long'] = 1

            # Remove helper columns
            if 'buy_support' in dataframe.columns:
                dataframe.drop(['buy_support'], axis=1, inplace=True)
            if 'buy_ema' in dataframe.columns:
                dataframe.drop(['buy_ema'], axis=1, inplace=True)

            dont_buy_conditions = [
                (dataframe['enter_long'].shift(1) == 1 & (dataframe['sma_2'].shift(1) < dataframe['sma_2'])) |
                (dataframe['trend'] == 'downtrend')
            ]

            if conditions:
                final_condition = reduce(lambda x, y: x | y, conditions)
                dataframe.loc[final_condition, 'enter_long'] = 1
            if dont_buy_conditions:
                for condition in dont_buy_conditions:
                    dataframe.loc[condition, 'enter_long'] = 0

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
            (dataframe['mfi_sell'].rolling(window=self.mfi_buy_treshold.value).min() > 0), 'exit_tag'] += 'mfi_sell_'
        dataframe.loc[(dataframe['mfi_sell'].rolling(window=self.mfi_buy_treshold.value).min() > 0), 'exit_long'] = 1

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

        dataframe.loc[:, 'exit_short'] = 0
        return dataframe

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                            time_in_force: str, current_time: datetime, entry_tag: Optional[str],
                            side: str, **kwargs) -> bool:
        # try:
        #     dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        #     df = dataframe.copy()
        # except Exception as e:
        #     logging.error(f"Error getting analyzed dataframe: {e}")
        #     return None
        #
        # last_candle = df.iloc[-1]
        # if last_candle['trend'] == 'downtrend':
        #     logging.info(f"Skipping confirm_trade_entry for {pair} - price is still in downtrend")
        #     return False

        logging.info(
            f"confirm_trade_entry: {pair}, {order_type}, {amount}, {rate}, {time_in_force}, {current_time}, {entry_tag}, {side}, {kwargs}")
        # t = Trade.get_open_trade_count() <= self.out_open_trades_limit
        # if t:
        #     if not self.is_pair_locked(pair=pair):
        #         self.lock_pair(pair=pair, reason='ENTRY_COOLDOWN',
        #                        until=datetime.now(timezone.utc) + timedelta(
        #                            minutes=self.timeframe_to_minutes(self.timeframe) * 1))
        #     logging.info(
        #         f"Locked pair {pair} after buying until {datetime.now(timezone.utc) + timedelta(minutes=self.timeframe_to_minutes(self.timeframe) * 5)}")
        return Trade.get_open_trade_count() <= self.out_open_trades_limit

    def custom_exit(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                    current_profit: float, **kwargs):
        if current_profit < -0.15 and (current_time - trade.open_date_utc).days >= 60:
            return 'unclog'

    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                           rate: float, time_in_force: str, exit_reason: str,
                           current_time: datetime, **kwargs) -> bool:
        exit_reason = f"{exit_reason}_{trade.enter_tag}"

        # dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        # last_candle = dataframe.iloc[-1].squeeze()
        # if trade.is_short:
        #     if last_candle['enter_long'] == 1:
        #         if not pair in self.custom_info:
        #             self.custom_info[pair] = {}
        #         self.custom_info[pair]["unlock_me"] = True
        # else:
        #     if last_candle['enter_short'] == 1:
        #         if not pair in self.custom_info:
        #             self.custom_info[pair] = {}
        #         self.custom_info[pair]["unlock_me"] = True

        if 'unclog' in exit_reason or 'force' in exit_reason or 'stoploss' == exit_reason or 'stop-loss' == exit_reason:
            # logging.info(f"CTE - FORCE or UNCLOG, EXIT")
            return True

        current_profit = trade.calc_profit_ratio(rate)
        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)

        if current_profit >= 0.005 and 'psl' in exit_reason:
            logging.info(f"CTE - PSL EXIT: {pair}, {current_profit}, {rate}, {exit_reason}, {amount}")
            return True

        # Checking if the current high is higher than the open of the last candle
        last_candle = dataframe.iloc[-1]

        if last_candle['high'] > last_candle['open']:
            # logging.info(f"CTE - Cena stále roste (high > open), HOLD")
            return False

        if current_profit <= 0.005:
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

        # if current_profit >= 0.0025:
        if ema_8_current <= ema_14_current and diff_change_pct >= 0.025:
            # logging.info(f"CTE - EMA 8 {ema_8_current} <= EMA 14 {ema_14_current} with decrease in difference >= 3%, EXIT")
            return True
        elif ema_8_current > ema_14_current and diff_current > diff_previous:
            # logging.info(f"CTE - EMA 8 {ema_8_current} > EMA 14 {ema_14_current} with increasing difference, HOLD")
            return False
        else:
            # logging.info(f"CTE - Conditions not met, EXIT")
            return True
        # else:
        #     return False

    pass

    def prepare(self, dataframe: DataFrame):
        if 'enter_tag' not in dataframe.columns:
            dataframe.loc[:, 'enter_tag'] = ''
        if 'exit_tag' not in dataframe.columns:
            dataframe.loc[:, 'exit_tag'] = ''
        if 'enter_long' not in dataframe.columns:
            dataframe.loc[:, 'enter_long'] = 0
        if 'exit_long' not in dataframe.columns:
            dataframe.loc[:, 'exit_long'] = 0
        pass

    def prepare_rsi(self, dataframe):
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['rsi_fast'] = ta.RSI(dataframe, timeperiod=4)
        dataframe['rsi_slow'] = ta.RSI(dataframe, timeperiod=20)
        # Calculation of weighted average RSI for the last 300 candles
        weights = np.linspace(1, 0, 300)  # Weights from 1 (newest) to 0 (oldest)
        weights /= weights.sum()  # Normalizing the weights so that their sum is 1
        dataframe['weighted_rsi'] = dataframe['rsi'].rolling(window=300).apply(
            lambda x: np.sum(weights * x[-300:]), raw=False
        )
        pass

    def prepare_stochastic(self, dataframe):
        stoch_fast = ta.STOCHF(dataframe, 5, 3, 0, 3, 0)
        dataframe['fastd'] = stoch_fast['fastd']
        dataframe['fastk'] = stoch_fast['fastk']
        pass

    def prepare_ema_diff_buy_signal(self, dataframe):
        ema_8 = ta.EMA(dataframe, timeperiod=8)
        ema_14 = ta.EMA(dataframe, timeperiod=14)
        # dataframe['ema_21'] = ta.EMA(dataframe, timeperiod=21)
        # dataframe['ema_50'] = ta.EMA(dataframe, timeperiod=50)
        condition = ema_8 > ema_14
        percentage_difference = 100 * (ema_8 - ema_14).abs() / ema_14
        ema_pct_diff = percentage_difference.where(condition, -percentage_difference)
        prev_ema_pct_diff = ema_pct_diff.shift(1)
        crossover_up = (ema_8.shift(1) < ema_14.shift(1)) & (ema_8 > ema_14)
        close_to_crossover_up = (ema_8 < ema_14) & (ema_8.shift(1) < ema_14.shift(1)) & (ema_8 > ema_8.shift(1))
        ema_buy_signal = ((ema_pct_diff < 0) & (prev_ema_pct_diff < 0) & (ema_pct_diff.abs() < prev_ema_pct_diff.abs()))
        dataframe['ema_diff_buy_signal'] = (
                (ema_buy_signal | crossover_up | close_to_crossover_up) & (dataframe['rsi'] <= 55) & (
                dataframe['volume'] > 0))
        dataframe['ema_8'] = ema_8
        dataframe['ema_14'] = ema_14
        pass

    def prepare_sma(self, dataframe):
        dataframe['sma_2'] = ta.SMA(dataframe, timeperiod=2)
        dataframe['sma_9'] = ta.SMA(dataframe, timeperiod=9)
        dataframe['sma_50'] = ta.SMA(dataframe, timeperiod=50)
        pass

    def calculate_percentage_difference(self, original_price, current_price):
        percentage_diff = ((current_price - original_price) / original_price)
        return percentage_diff

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

    def prepare_ewo(self, dataframe):
        dataframe['EWO'] = self.__EWO(dataframe, Elliot.fast_ewo, Elliot.slow_ewo)
        pass

    def __EWO(self, dataframe: DataFrame, ema_length=5, ema2_length=3):
        df = dataframe.copy()
        ema1 = ta.EMA(df, timeperiod=ema_length)
        ema2 = ta.EMA(df, timeperiod=ema2_length)
        return (ema1 - ema2) / df['close'] * 100

    def prepare_doji(self, dataframe):
        dataframe['doji_candle'] = (
                CDLDOJI(dataframe['open'], dataframe['high'], dataframe['low'], dataframe['close']) > 0).astype(int)

        # DOJI Candle Logic
        dataframe['upper_shadow'] = dataframe['high'] - np.maximum(dataframe['open'], dataframe['close'])
        dataframe['lower_shadow'] = np.minimum(dataframe['open'], dataframe['close']) - dataframe['low']
        is_doji = dataframe['doji_candle'] == 1

        # Condition for longer lower shadow in DOJI
        condition_longer_lower_shadow = (dataframe['lower_shadow'] > dataframe['upper_shadow']) & is_doji

        # Condition for longer upper shadow in DOJI
        condition_longer_upper_shadow = (dataframe['upper_shadow'] > dataframe['lower_shadow']) & is_doji

        # Implementing the DOJI logic for locking/unlocking or allowing buying
        # dataframe.loc[condition_longer_upper_shadow, 'enter_tag'] += 'doji_longer_upper_'
        dataframe.loc[condition_longer_upper_shadow, 'lock_pair'] = 1  # This will mark the pair to be locked
        dataframe.loc[condition_longer_lower_shadow, 'unlock_pair'] = 1
        pass

    def calculate_support_resistance_dicts(self, pair: str, df: DataFrame):
        try:
            df = self.calculate_support_resistance(df)
            self.support_dict[pair] = self.calculate_dynamic_clusters(df['support'].dropna().tolist(), 4)
            self.resistance_dict[pair] = self.calculate_dynamic_clusters(df['resistance'].dropna().tolist(), 4)
        except Exception as ex:
            logging.error(str(ex))

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

    def prepare_fibs(self, dataframe):
        high_max = dataframe['high'].rolling(window=30).max()
        low_min = dataframe['low'].rolling(window=30).min()
        diff = high_max - low_min
        dataframe['fib_236'] = high_max - 0.236 * diff
        dataframe['fib_382'] = high_max - 0.382 * diff
        dataframe['fib_500'] = high_max - 0.500 * diff
        dataframe['fib_618'] = high_max - 0.618 * diff
        dataframe['fib_786'] = high_max - 0.786 * diff
        pass

    def prepare_adx(self, dataframe):
        dataframe['adx'] = ta.ADX(dataframe)
        pass

    def prepare_mfi(self, dataframe):
        mfi_period = 14
        mfi = MFI(dataframe['high'], dataframe['low'], dataframe['close'], dataframe['volume'], timeperiod=mfi_period)
        overbought_threshold = 80
        oversold_threshold = 20
        dataframe['mfi_buy'] = (mfi < oversold_threshold).astype(int)
        dataframe['mfi_sell'] = (mfi > overbought_threshold).astype(int)
        pass

    def detect_trend(self, dataframe):
        x = self.downtrend_max_candles.value
        t = self.downtrend_pct_treshold.value
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

        df = dataframe[-200:]
        cumulative_max = df['close'].cummax()
        drawdowns = (df['close'] - cumulative_max) / cumulative_max
        max_drawdown = drawdowns.min()
        if -max_drawdown > self.timeframed_drops[self.timeframe]:
            dataframe['max_drawdown'] = self.timeframed_drops[self.timeframe] * t
        else:
            dataframe['max_drawdown'] = -max_drawdown * t
