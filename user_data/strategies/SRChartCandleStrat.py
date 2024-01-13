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
        """
        Initializes the hyperparameters for the optimization process.

        Args:
            base_nb_candles_sell (int): The number of base candles to consider for selling strategy.
            base_nb_candles_buy (int): The number of base candles to consider for buying strategy.
            low_offset (float): The offset value for calculating the lower bound for buying.
            ewo_high (float): The upper bound value for the EWO (Elder's Force Index).
            ewo_low (float): The lower bound value for the EWO (Elder's Force Index).
            rsi_buy (float): The RSI (Relative Strength Index) value for the buying strategy.
            high_offset (float): The offset value for calculating the upper bound for selling.
            buy_ema_cofi (float): The EMA (Exponential Moving Average) value for the buying strategy.
            buy_fastk (float): The Fast %K value for the buying strategy.
            buy_fastd (float): The Fast %D value for the buying strategy.
            buy_adx (float): The ADX (Average Directional Index) value for the buying strategy.
            buy_ewo_high (float): The upper bound value for the EWO (Elder's Force Index) in the buying strategy.

        Returns:
            None
        """
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

    def populate_indicators(self, dataframe: DataFrame) -> DataFrame:
        """
        Populates various technical indicators in the given DataFrame.

        Args:
            dataframe (DataFrame): The DataFrame to populate with indicators.

        Returns:
            DataFrame: The modified DataFrame with the populated indicators.
        """
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['rsi_fast'] = ta.RSI(dataframe, timeperiod=4)
        dataframe['EWO'] = self.__EWO(dataframe, self.fast_ewo, self.slow_ewo)
        for val in self.base_nb_candles_buy.range:
            dataframe[f'ma_buy_{val}'] = ta.EMA(dataframe, timeperiod=val)
        for val in self.base_nb_candles_sell.range:
            dataframe[f'ma_sell_{val}'] = ta.EMA(dataframe, timeperiod=val)
        return dataframe

    def populate_entry_trend_v1(self, dataframe: DataFrame, conditions: list):
        """
        Populates the 'enter_tag' column of the given DataFrame with the tag 'buy1eworsi_' for rows that meet the following conditions:

        - The 'rsi_fast' value is less than 35.
        - The 'close' value is less than the moving average of the specified number of candles multiplied by the 'low_offset' value.
        - The 'EWO' value is greater than the specified 'ewo_high' value.
        - The 'rsi' value is less than the specified 'rsi_buy' value.
        - The 'volume' value is greater than 0.
        - The 'close' value is less than the moving average of the specified number of candles multiplied by the 'high_offset' value.

        Args:
            dataframe (DataFrame): The DataFrame to populate.
            conditions (list): The list of conditions to append the 'buy1ewo' condition to.

        Returns:
            Tuple[DataFrame, list]: A tuple containing the updated DataFrame and the updated list of conditions.
        """
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
        """
        Populate entry trend v2.

        Args:
            dataframe (DataFrame): The input DataFrame.
            conditions (list): The list of conditions.

        Returns:
            tuple: A tuple containing the updated DataFrame and the updated conditions.
        """
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
        """
        Populates the 'entry_trend_cofi' field in the given DataFrame based on the specified conditions.

        Args:
            dataframe (DataFrame): The DataFrame to be modified.
            conditions (list): The list of conditions to be checked.

        Returns:
            tuple: A tuple containing the modified DataFrame and the updated list of conditions.
        """
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
        """
        Calculates the EWO (Elder's Force Index) indicator for a given DataFrame.

        Args:
            dataframe (DataFrame): The input DataFrame containing the OHLC (Open, High, Low, Close) data.
            ema_length (int, optional): The length of the first Exponential Moving Average (EMA). Defaults to 5.
            ema2_length (int, optional): The length of the second Exponential Moving Average (EMA). Defaults to 3.

        Returns:
            Series: The EWO indicator values.
        """
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
    exit_profit_only = True
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
        '1h': -0.1,
        '2h': -0.1,
        '4h': -0.1,
        '8h': -0.1,
        '1d': -0.1
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

    def prepare_psl(self, dataframe: DataFrame) -> DataFrame:
        """
        Prepare the given DataFrame for PSL analysis.
        Args:
            dataframe (DataFrame): The input DataFrame containing the data.
        Returns:
            DataFrame: The prepared DataFrame with added columns for stop loss and take profit values.

        This code defines a method called prepare_psl that takes a DataFrame as input and prepares it for PSL analysis.
        It calculates the average true range (atr) using the ta.ATR function from a library called ta.
        It then iterates over each row of the dataframe and checks if the current row is a hammer or doji candlestick pattern.
        If it is, it calculates the stop loss and take profit values based on the previous row's low and high values and
        the calculated average true range. If it's not a hammer or doji, it calculates stop loss and take profit values using
        different multipliers for the average true range. Finally, it returns the modified dataframe with additional columns
        for stop loss and take profit values.
        """
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
        """
        Calculate the Dollar Cost Averaging (DCA) price.
        Args:
            base_value (float): The base value of the investment.
            decline (float): The percentage decline of the investment.
            target_percent (float): The target percentage of the investment.
        Returns:
            float: The DCA price.

        This code calculates the Dollar Cost Averaging (DCA) price based on the base value,
        the percentage decline, and the target percentage of the investment.
        It returns the calculated DCA price.
        """
        return (((base_value / 100) * abs(decline)) / target_percent) * 100

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
            logging.info(f"Adjusting trade - free {for_free} from {pair}")
            return for_free
        return 0

    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                              current_rate: float, current_profit: float, min_stake: float,
                              max_stake: float, **kwargs):
        """
        Adjusts the trade position based on the current market conditions.

        Args:
            trade (Trade): The trade object representing the current trade.
            current_time (datetime): The current time.
            current_rate (float): The current exchange rate.
            current_profit (float): The current profit.
            min_stake (float): The minimum stake amount.
            max_stake (float): The maximum stake amount.
            **kwargs: Additional keyword arguments.

        Returns:
            float: The calculated DCA stake amount if conditions are met, otherwise 0.

        This code snippet is a method called adjust_trade_position that adjusts the position of a trade based
        on current market conditions. It takes in various parameters such as the trade object, current time,
        exchange rate, profit, and stake amounts.
        The method first checks if the current rate is None or if the trading pair is locked. If either of these
        conditions is true, it logs a message and returns None. Next, it tries to retrieve an analyzed dataframe
        for the specified trading pair and timeframe. If there's an error retrieving the dataframe, it logs an
        error message and returns None. It then checks the last candle of the dataframe and calculates a percentage
        threshold based on the maximum drawdown.
        If the trend is a downtrend, it checks the total stake amount and if it's less than or equal to 0,
        it calculates a release percentage based on certain parameters and returns that value.
        Otherwise, it returns None.
        If the count of closed buy orders in the trade is less than or equal to a maximum safety order limit,
        it calculates the percentage difference between the current rate and the price of the last closed buy order.
        If the percentage difference is less than or equal to the percentage threshold, it performs additional checks
        and calculations to determine if it needs to adjust the trade position.
        If certain conditions are met, it calculates a DCA (dollar-cost averaging) stake amount and returns that value.
        Otherwise, it returns 0. If none of the above conditions are met, the method returns 0.
        """

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
        pct_threshold = last_candle['max_drawdown']

        if last_candle['trend'] == 'downtrend':
            total_stake_amount = self.wallets.get_total_stake_amount()

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
        """
        Populates various indicators in the given dataframe using the provided metadata.
        Args:
            dataframe (DataFrame): The dataframe to populate the indicators in.
            metadata (dict): The metadata containing information about the indicators.
        Returns:
            DataFrame: The dataframe with the populated indicators.
        """
        self.prepare(dataframe=dataframe)
        self.prepare_psl(dataframe=dataframe)
        self.prepare_adx(dataframe=dataframe)
        self.prepare_rsi(dataframe=dataframe)
        self.prepare_stochastic(dataframe=dataframe)
        self.prepare_ema_diff(dataframe=dataframe)
        self.prepare_sma(dataframe=dataframe)
        self.prepare_ewo(dataframe=dataframe)
        self.prepare_doji(dataframe=dataframe)
        self.prepare_fibs(dataframe=dataframe)
        self.prepare_mfi(dataframe=dataframe)
        self.prepare_support_resistance_dicts(metadata['pair'], dataframe)
        self.elliot.populate_indicators(dataframe=dataframe)
        self.detect_trend(dataframe=dataframe)
        self.get_max_drawdown(dataframe=dataframe)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Populates the entry trend in the given DataFrame based on various conditions.
        Args:
            dataframe (DataFrame): The DataFrame to populate the entry trend in.
            metadata (dict): Additional metadata for the DataFrame.
        Returns:
            DataFrame: The DataFrame with the entry trend populated.

        This code snippet defines a method called populate_entry_trend in a class.
        It takes a DataFrame and a metadata dictionary as input and returns a DataFrame with the entry trend populated.
        The method applies various conditions to the DataFrame to determine the entry trend.
        It calculates the rolling minimum of a column called 'mfi_buy' and updates the 'enter_tag' and 'enter_long'
        columns based on the condition.
        It also calls other methods to populate the entry trend and performs additional calculations based
        on the metadata and support levels.
        Finally, it applies conditions to determine the 'enter_long' column and removes helper columns.
        """

        conditions = []

        if dataframe is not None:
            dataframe.loc[
                (dataframe['mfi_buy'].rolling(window=self.mfi_buy_treshold.value).min() > 0), 'enter_tag'] += 'mfi_buy_'
            dataframe.loc[
                (dataframe['mfi_buy'].rolling(window=self.mfi_buy_treshold.value).min() > 0), 'enter_long'] = 1

            # Elliot Waves
            # (dataframe, conditions) = self.elliot.populate_entry_trend_v1(dataframe, conditions)
            (dataframe, conditions) = self.elliot.populate_entry_trend_v2(dataframe, conditions)
            (dataframe, conditions) = self.elliot.populate_entry_trend_cofi(dataframe, conditions)
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
                (dataframe['enter_long'].shift(1) == 1 & (dataframe['sma_2'].shift(1) < dataframe['sma_2']))
                # | (dataframe['trend'] == 'downtrend')
            ]

            if conditions:
                final_condition = reduce(lambda x, y: x | y, conditions)
                dataframe.loc[final_condition, 'enter_long'] = 1
            if dont_buy_conditions:
                for condition in dont_buy_conditions:
                    dataframe.loc[condition, 'enter_long'] = 0

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Populates the 'exit_tag', 'exit_long', and 'exit_short' columns in the given dataframe based on certain conditions.
        Args:
            dataframe (DataFrame): The input dataframe.
            metadata (dict): Additional metadata.
        Returns:
            DataFrame: The modified dataframe with the 'exit_tag', 'exit_long', and 'exit_short' columns populated.

        This code snippet defines a method called populate_exit_trend that takes in a DataFrame and a dictionary as input.
        It populates the 'exit_tag', 'exit_long', and 'exit_short' columns in the given DataFrame based on certain conditions.
        The method then returns the modified DataFrame.
        """

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
        """
        Confirms a trade entry based on various conditions.

        Args:
            pair (str): The trading pair.
            order_type (str): The type of order.
            amount (float): The amount of the trade.
            rate (float): The rate of the trade.
            time_in_force (str): The time in force for the order.
            current_time (datetime): The current time.
            entry_tag (Optional[str]): An optional tag for the entry.
            side (str): The side of the trade.
            **kwargs: Additional keyword arguments.

        Returns:
            bool: True if the trade entry is confirmed, False otherwise.
        """
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

        return Trade.get_open_trade_count() <= self.out_open_trades_limit

    def custom_exit(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                    current_profit: float, **kwargs):
        """
        A function that handles custom exit logic for a trade.

        :param pair: A string representing the trading pair.
        :param trade: An instance of the Trade class representing the trade.
        :param current_time: A datetime object representing the current time.
        :param current_rate: A float representing the current rate.
        :param current_profit: A float representing the current profit.
        :param **kwargs: Additional keyword arguments.

        :return: A string indicating whether to 'unclog' the trade or not.

        This code defines a function called custom_exit that handles custom exit logic for a trade.
        It takes several parameters, including pair, trade, current_time, current_rate, current_profit,
        and **kwargs. The function checks if the current_profit is less than -0.15 and if the number
        of days between the current_time and the trade.open_date_utc is greater than or equal to 60.
        If both conditions are true, the function returns the string 'unclog'.
        """
        if current_profit < -0.15 and (current_time - trade.open_date_utc).days >= 60:
            return 'unclog'

    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                           rate: float, time_in_force: str, exit_reason: str,
                           current_time: datetime, **kwargs) -> bool:
        """
        Confirms the exit of a trade.

        Args:
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
            bool: True if the trade exit is confirmed, False otherwise.
        This code snippet is a method called confirm_trade_exit that determines whether
        a trade should be exited or not based on various conditions. It takes in several
        parameters such as the trading pair, trade object, order type, amount, rate,
        time in force, exit reason, current time, and additional keyword arguments.
        It returns a boolean value indicating whether the trade exit is confirmed or not.
        The method checks for specific exit reasons and if they match certain conditions,
        it returns True. It also performs calculations on the current profit ratio and
        checks for specific conditions related to the analyzed data.
        If none of the conditions are met, it returns True as a default.
        """
        exit_reason = f"{exit_reason}_{trade.enter_tag}"

        if 'unclog' in exit_reason or 'force' in exit_reason or 'stoploss' == exit_reason or 'stop-loss' == exit_reason:
            return True

        current_profit = trade.calc_profit_ratio(rate)

        if self.exit_profit_only:
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
                logging.info(f"CTE - EMA 8 {ema_8_current} > EMA 14 {ema_14_current} with increasing difference, HOLD")
                return False
            else:
                return True
        else:
            logging.info(f"CTE - EXIT: {pair}, {current_profit}, {rate}, {exit_reason}, {amount}")
            return True

    def prepare(self, dataframe: DataFrame):
        """
        Prepare the given DataFrame by adding missing columns if necessary.
        Args:
            dataframe (DataFrame): The DataFrame to be prepared.
        Returns:
            None
        """
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
        """
        Calculates the Relative Strength Index (RSI) and weighted average RSI for a given dataframe.
        Args:
            dataframe (pandas.DataFrame): The input dataframe containing the necessary data.
        Returns:
            None
        """
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
        """
        Prepare the stochastic indicator for a given dataframe.
        Args:
            dataframe (pandas.DataFrame): The input dataframe.
        Returns:
            None
        """
        stoch_fast = ta.STOCHF(dataframe, 5, 3, 0, 3, 0)
        dataframe['fastd'] = stoch_fast['fastd']
        dataframe['fastk'] = stoch_fast['fastk']
        pass

    def prepare_ema_diff(self, dataframe):
        """
        Prepare the exponential moving average (EMA) difference.
        This function calculates the EMA difference between two different time periods,
        specifically the EMA with a time period of 8 and the EMA with a time period of 14.
        Parameters:
        - dataframe: The input dataframe containing the necessary data.
        Returns:
        This function does not return anything.
        Example Usage:
        prepare_ema_diff(dataframe)
        """
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
        """
        Generate the simple moving averages (SMA) for the given dataframe.
        :param dataframe: The input dataframe.
        :type dataframe: pandas.DataFrame
        :return: None
        """
        dataframe['sma_2'] = ta.SMA(dataframe, timeperiod=2)
        dataframe['sma_9'] = ta.SMA(dataframe, timeperiod=9)
        dataframe['sma_50'] = ta.SMA(dataframe, timeperiod=50)
        pass

    def calculate_percentage_difference(self, original_price, current_price):
        """
        Calculate the percentage difference between the original price and the current price.
        Parameters:
            original_price (float): The original price.
            current_price (float): The current price.
        Returns:
            float: The percentage difference between the original price and the current price.
        """
        percentage_diff = ((current_price - original_price) / original_price)
        return percentage_diff

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

    def prepare_ewo(self, dataframe):
        """
        Generate the Exponential Weighted Oscillator (EWO) for a given dataframe.
        Args:
            dataframe (pandas.DataFrame): The input dataframe.
        Returns:
            None
        """
        dataframe['EWO'] = self.__EWO(dataframe, Elliot.fast_ewo, Elliot.slow_ewo)
        pass

    def __EWO(self, dataframe: DataFrame, ema_length=5, ema2_length=3):
        """
        Calculate the Exponential Weighted Oscillator (EWO) for a given DataFrame.

        Parameters:
            dataframe (DataFrame): The input DataFrame containing the OHLC data.
            ema_length (int, optional): The length of the first Exponential Moving Average (EMA). Default is 5.
            ema2_length (int, optional): The length of the second Exponential Moving Average (EMA). Default is 3.

        Returns:
            Series: The EWO values calculated based on the formula: (ema1 - ema2) / close * 100.
        """
        df = dataframe.copy()
        ema1 = ta.EMA(df, timeperiod=ema_length)
        ema2 = ta.EMA(df, timeperiod=ema2_length)
        return (ema1 - ema2) / df['close'] * 100

    def prepare_doji(self, dataframe):
        """
        This function prepares the dataframe for identifying the presence of a doji candle. It takes a dataframe as input and adds the following columns to the dataframe:

        - 'doji_candle': A binary column indicating whether a doji candle is present or not.
        - 'upper_shadow': The length of the upper shadow of the candle.
        - 'lower_shadow': The length of the lower shadow of the candle.

        The function then applies the following logic:

        - It checks if a candle is a doji candle by comparing the doji candle indicator with 1.
        - It checks if a candle has a longer lower shadow than the upper shadow and is a doji candle.
        - It checks if a candle has a longer upper shadow than the lower shadow and is a doji candle.

        Based on these conditions, the function performs the following actions:

        - If a candle has a longer upper shadow than the lower shadow and is a doji candle, it marks the pair to be locked.
        - If a candle has a longer lower shadow than the upper shadow and is a doji candle, it marks the pair to be unlocked.

        Note: The 'enter_tag' column is not modified in this function.

        Parameters:
        - dataframe (pandas.DataFrame): The input dataframe containing the candle data.

        Returns:
        None
        """
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

    def prepare_support_resistance_dicts(self, pair: str, df: DataFrame):
        """
        Generates the support and resistance dictionaries for a given pair.

        Parameters:
            pair (str): The currency pair.
            df (DataFrame): The input DataFrame.

        Returns:
            None

        Raises:
            Exception: If an error occurs during the calculation process.
        """
        try:
            df = self.calculate_support_resistance(df)
            self.support_dict[pair] = self.calculate_dynamic_clusters(df['support'].dropna().tolist(), 4)
            self.resistance_dict[pair] = self.calculate_dynamic_clusters(df['resistance'].dropna().tolist(), 4)
        except Exception as ex:
            logging.error(str(ex))

    def pivot_points(self, high, low, period=10):
        """
        Calculate the pivot points based on the given high and low values.

        Parameters:
            high (Series): The series of high values.
            low (Series): The series of low values.
            period (int, optional): The period used for the rolling calculations. Defaults to 10.

        Returns:
            tuple: A tuple with two boolean series. The first series indicates if the high values are equal to the
            pivot high values. The second series indicates if the low values are equal to the pivot low values.
        """
        pivot_high = high.rolling(window=2 * period + 1, center=True).max()
        pivot_low = low.rolling(window=2 * period + 1, center=True).min()
        return high == pivot_high, low == pivot_low

    def calculate_support_resistance(self, df, period=10, loopback=290):
        """
        Calculate support and resistance levels based on high and low values in a dataframe.

        Args:
            df (pandas.DataFrame): The input dataframe containing 'high' and 'low' columns.
            period (int, optional): The period used for calculating the pivot points. Defaults to 10.
            loopback (int, optional): The number of previous rows to consider for calculating the pivot points. Defaults to 290.

        Returns:
            pandas.DataFrame: The input dataframe with additional 'resistance' and 'support' columns.
        """
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
            """
            A function to cluster values based on a threshold.

            :param threshold: The maximum difference between values for them to be considered part of the same cluster.
            :type threshold: int or float

            :return: A list of clusters where each cluster is a list of values.
            :rtype: list[list]
            """
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
        """
        Calculate Fibonacci retracement levels for the given dataframe.

        Args:
            dataframe (pandas.DataFrame): The input dataframe containing the 'high'
                and 'low' columns.

        Returns:
            None

        Side Effects:
            Modifies the input dataframe by adding new columns for each Fibonacci
            retracement level.

        """
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
        """
        Prepares the adx column for the given dataframe.

        Parameters:
            dataframe (pandas.DataFrame): The input dataframe.

        Returns:
            None
        """
        mfi_period = 14
        mfi = MFI(dataframe['high'], dataframe['low'], dataframe['close'], dataframe['volume'], timeperiod=mfi_period)
        overbought_threshold = 80
        oversold_threshold = 20
        dataframe['mfi_buy'] = (mfi < oversold_threshold).astype(int)
        dataframe['mfi_sell'] = (mfi > overbought_threshold).astype(int)
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
            # logging.error(str(ex))
            pass

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
            # logging.error(str(ex))
            pass


class STPyramideStrategy(SRChartCandleStrat):
    INTERFACE_VERSION = 3
    position_adjustment_enable = True
    trailing_stop = True
    trailing_stop_positive = 0.005
    trailing_stop_positive_offset = 0.01
    trailing_only_offset_is_reached = True
    ignore_roi_if_entry_signal = True
    exit_profit_only = True
    last_adjustment_time = {}
    order_openrates = {}

    # Definice parametrů strategie
    minimal_roi = {
        "0": 0.10
    }

    stoploss = -0.05

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = super().populate_indicators(dataframe, metadata)
        # EMA
        dataframe['ema_fast'] = ta.EMA(dataframe, timeperiod=5)
        dataframe['ema_slow'] = ta.EMA(dataframe, timeperiod=10)

        # RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = super().populate_entry_trend(dataframe, metadata)
        dataframe.loc[
            (
                    (dataframe['ema_fast'] > dataframe['ema_slow']) &  # EMA Fast je nad EMA Slow
                    (dataframe['rsi'] < 70)  # RSI je pod 70
            ),
            'enter_long'] = 1
        return dataframe

    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                              current_rate: float, current_profit: float, min_stake: float,
                              max_stake: float, **kwargs):
        # Získání analyzovaných dat
        self.order_openrates[trade.pair] = trade.open_rate
        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        last_candle = dataframe.iloc[-1]

        if current_time.tzinfo is not None and current_time.tzinfo.utcoffset(current_time) is not None:
            current_time = current_time.replace(tzinfo=None)

        # Logika pro zabránění opakovaných úprav v stejném časovém segmentu
        last_adjustment = self.last_adjustment_time.get(trade.pair, datetime.min)
        if current_time - last_adjustment < timedelta(minutes=self.timeframe_to_minutes(self.timeframe)):
            return None  # Žádná úprava, pokud jsme již v daném segmentu provedli úpravu

        rsi = last_candle['rsi']
        if rsi < 70:
            decrease_percentage = 0.05  # Určení prahové hodnoty pro snížení
            increase_percentage = 0.1  # Určení prahové hodnoty pro zvýšení

            if (current_rate - last_candle['close']) / current_rate >= decrease_percentage:
                new_amount = min(trade.amount, max_stake)
                logging.info(f"Adjusting position for {trade.pair} profit to {new_amount}")
                self.last_adjustment_time[trade.pair] = current_time
                return new_amount

            if (last_candle['close'] - current_rate) / current_rate >= increase_percentage:
                new_amount = min(trade.amount, max_stake)
                logging.info(f"Adjusting position for {trade.pair} profit to {new_amount}")
                self.last_adjustment_time[trade.pair] = current_time
                return new_amount

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                            time_in_force: str, current_time: datetime, entry_tag: Optional[str],
                            side: str, **kwargs) -> bool:
        confirm = True
        open_rate = self.order_openrates.get(pair, None)
        if open_rate is not None:
            confirm = open_rate < rate
        if confirm:
            logging.info(
                f"Confirmed trade for {pair}, {order_type}, {amount}, {rate}, {time_in_force}, {current_time}, {entry_tag}, {side}, {kwargs}")
        return confirm

    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                           rate: float, time_in_force: str, exit_reason: str,
                           current_time: datetime, **kwargs) -> bool:

        exit_reason = f"{exit_reason}_{trade.enter_tag}"

        if ('unclog' in exit_reason or
                'force' in exit_reason or
                'stoploss' == exit_reason or
                'stop-loss' == exit_reason or
                'psl takeprofit' == exit_reason or
                'psl stoploss' == exit_reason):
            return True

        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        last_candle = dataframe.iloc[-1]
        current_profit = trade.calc_profit_ratio(rate)

        if last_candle['ema_fast'] > last_candle['ema_slow']:
            return False
        t = True if current_profit > 0.001 else False
        if t:
            logging.info(f"CTE - Profit > 0.001, EXIT")
        return t

    def custom_exit(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                    current_profit: float, **kwargs) -> bool:
        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        last_candle = dataframe.iloc[-1]
        current_profit = trade.calc_profit_ratio(current_rate)

        if last_candle['high'] > last_candle['open'] and (current_profit > 0.001):
            logging.info(f"CTE - Cena stále roste (high > open), HOLD")
            return False
        return True

    def prepare_psl(self, dataframe: DataFrame) -> DataFrame:
        dataframe = super().prepare_psl(dataframe=dataframe)
        atr = ta.ATR(dataframe, timeperiod=3)
        dataframe = self.detect_hammer_doji(dataframe)
        for i in range(1, len(dataframe)):
            row = dataframe.iloc[i]
            prev_row = dataframe.iloc[i - 1]
            if row['is_hammer'] or row['is_doji']:
                dataframe.at[i, 'stop_loss'] = prev_row['low'] - atr.iloc[i] * 0.25
                dataframe.at[i, 'take_profit'] = prev_row['high'] + atr.iloc[i] * 1.5
            else:
                dataframe.at[i, 'stop_loss'] = prev_row['low'] - atr.iloc[i] * 0.4
                dataframe.at[i, 'take_profit'] = prev_row['high'] + atr.iloc[i] * 1.25
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        profit_cond = (
                (dataframe['high'] > dataframe['take_profit']) |
                (dataframe['close'] > dataframe['take_profit']) |
                (dataframe['open'] > dataframe['take_profit'])
        )
        dataframe.loc[profit_cond, 'exit_tag'] = 'psl takeprofit'
        dataframe.loc[profit_cond, 'exit_long'] = 1

        stoploss_cond = (
                (dataframe['low'] < dataframe['stop_loss']) |
                (dataframe['close'] < dataframe['stop_loss']) |
                (dataframe['open'] < dataframe['stop_loss'])
        )
        dataframe.loc[stoploss_cond, 'exit_tag'] = 'psl stoploss'
        dataframe.loc[stoploss_cond, 'exit_long'] = 1

        return dataframe
