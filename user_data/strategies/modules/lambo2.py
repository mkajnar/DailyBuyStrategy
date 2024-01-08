from pandas import DataFrame
import talib.abstract as ta




class Lambo(): 
    lambo2_ema_14_factor = 0.981
    lambo2_rsi_14_limit = 39
    lambo2_rsi_4_limit = 44

    def use_hyperopts(self, lambo2_ema_14_factor, lambo2_rsi_14_limit, lambo2_rsi_4_limit):
        self.lambo2_ema_14_factor = lambo2_ema_14_factor
        self.lambo2_rsi_14_limit = lambo2_rsi_14_limit
        self.lambo2_rsi_4_limit = lambo2_rsi_4_limit
        
    def populate_indicators(self, dataframe: DataFrame):
        """
        Populates the given dataframe with additional indicators.

        Parameters:
        dataframe (DataFrame): The dataframe to populate with indicators.

        Returns:
        DataFrame: The dataframe with populated indicators.
        """
        dataframe['ema_14'] = ta.EMA(dataframe, timeperiod=14)
        dataframe['rsi_4'] = ta.RSI(dataframe, timeperiod=4)
        dataframe['rsi_14'] = ta.RSI(dataframe, timeperiod=14)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, conditions: list):
        lambo2 = (
                (dataframe['close'] < (dataframe['ema_14'] * self.lambo2_ema_14_factor.value)) &
                (dataframe['rsi_4'] < int(self.lambo2_rsi_4_limit.value)) &
                (dataframe['rsi_14'] < int(self.lambo2_rsi_14_limit.value)) &
                (dataframe['sma_2'].shift(1) < dataframe['sma_2'])
        )
        dataframe.loc[lambo2, 'enter_tag'] += 'lambo2_'
        conditions.append(lambo2)
        return (dataframe, conditions)