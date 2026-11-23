"""Candidate native Freqtrade integration. Live execution intentionally forbidden."""
from datetime import datetime
import math
import pandas as pd
import talib.abstract as ta
from freqtrade.strategy import IStrategy, stoploss_from_absolute
from trading_lab.perps.risk import Spec, size_order, worst_stop_loss_per_contract, assert_simulation

class BitgetBaseline(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = '1h'
    can_short = True
    startup_candle_count = 60
    process_only_new_candles = True
    minimal_roi = {}
    stoploss = -.10  # emergency margin-return cap; ATR stop callback tightens it
    use_custom_stoploss = True
    position_adjustment_enable = False
    trailing_stop = False
    order_types = {'entry':'market','exit':'market','emergency_exit':'market',
                   'force_entry':'market','force_exit':'market','stoploss':'market',
                   'stoploss_on_exchange':False}  # local simulation, NOT hosted protection
    protections = [{'method':'StoplossGuard','lookback_period_candles':24,
                    'trade_limit':2,'stop_duration_candles':24,'only_per_pair':False}]

    def __init__(self, config):
        assert_simulation(config)
        super().__init__(config)

    def bot_start(self, **kwargs):
        assert_simulation(self.config)
        mode=self.config.get('runmode')
        mode=getattr(mode,'value',mode)
        if mode not in ('dry_run','backtest'):
            raise ValueError('Only dry_run/backtest permitted; no live or optimization')

    def populate_indicators(self, dataframe, metadata):
        dataframe['ema20']=ta.EMA(dataframe,timeperiod=20)
        dataframe['ema50']=ta.EMA(dataframe,timeperiod=50)
        dataframe['atr']=ta.ATR(dataframe,timeperiod=14)
        return dataframe

    def populate_entry_trend(self, dataframe, metadata):
        long=(dataframe.ema20>dataframe.ema50)&(dataframe.ema20.shift(1)<=dataframe.ema50.shift(1))
        short=(dataframe.ema20<dataframe.ema50)&(dataframe.ema20.shift(1)>=dataframe.ema50.shift(1))
        dataframe.loc[long & (dataframe.volume>0),'enter_long']=1
        dataframe.loc[short & (dataframe.volume>0),'enter_short']=1
        return dataframe

    def populate_exit_trend(self, dataframe, metadata):
        dataframe.loc[dataframe.ema20<dataframe.ema50,'exit_long']=1
        dataframe.loc[dataframe.ema20>dataframe.ema50,'exit_short']=1
        return dataframe

    def leverage(self, pair, current_time, current_rate, proposed_leverage, max_leverage, entry_tag, side, **kwargs):
        return min(2.,max_leverage)  # virtual example, not an account recommendation

    def _closed(self,pair,when):
        df,_=self.dp.get_analyzed_dataframe(pair,self.timeframe)
        return df.loc[df.date+pd.Timedelta(hours=1)<=pd.Timestamp(when)]

    def custom_stake_amount(self,pair,current_time,current_rate,proposed_stake,min_stake,max_stake,
                            leverage,entry_tag,side,**kwargs):
        try:
            return self._risk_stake(pair,current_time,current_rate,min_stake,max_stake,leverage)
        except Exception:
            # Framework callback errors otherwise fall back to proposed stake. Fail closed.
            return 0.

    def _risk_stake(self,pair,current_time,current_rate,min_stake,max_stake,leverage):
        rows=self._closed(pair,current_time)
        if rows.empty or pd.Timestamp(current_time)-(rows.iloc[-1].date+pd.Timedelta(hours=1))>pd.Timedelta(hours=2):
            return 0.
        market=self.dp.market(pair)
        limits=market['limits']; step=market['precision']['amount']
        taker=float(market.get('taker') or .0006)
        contract_size=float(market.get('contractSize') or 1.)
        spec=Spec(pair,limits['amount']['min'] or step,step,limits['cost']['min'] or 5,
                  taker=taker,contract_size=contract_size)
        qty,_=size_order(self.wallets.get_total_stake_amount(),self.wallets.get_free('USDT'),current_rate,
                         2*rows.iloc[-1].atr,spec,leverage=leverage)
        stake=qty*spec.contract_size*current_rate/leverage
        return 0. if not qty or stake>max_stake or (min_stake and stake<min_stake) else stake

    def confirm_trade_entry(self,pair,order_type,amount,rate,time_in_force,current_time,entry_tag,side,**kwargs):
        try:
            rows=self._closed(pair,current_time)
            if rows.empty or pd.Timestamp(current_time)-(rows.iloc[-1].date+pd.Timedelta(hours=1))>pd.Timedelta(hours=2):
                return False
            market = self.dp.market(pair)
            contract_size = float(market.get('contractSize') or 1.)
            taker = float(market.get('taker') or .0006)
            spec = Spec(pair, 1., 1., 0., taker=taker, contract_size=contract_size)
            quantity, equity = float(amount), float(self.wallets.get_total_stake_amount())
            if not math.isfinite(quantity) or not math.isfinite(equity) or quantity<=0 or equity<=0:
                return False
            loss=quantity*worst_stop_loss_per_contract(float(rate),2*float(rows.iloc[-1].atr),spec,slippage=.0005)
            return bool(loss<=equity*.01+1e-9)
        except Exception:
            return False

    def custom_stoploss(self,pair,trade,current_time,current_rate,current_profit,after_fill,**kwargs):
        rows=self._closed(pair,trade.open_date_utc)
        if rows.empty: return None
        distance=2*float(rows.iloc[-1].atr)
        stop=trade.open_rate+distance if trade.is_short else trade.open_rate-distance
        return stoploss_from_absolute(stop,current_rate=current_rate,is_short=trade.is_short,leverage=trade.leverage)
