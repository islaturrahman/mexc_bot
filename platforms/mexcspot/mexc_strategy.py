import platforms.mexcspot.mexc_spot as mexc
import pandas as pd
import pandas_ta as ta
import numpy as np


import requests
from datetime import datetime, timedelta
import pytz
import time

from datetime import datetime, timedelta
from backtesting import Strategy

def RSI(array, period=14):
    """Hitung RSI indicator"""
    array = np.array(array)
    deltas = np.diff(array)
    seed = deltas[:period+1]
    up = seed[seed >= 0].sum() / period
    down = -seed[seed < 0].sum() / period
    rs = up / down if down != 0 else 0
    rsi = np.zeros_like(array)
    rsi[:period] = 100. - 100. / (1. + rs)

    for i in range(period, len(array)):
        delta = deltas[i - 1]
        if delta > 0:
            upval = delta
            downval = 0.
        else:
            upval = 0.
            downval = -delta

        up = (up * (period - 1) + upval) / period
        down = (down * (period - 1) + downval) / period
        rs = up / down if down != 0 else 0
        rsi[i] = 100. - 100. / (1. + rs)

    return rsi


class RSIStrategy(Strategy):
    # Parameter strategi yang bisa dioptimasi
    rsi_period = 14
    rsi_upper = 70  # Overbought
    rsi_lower = 30  # Oversold
    
    def init(self):
        # Inisialisasi indikator RSI
        self.rsi = self.I(RSI, self.data.Close, self.rsi_period)
        
    def next(self):
        # Skip jika RSI belum terhitung (nilai NaN di awal)
        if np.isnan(self.rsi[-1]):
            return
            
        # Logika trading RSI
        # Beli ketika RSI < 30 (oversold) dan belum ada posisi
        if self.rsi[-1] < self.rsi_lower:
            if not self.position:
                self.buy()
        
        # Jual ketika RSI > 70 (overbought) dan ada posisi
        elif self.rsi[-1] > self.rsi_upper:
            if self.position:
                self.position.close()


class RSIMeanReversionStrategy(Strategy):
    """
    Strategi RSI Mean Reversion yang lebih agresif
    """
    rsi_period = 14
    rsi_oversold = 30
    rsi_overbought = 70
    
    def init(self):
        self.rsi = self.I(RSI, self.data.Close, self.rsi_period)
        
    def next(self):
        if np.isnan(self.rsi[-1]):
            return
        
        # Buy pada oversold
        if self.rsi[-1] < self.rsi_oversold and not self.position:
            self.buy()
        
        # Sell pada overbought ATAU take profit ketika RSI kembali ke 50
        elif self.position:
            if self.rsi[-1] > self.rsi_overbought or self.rsi[-1] > 50:
                self.position.close()


class RSITrendFollowingStrategy(Strategy):
    """
    Strategi RSI Trend Following
    """
    rsi_period = 14
    rsi_bull = 50  # Batas bullish
    rsi_bear = 50  # Batas bearish
    
    def init(self):
        self.rsi = self.I(RSI, self.data.Close, self.rsi_period)
        
    def next(self):
        if np.isnan(self.rsi[-1]):
            return
        
        # Buy ketika RSI cross atas 50 (momentum bullish)
        if self.rsi[-1] > self.rsi_bull and self.rsi[-2] <= self.rsi_bull:
            if not self.position:
                self.buy()
        
        # Sell ketika RSI cross bawah 50 (momentum bearish)
        elif self.rsi[-1] < self.rsi_bear and self.rsi[-2] >= self.rsi_bear:
            if self.position:
                self.position.close()


class AlwaysInMarketStrategy(Strategy):
    """Strategi untuk memastikan data & koneksi backtest OK"""
    def init(self):
        print(f"[DEBUG] AlwaysInMarket initialized with {len(self.data)} candles")
    
    def next(self):
        # 1. Pastikan gunakan size=0.9 agar modal $89 cukup untuk beli fraksi BTC
        if not self.position:
            self.buy(size=0.9) 
            print(f"[TRADE] BUY at {self.data.index[-1]} | Price: {self.data.Close[-1]}")

        # 2. Opsional: Tutup di bar terakhir supaya muncul di statistik # Trades
        if len(self.data) == len(self.data.df) - 1:
            self.position.close()

class SimpleRSIStrategy(Strategy):
    rsi_period = 14
    rsi_lower = 35 # Sedikit lebih longgar agar lebih banyak trade
    rsi_upper = 65
    
    def init(self):
        self.rsi = self.I(RSI, self.data.Close, self.rsi_period)
        
    def next(self):
        if np.isnan(self.rsi[-1]):
            return
            
        # Logika BELI: RSI < 35 & Belum ada posisi
        # Gunakan size=0.9 (90% dari equity)
        if self.rsi[-1] < self.rsi_lower and not self.position:
            self.buy(size=0.9)
            print(f"DEBUG: BUY SIGNAL | RSI: {self.rsi[-1]:.2f} | Price: {self.data.Close[-1]}")
        
        # Logika JUAL: RSI > 65 & Ada posisi
        elif self.rsi[-1] > self.rsi_upper and self.position:
            self.position.close()
            print(f"DEBUG: SELL SIGNAL | RSI: {self.rsi[-1]:.2f} | Price: {self.data.Close[-1]}")




class MeanStrategySimbolic(Strategy):

    # ===================================================
    #                       RISK MANAGEMENT
    # ===================================================
    def market_regime_detection(self):
        
        return 0
    
    def sentiment_analysis(self):
        return 0
    
    def position_sizing(self):
        return 0
    
    def condition_performance_optimization(self):
        return 0
    
    def free_transaction_costs(self):
        fee_type = {
            'commission' : 0,
            'slippage' : 0,
            'spread' : 0,
            'liquidity cost' : 0,
            'opportunity Cost' : 0
        }
        return fee_type

    def sample_size(self):
        return 0
    
    def trailing_stop_loss(self):
        return 0
    
    def stop_loss(self):
        return 0
    
    def take_profit(self):
        return 0
    
    def drawdown_control(self):
        return 0
    

    
    def run(self):
        print("Running strategy...")

    


class SupertrendFootprintStrategy(Strategy):
    # --- Parameter Supertrend ---
    atr_period = 14
    atr_multiplier = 3.0
    
    # --- Parameter Footprint & Filter ---
    vol_threshold = 2.0    # Volume harus 2x rata-rata
    delta_strength = 2.0   # Kekuatan Delta
    conf_bars_req = 2      # Minimal bar konfirmasi delta
    ema_trend_len = 200    # Filter Trend Besar
    
    # --- Parameter Exit ---
    tp_atr_mult = 3.0      # Take Profit
    partial_tp_mult = 1.5  # Partial TP
    trail_atr_mult = 2.0   # Trailing Stop
    be_trigger_atr = 1.0   # Break Even Trigger

    def init(self):
        # 1. Hitung Supertrend menggunakan pandas_ta
        st = ta.supertrend(
            high=pd.Series(self.data.High), 
            low=pd.Series(self.data.Low), 
            close=pd.Series(self.data.Close), 
            period=self.atr_period, 
            multiplier=self.atr_multiplier
        )
        # SUPERT_d_14_3.0 (direction: 1 bullish, -1 bearish)
        self.st_dir = self.I(lambda: st.iloc[:, 1]) 
        
        # 2. Footprint / Delta Logic
        # Karena kita hanya punya OHLCV, delta disimulasikan:
        # Buy Vol jika Close > Open, else Sell Vol
        def calc_delta(close, open_p, vol):
            buy_vol = np.where(close > open_p, vol, 0)
            sell_vol = np.where(close < open_p, vol, 0)
            return buy_vol - sell_vol

        self.delta = self.I(calc_delta, self.data.Close, self.data.Open, self.data.Volume)
        self.delta_sma = self.I(ta.sma, pd.Series(self.delta), 10)
        self.avg_vol = self.I(ta.sma, pd.Series(self.data.Volume), 20)
        
        # 3. Trend & Momentum Filters
        self.ema_trend = self.I(ta.ema, pd.Series(self.data.Close), self.ema_trend_len)
        self.rsi = self.I(ta.rsi, pd.Series(self.data.Close), 14)
        self.atr = self.I(ta.atr, pd.Series(self.data.High), pd.Series(self.data.Low), pd.Series(self.data.Close), 14)

        # Variabel bantu untuk tracking
        self.long_be_active = False
        self.short_be_active = False
        self.conf_count = 0

    def next(self):
        price = self.data.Close[-1]
        atr_val = self.atr[-1]
        
        # --- HITUNG LOGIKA FOOTPRINT & VOLUME ---
        is_high_vol = self.data.Volume[-1] > (self.avg_vol[-1] * self.vol_threshold)
        is_strong_delta = abs(self.delta[-1]) > (abs(self.delta_sma[-1]) * self.delta_strength)
        
        # Hitung konfirmasi bar beruntun (Delta Confirmation)
        if (self.delta[-1] > 0 and self.st_dir[-1] == 1) or (self.delta[-1] < 0 and self.st_dir[-1] == -1):
            self.conf_count += 1
        else:
            self.conf_count = 0

        # --- LOGIKA ENTRY ---
        
        # LONG: Supertrend Up + Price > EMA 200 + RSI Bullish + High Vol + Delta Confirmed
        long_condition = (
            self.st_dir[-1] == 1 and 
            price > self.ema_trend[-1] and 
            50 < self.rsi[-1] < 70 and
            is_high_vol and 
            self.delta[-1] > 0 and 
            self.conf_count >= self.conf_bars_req
        )

        if long_condition and not self.position:
            # Set target awal
            self.entry_price = price
            self.buy(size=0.9) 
            self.long_be_active = False
            print(f"DEBUG: LONG ENTRY | Price: {price} | ATR: {atr_val:.2f}")

        # SHORT: Supertrend Down + Price < EMA 200 + RSI Bearish + High Vol + Delta Confirmed
        short_condition = (
            self.st_dir[-1] == -1 and 
            price < self.ema_trend[-1] and 
            30 < self.rsi[-1] < 50 and
            is_high_vol and 
            self.delta[-1] < 0 and 
            self.conf_count >= self.conf_bars_req
        )

        if short_condition and not self.position:
            self.entry_price = price
            self.sell(size=0.9)
            self.short_be_active = False
            print(f"DEBUG: SHORT ENTRY | Price: {price}")

        # --- MANAJEMEN POSISI (EXIT & TRAILING) ---
        
        if self.position.is_long:
            # 1. Break Even Logic
            if not self.long_be_active and price >= (self.entry_price + (atr_val * self.be_trigger_atr)):
                self.long_be_active = True
                print("DEBUG: Break Even Activated for Long")

            # 2. Exit Conditions
            # Profit Target
            if price >= (self.entry_price + (atr_val * self.tp_atr_mult)):
                self.position.close()
            # Trailing Stop (Sederhana)
            elif self.long_be_active and price <= self.entry_price:
                self.position.close()
            # Trend Reverse
            elif self.st_dir[-1] == -1:
                self.position.close()

        elif self.position.is_short:
            # 1. Break Even Logic
            if not self.short_be_active and price <= (self.entry_price - (atr_val * self.be_trigger_atr)):
                self.short_be_active = True
                print("DEBUG: Break Even Activated for Short")

            # 2. Exit Conditions
            if price <= (self.entry_price - (atr_val * self.tp_atr_mult)):
                self.position.close()
            elif self.short_be_active and price >= self.entry_price:
                self.position.close()
            elif self.st_dir[-1] == 1:
                self.position.close()