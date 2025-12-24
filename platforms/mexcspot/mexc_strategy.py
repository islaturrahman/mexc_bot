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

# IMPLEMENTASI AI Pada Trading
# https://z-library.ec/book/115582930/abf831/handson-ai-trading-with-python-quantconnect-and-aws.html
# https://github.com/QuantConnect/HandsOnAITradingBook/tree/master/



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

class BollingerMacdStrategy(Strategy):

    # Parameter sesuai dengan Pine Script asli
    rapida = 8
    lenta = 26
    stdv_mult = 0.8
    signal_period = 9
    
    def init(self):
        close = pd.Series(self.data.Close)
        
        # 1. Hitung MACD (m_rapida - m_lenta)
        ema_fast = ta.ema(close, length=self.rapida)
        ema_slow = ta.ema(close, length=self.lenta)
        
        self.bb_macd = self.I(lambda: ema_fast - ema_slow)
        
        # 2. Hitung Bollinger Bands di atas garis MACD
        # Avg = ema(BBMacd, 9)
        self.avg = self.I(ta.ema, pd.Series(self.bb_macd), length=self.signal_period)
        
        # SDev = stdev(BBMacd, 9)
        self.sdev = self.I(ta.stdev, pd.Series(self.bb_macd), length=self.signal_period)
        
        # 3. Kalkulasi Pita (Bands)
        self.banda_supe = self.I(lambda: self.avg + self.stdv_mult * self.sdev)
        self.banda_inf = self.I(lambda: self.avg - self.stdv_mult * self.sdev)

    def next(self):
        # Logika Sinyal (Crossover & Crossunder)
        # Compra = crossover(BBMacd, banda_supe)
        if self.bb_macd[-1] > self.banda_supe[-1] and self.bb_macd[-2] <= self.banda_supe[-2]:
            if not self.position:
                self.buy(size=0.9) # Menggunakan 90% equity
                print(f"DEBUG: BUY | MACD: {self.bb_macd[-1]:.4f} > Upper: {self.banda_supe[-1]:.4f}")
        
        # Venta = crossunder(BBMacd, banda_inf)
        elif self.bb_macd[-1] < self.banda_inf[-1] and self.bb_macd[-2] >= self.banda_inf[-2]:
            if self.position:
                self.position.close()
                print(f"DEBUG: SELL | MACD: {self.bb_macd[-1]:.4f} < Lower: {self.banda_inf[-1]:.4f}")


class TurtleTrading1(Strategy):
    # Optimizable Parameters
    system1_entry = 20
    system1_exit = 10
    system2_entry = 55
    system2_exit = 20
    atr_period = 20
    risk_per_trade = 0.0
    max_units = 4
    stop_loss_atr = 2.0
    pyramid_atr = 0.5
    
    def init(self):
        # Pre-calculate all indicators once
        close = self.data.Close
        high = self.data.High
        low = self.data.Low
        
        # ATR calculation (vectorized)
        self.atr = self.I(self._calculate_atr, high, low, close, self.atr_period)
        
        # System 1 indicators
        self.sys1_high = self.I(lambda x: pd.Series(x).rolling(self.system1_entry).max(), high)
        self.sys1_low = self.I(lambda x: pd.Series(x).rolling(self.system1_entry).min(), low)
        self.sys1_exit_high = self.I(lambda x: pd.Series(x).rolling(self.system1_exit).max(), high)
        self.sys1_exit_low = self.I(lambda x: pd.Series(x).rolling(self.system1_exit).min(), low)
        
        # System 2 indicators
        self.sys2_high = self.I(lambda x: pd.Series(x).rolling(self.system2_entry).max(), high)
        self.sys2_low = self.I(lambda x: pd.Series(x).rolling(self.system2_entry).min(), low)
        self.sys2_exit_high = self.I(lambda x: pd.Series(x).rolling(self.system2_exit).max(), high)
        self.sys2_exit_low = self.I(lambda x: pd.Series(x).rolling(self.system2_exit).min(), low)
        
        # State variables
        self.last_trade_loser = False
        self.system_initiated = None
        self.entry_price = 0.0
        self.units_held = 0
        self.last_pyramid_price = 0.0
        
    @staticmethod
    def _calculate_atr(high, low, close, period):
        """Optimized ATR calculation"""
        h = pd.Series(high)
        l = pd.Series(low)
        c = pd.Series(close)
        
        tr = pd.concat([
            h - l,
            (h - c.shift(1)).abs(),
            (l - c.shift(1)).abs()
        ], axis=1).max(axis=1)
        
        return tr.rolling(period).mean()
    
    def calculate_position_size(self, atr, price):
        """Optimized position sizing"""
        if atr <= 0 or price <= 0:
            return 0.0
        
        risk_amount = self.equity * self.risk_per_trade
        size_fraction = min((risk_amount / atr) * price / self.equity, 0.95)
        
        return max(0.01, size_fraction)
    
    def next(self):
        # Early exit if insufficient data
        if len(self.data) < max(self.system2_entry, self.atr_period):
            return
        
        price = self.data.Close[-1]
        atr = self.atr[-1]
        
        if np.isnan(atr) or atr <= 0:
            return
        
        # Pre-calculate common values
        stop_distance = atr * self.stop_loss_atr
        size = self.calculate_position_size(atr, price)
        
        if size <= 0:
            return
        
        # === EXIT LOGIC ===
        if self.position:
            self._handle_exit(price, atr, stop_distance)
            if not self.position:  # Position closed
                return
            
            # Pyramiding
            self._handle_pyramiding(price, atr, size)
            return
        
        # === ENTRY LOGIC ===
        self._handle_entry(price, size)
    
    def _handle_exit(self, price, atr, stop_distance):
        """Handle position exits and reversals"""
        if self.position.is_long:
            # Calculate exit level
            if self.system_initiated == 'system1':
                exit_level = max(self.sys1_exit_low[-2], self.entry_price - stop_distance)
            else:
                exit_level = max(self.sys2_exit_low[-2], self.entry_price - stop_distance)
            
            # Check exit
            if price <= exit_level:
                self.last_trade_loser = (price < self.entry_price)
                self.position.close()
                self._reset_state()
                return
            
            # Check reversal to short
            if price <= self.sys1_low[-2] or price <= self.sys2_low[-2]:
                self.position.close()
                self.sell(size=self.calculate_position_size(atr, price))
                self.entry_price = price
                self.last_pyramid_price = price
                self.units_held = 1
                self.system_initiated = 'system1' if price <= self.sys1_low[-2] else 'system2'
                
        else:  # Short position
            # Calculate exit level
            if self.system_initiated == 'system1':
                exit_level = min(self.sys1_exit_high[-2], self.entry_price + stop_distance)
            else:
                exit_level = min(self.sys2_exit_high[-2], self.entry_price + stop_distance)
            
            # Check exit
            if price >= exit_level:
                self.last_trade_loser = (price > self.entry_price)
                self.position.close()
                self._reset_state()
                return
            
            # Check reversal to long
            if price >= self.sys1_high[-2] or price >= self.sys2_high[-2]:
                self.position.close()
                self.buy(size=self.calculate_position_size(atr, price))
                self.entry_price = price
                self.last_pyramid_price = price
                self.units_held = 1
                self.system_initiated = 'system1' if price >= self.sys1_high[-2] else 'system2'
    
    def _handle_entry(self, price, size):
        """Handle new position entries"""
        # Long entries
        if price >= self.sys1_high[-2] and not self.last_trade_loser:
            self.buy(size=size)
            self.entry_price = price
            self.last_pyramid_price = price
            self.system_initiated = 'system1'
            self.units_held = 1
            
        elif price >= self.sys2_high[-2]:
            self.buy(size=size)
            self.entry_price = price
            self.last_pyramid_price = price
            self.system_initiated = 'system2'
            self.units_held = 1
        
        # Short entries
        elif price <= self.sys1_low[-2] and not self.last_trade_loser:
            self.sell(size=size)
            self.entry_price = price
            self.last_pyramid_price = price
            self.system_initiated = 'system1'
            self.units_held = 1
            
        elif price <= self.sys2_low[-2]:
            self.sell(size=size)
            self.entry_price = price
            self.last_pyramid_price = price
            self.system_initiated = 'system2'
            self.units_held = 1
    
    def _handle_pyramiding(self, price, atr, size):
        """Handle adding to winning positions"""
        if self.units_held >= self.max_units:
            return
        
        pyramid_distance = atr * self.pyramid_atr
        
        if self.position.is_long:
            if price >= self.last_pyramid_price + pyramid_distance:
                current_exposure = abs(self.position.size * price / self.equity)
                if current_exposure < 0.90:
                    add_size = min(size, 0.90 - current_exposure)
                    if add_size >= 0.01:
                        self.buy(size=add_size)
                        self.units_held += 1
                        self.last_pyramid_price = price
                        
        elif self.position.is_short:
            if price <= self.last_pyramid_price - pyramid_distance:
                current_exposure = abs(self.position.size * price / self.equity)
                if current_exposure < 0.90:
                    add_size = min(size, 0.90 - current_exposure)
                    if add_size >= 0.01:
                        self.sell(size=add_size)
                        self.units_held += 1
                        self.last_pyramid_price = price
    
    def _reset_state(self):
        """Reset position state"""
        self.system_initiated = None
        self.units_held = 0
        self.entry_price = 0.0
        self.last_pyramid_price = 0.0



class TurtleTrading2(Strategy):
    # Optimizable Parameters
    system1_entry = 25          # Increased: more selective entries
    system1_exit = 12           # Wider exit to let profits run
    system2_entry = 60          # Increased: stronger trend confirmation
    system2_exit = 25           # Wider exit
    atr_period = 20
    risk_per_trade = 0.015      # Increased position size for bigger wins
    max_units = 2               # Reduced: fewer additions, bigger base size
    stop_loss_atr = 2.5         # Wider stop to avoid premature exits
    pyramid_atr = 1.0           # Increased: only add on strong moves
    
    # NEW: Quality filters
    min_atr_filter = 0.5        # Minimum ATR % of price (volatility filter)
    trend_strength = 15         # ADX-like filter (days for trend strength)
    skip_after_loss = True      # Skip System 1 after loss (original turtle rule)
    consolidation_filter = True # Skip entries during consolidation
    
    def init(self):
        close = self.data.Close
        high = self.data.High
        low = self.data.Low
        
        # ATR calculation
        self.atr = self.I(self._calculate_atr, high, low, close, self.atr_period)
        
        # ATR percentage (volatility filter)
        self.atr_pct = self.I(lambda: (self.atr / close) * 100)
        
        # Trend strength indicator (simplified ADX)
        self.trend_strength_ind = self.I(self._calculate_trend_strength, high, low, close, self.trend_strength)
        
        # System 1 indicators
        self.sys1_high = self.I(lambda x: pd.Series(x).rolling(self.system1_entry).max(), high)
        self.sys1_low = self.I(lambda x: pd.Series(x).rolling(self.system1_entry).min(), low)
        self.sys1_exit_high = self.I(lambda x: pd.Series(x).rolling(self.system1_exit).max(), high)
        self.sys1_exit_low = self.I(lambda x: pd.Series(x).rolling(self.system1_exit).min(), low)
        
        # System 2 indicators
        self.sys2_high = self.I(lambda x: pd.Series(x).rolling(self.system2_entry).max(), high)
        self.sys2_low = self.I(lambda x: pd.Series(x).rolling(self.system2_entry).min(), low)
        self.sys2_exit_high = self.I(lambda x: pd.Series(x).rolling(self.system2_exit).max(), high)
        self.sys2_exit_low = self.I(lambda x: pd.Series(x).rolling(self.system2_exit).min(), low)
        
        # Moving averages for trend confirmation
        self.sma_fast = self.I(lambda x: pd.Series(x).rolling(20).mean(), close)
        self.sma_slow = self.I(lambda x: pd.Series(x).rolling(50).mean(), close)
        
        # State variables
        self.last_trade_loser = False
        self.system_initiated = None
        self.entry_price = 0.0
        self.units_held = 0
        self.last_pyramid_price = 0.0
        self.trades_count = 0
        
    @staticmethod
    def _calculate_atr(high, low, close, period):
        """Calculate ATR"""
        h = pd.Series(high)
        l = pd.Series(low)
        c = pd.Series(close)
        
        tr = pd.concat([
            h - l,
            (h - c.shift(1)).abs(),
            (l - c.shift(1)).abs()
        ], axis=1).max(axis=1)
        
        return tr.rolling(period).mean()
    
    @staticmethod
    def _calculate_trend_strength(high, low, close, period):
        """Calculate trend strength (simplified directional movement)"""
        h = pd.Series(high)
        l = pd.Series(low)
        c = pd.Series(close)
        
        # Directional movement
        up_move = h - h.shift(1)
        down_move = l.shift(1) - l
        
        # Smooth movements
        up_avg = up_move.rolling(period).mean()
        down_avg = down_move.rolling(period).mean()
        
        # Trend strength score
        trend = (up_avg - down_avg).abs() / (up_avg + down_avg + 0.0001) * 100
        
        return trend
    
    def calculate_position_size(self, atr, price):
        """Optimized position sizing - larger base size, fewer additions"""
        if atr <= 0 or price <= 0:
            return 0.0
        
        # Larger position size for quality trades
        risk_amount = self.equity * self.risk_per_trade
        size_fraction = min((risk_amount / atr) * price / self.equity, 0.90)
        
        return max(0.02, size_fraction)
    
    def check_entry_quality(self, price, direction='long'):
        """Quality filters for entries"""
        # Check if enough data
        if len(self.data) < max(self.system2_entry, 50):
            return False
        
        atr = self.atr[-1]
        atr_pct = self.atr_pct[-1]
        
        # Filter 1: Minimum volatility (avoid ranging markets)
        if atr_pct < self.min_atr_filter:
            return False
        
        # Filter 2: Trend strength
        if self.consolidation_filter:
            trend = self.trend_strength_ind[-1]
            if np.isnan(trend) or trend < 20:  # Low trend strength
                return False
        
        # Filter 3: Moving average confirmation
        sma_fast = self.sma_fast[-1]
        sma_slow = self.sma_slow[-1]
        
        if np.isnan(sma_fast) or np.isnan(sma_slow):
            return False
        
        if direction == 'long':
            # Long only if fast MA > slow MA (uptrend)
            if sma_fast <= sma_slow:
                return False
            # Price should be above both MAs
            if price < sma_fast:
                return False
        else:  # short
            # Short only if fast MA < slow MA (downtrend)
            if sma_fast >= sma_slow:
                return False
            # Price should be below both MAs
            if price > sma_fast:
                return False
        
        # Filter 4: Range filter (avoid choppy markets)
        if self.consolidation_filter:
            high_range = self.sys2_high[-1] - self.sys2_low[-1]
            avg_range = atr * 10
            if high_range < avg_range:  # Too narrow range
                return False
        
        return True
    
    def next(self):
        if len(self.data) < max(self.system2_entry, 50):
            return
        
        price = self.data.Close[-1]
        atr = self.atr[-1]
        
        if np.isnan(atr) or atr <= 0:
            return
        
        stop_distance = atr * self.stop_loss_atr
        size = self.calculate_position_size(atr, price)
        
        if size <= 0:
            return
        
        # === EXIT LOGIC ===
        if self.position:
            self._handle_exit(price, atr, stop_distance)
            if not self.position:
                return
            
            # Pyramiding (very selective)
            self._handle_pyramiding(price, atr, size)
            return
        
        # === ENTRY LOGIC (with quality filters) ===
        self._handle_entry(price, size)
    
    def _handle_exit(self, price, atr, stop_distance):
        """Handle exits with trailing stops"""
        if self.position.is_long:
            # Use wider exit and trailing stop
            if self.system_initiated == 'system1':
                exit_level = max(self.sys1_exit_low[-2], self.entry_price - stop_distance)
            else:
                exit_level = max(self.sys2_exit_low[-2], self.entry_price - stop_distance)
            
            # Trailing stop: tighten if in profit
            if price > self.entry_price * 1.05:  # 5% profit
                exit_level = max(exit_level, price - (atr * 1.5))
            
            if price <= exit_level:
                self.last_trade_loser = (price < self.entry_price)
                self.position.close()
                self._reset_state()
                return
            
            # No reversal - exit and wait for new signal
            if price <= self.sys1_low[-2]:
                self.position.close()
                self.last_trade_loser = (price < self.entry_price)
                self._reset_state()
                
        else:  # Short position
            if self.system_initiated == 'system1':
                exit_level = min(self.sys1_exit_high[-2], self.entry_price + stop_distance)
            else:
                exit_level = min(self.sys2_exit_high[-2], self.entry_price + stop_distance)
            
            # Trailing stop for shorts
            if price < self.entry_price * 0.95:  # 5% profit
                exit_level = min(exit_level, price + (atr * 1.5))
            
            if price >= exit_level:
                self.last_trade_loser = (price > self.entry_price)
                self.position.close()
                self._reset_state()
                return
            
            # No reversal - exit and wait
            if price >= self.sys1_high[-2]:
                self.position.close()
                self.last_trade_loser = (price > self.entry_price)
                self._reset_state()
    
    def _handle_entry(self, price, size):
        """Handle entries with quality filters"""
        # System 1 Long Entry
        if price >= self.sys1_high[-2]:
            # Skip if last trade was loser (original turtle rule)
            if self.skip_after_loss and self.last_trade_loser:
                return
            
            # Check entry quality
            if not self.check_entry_quality(price, 'long'):
                return
            
            self.buy(size=size)
            self.entry_price = price
            self.last_pyramid_price = price
            self.system_initiated = 'system1'
            self.units_held = 1
            self.trades_count += 1
            return
        
        # System 2 Long Entry
        if price >= self.sys2_high[-2]:
            if not self.check_entry_quality(price, 'long'):
                return
            
            self.buy(size=size)
            self.entry_price = price
            self.last_pyramid_price = price
            self.system_initiated = 'system2'
            self.units_held = 1
            self.trades_count += 1
            return
        
        # System 1 Short Entry
        if price <= self.sys1_low[-2]:
            if self.skip_after_loss and self.last_trade_loser:
                return
            
            if not self.check_entry_quality(price, 'short'):
                return
            
            self.sell(size=size)
            self.entry_price = price
            self.last_pyramid_price = price
            self.system_initiated = 'system1'
            self.units_held = 1
            self.trades_count += 1
            return
        
        # System 2 Short Entry
        if price <= self.sys2_low[-2]:
            if not self.check_entry_quality(price, 'short'):
                return
            
            self.sell(size=size)
            self.entry_price = price
            self.last_pyramid_price = price
            self.system_initiated = 'system2'
            self.units_held = 1
            self.trades_count += 1
    
    def _handle_pyramiding(self, price, atr, size):
        """Very selective pyramiding - only on strong trends"""
        if self.units_held >= self.max_units:
            return
        
        pyramid_distance = atr * self.pyramid_atr
        
        # Only add if we're significantly in profit
        min_profit_pct = 0.03  # 3% minimum profit to add
        
        if self.position.is_long:
            if price >= self.last_pyramid_price + pyramid_distance:
                # Check if in profit enough
                if price < self.entry_price * (1 + min_profit_pct):
                    return
                
                # Check trend still strong
                if self.sma_fast[-1] <= self.sma_slow[-1]:
                    return
                
                current_exposure = abs(self.position.size * price / self.equity)
                if current_exposure < 0.85:
                    add_size = min(size * 0.5, 0.85 - current_exposure)  # Smaller additions
                    if add_size >= 0.02:
                        self.buy(size=add_size)
                        self.units_held += 1
                        self.last_pyramid_price = price
                        
        elif self.position.is_short:
            if price <= self.last_pyramid_price - pyramid_distance:
                if price > self.entry_price * (1 - min_profit_pct):
                    return
                
                if self.sma_fast[-1] >= self.sma_slow[-1]:
                    return
                
                current_exposure = abs(self.position.size * price / self.equity)
                if current_exposure < 0.85:
                    add_size = min(size * 0.5, 0.85 - current_exposure)
                    if add_size >= 0.02:
                        self.sell(size=add_size)
                        self.units_held += 1
                        self.last_pyramid_price = price
    
    def _reset_state(self):
        """Reset position state"""
        self.system_initiated = None
        self.units_held = 0
        self.entry_price = 0.0
        self.last_pyramid_price = 0.0



class TurtleClassic(Strategy):
    # === PARAMETERS ===
    sys1_entry = 20
    sys1_exit = 10

    sys2_entry = 55
    sys2_exit = 20

    risk_fraction = 0.02      # 2% initial capital
    tick = 0.0001             # price tick
    bigPointValue = 1.0

    position_fraction = 0.25  # fixed position size (Turtle original pakai unit, bukan %)

    def init(self):
        high = self.data.High
        low = self.data.Low
        close = self.data.Close

        # === CHANNELS ===
        self.sys1_high = self.I(lambda x: pd.Series(x).rolling(self.sys1_entry).max(), high)
        self.sys1_low = self.I(lambda x: pd.Series(x).rolling(self.sys1_entry).min(), low)
        self.sys1_exit_low = self.I(lambda x: pd.Series(x).rolling(self.sys1_exit).min(), low)
        self.sys1_exit_high = self.I(lambda x: pd.Series(x).rolling(self.sys1_exit).max(), high)

        self.sys2_high = self.I(lambda x: pd.Series(x).rolling(self.sys2_entry).max(), high)
        self.sys2_low = self.I(lambda x: pd.Series(x).rolling(self.sys2_entry).min(), low)
        self.sys2_exit_low = self.I(lambda x: pd.Series(x).rolling(self.sys2_exit).min(), low)
        self.sys2_exit_high = self.I(lambda x: pd.Series(x).rolling(self.sys2_exit).max(), high)

        # === STATE ===
        self.entry_price = None
        self.system = None
        self.initial_capital = None
        self.cooldown = False

    def next(self):
        if len(self.data) < self.sys2_entry:
            return

        price = self.data.Close[-1]

        if self.initial_capital is None:
            self.initial_capital = self.equity

        stop_risk = (self.risk_fraction * self.initial_capital) / self.bigPointValue

        # ================= EXIT =================
        if self.position:
            if self.position.is_long:
                if self.system == 'sys1':
                    channel_stop = self.sys1_exit_low[-2] - self.tick
                else:
                    channel_stop = self.sys2_exit_low[-2] - self.tick

                fixed_stop = self.entry_price - stop_risk
                exit_price = max(channel_stop, fixed_stop)

                if price <= exit_price:
                    self.position.close()
                    self._reset()
                    return

            else:  # SHORT
                if self.system == 'sys1':
                    channel_stop = self.sys1_exit_high[-2] + self.tick
                else:
                    channel_stop = self.sys2_exit_high[-2] + self.tick

                fixed_stop = self.entry_price + stop_risk
                exit_price = min(channel_stop, fixed_stop)

                if price >= exit_price:
                    self.position.close()
                    self._reset()
                    return

            return

        # ================= ENTRY =================
        if self.cooldown:
            self.cooldown = False
            return

        size = self.position_fraction

        # --- SYSTEM 1 ---
        if price >= self.sys1_high[-2] + self.tick:
            self.buy(size=size)
            self.entry_price = price
            self.system = 'sys1'
            return

        if price <= self.sys1_low[-2] - self.tick:
            self.sell(size=size)
            self.entry_price = price
            self.system = 'sys1'
            return

        # --- SYSTEM 2 ---
        if price >= self.sys2_high[-2] + self.tick:
            self.buy(size=size)
            self.entry_price = price
            self.system = 'sys2'
            return

        if price <= self.sys2_low[-2] - self.tick:
            self.sell(size=size)
            self.entry_price = price
            self.system = 'sys2'
            return

    def _reset(self):
        self.entry_price = None
        self.system = None
        self.cooldown = True





class TurtleTradingOptimized(Strategy):
    # === PARAMETERS ===
    system1_entry = 75         # Standar Turtle (Lebih responsif untuk modal kecil)
    system1_exit = 10
    system2_entry = 80         
    system2_exit = 20

    atr_period = 20            # Standar teknikal (lebih stabil)
    adx_period = 14
    ma_period = 125            # Filter tren lebih kuat

    risk_per_trade = 0.02      # 2% risk agar tidak melanggar min_order $5
    stop_loss_atr = 2.0
    pyramid_atr = 0.5          # Tambah posisi lebih cepat saat profit (agresif)
    max_units = 2              # Dibatasi 2 agar margin aman

    max_exposure = 0.8         # Sisakan 20% untuk fee & buffer volatilitas
    tp_mult = 4.0

    

    def init(self):
        high = self.data.High
        low = self.data.Low
        close = self.data.Close

        # === INDICATORS ===
        self.atr = self.I(self._atr, high, low, close, self.atr_period)
        self.adx = self.I(self._adx, high, low, close, self.adx_period)

        self.ma50 = self.I(lambda x: pd.Series(x).rolling(self.ma_period).mean(), close)

        self.sys1_high = self.I(lambda x: pd.Series(x).rolling(self.system1_entry).max(), high)
        self.sys1_low = self.I(lambda x: pd.Series(x).rolling(self.system1_entry).min(), low)
        self.sys1_exit_high = self.I(lambda x: pd.Series(x).rolling(self.system1_exit).max(), high)
        self.sys1_exit_low = self.I(lambda x: pd.Series(x).rolling(self.system1_exit).min(), low)

        self.sys2_high = self.I(lambda x: pd.Series(x).rolling(self.system2_entry).max(), high)
        self.sys2_low = self.I(lambda x: pd.Series(x).rolling(self.system2_entry).min(), low)
        self.sys2_exit_high = self.I(lambda x: pd.Series(x).rolling(self.system2_exit).max(), high)
        self.sys2_exit_low = self.I(lambda x: pd.Series(x).rolling(self.system2_exit).min(), low)

        self.regime = self.I(self._market_regime, self.data.Close, self.adx, self.atr)
        self.dmi = self.I(self._dmi_components, self.data.High, self.data.Low, self.data.Close)

        # === STATE ===
        self.system1_allowed = True
        self.system_in_use = None
        self.entry_price = None
        self.units = 0
        self.last_pyramid_price = None
        self.cooldown = False


    # === ATR ===
    @staticmethod
    def _atr(h, l, c, p):
        h, l, c = pd.Series(h), pd.Series(l), pd.Series(c)
        tr = pd.concat([
            h - l,
            (h - c.shift()).abs(),
            (l - c.shift()).abs()
        ], axis=1).max(axis=1)
        return tr.rolling(p).mean()
    
    def _adx(self, high, low, close, timeperiod=14):
        high, low, close = pd.Series(high), pd.Series(low), pd.Series(close)
        
        # 1. Calculate True Range
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs()
        ], axis=1).max(axis=1)

        # 2. Calculate Directional Movement
        up_move = high.diff()
        down_move = low.diff()
        
        plus_dm = pd.Series(0, index=high.index)
        minus_dm = pd.Series(0, index=high.index)
        
        # Filter DM sesuai aturan standar
        plus_mask = (up_move > down_move.abs()) & (up_move > 0)
        minus_mask = (down_move.abs() > up_move) & (down_move.abs() > 0)
        
        plus_dm[plus_mask] = up_move[plus_mask]
        minus_dm[minus_mask] = down_move.abs()[minus_mask]

        # 3. Smoothing menggunakan Wilder's Method (EWM)
        atr = tr.ewm(alpha=1/timeperiod, min_periods=timeperiod).mean()
        plus_di = 100 * (plus_dm.ewm(alpha=1/timeperiod, min_periods=timeperiod).mean() / atr)
        minus_di = 100 * (minus_dm.ewm(alpha=1/timeperiod, min_periods=timeperiod).mean() / atr)

        # 4. Calculate DX and ADX
        dx = (abs(plus_di - minus_di) / (plus_di + minus_di + 0.0001)) * 100
        adx = dx.ewm(alpha=1/timeperiod, min_periods=timeperiod).mean()
        
        return adx

    def position_size(self, atr):
        # 1. Hitung jumlah uang yang siap dipertaruhkan (Risk Amount)
        risk_amount = self.equity * self.risk_per_trade
        
        # 2. Hitung jarak Stop Loss dalam satuan harga
        sl_dist = atr * self.stop_loss_atr
        
        if sl_dist <= 0:
            return 0
            
        # 3. Hitung berapa banyak unit aset (Quantity) yang bisa dibeli
        # Formula: Risk / Dist_to_SL
        raw_size = risk_amount / sl_dist
        
        # 4. Konversi ke bentuk proporsi (0.0 s/d 1.0) untuk backtesting.py
        # Kita tambahkan batas (clamping) agar tidak melebihi equity atau max_exposure
        size_proportion = (raw_size * self.data.Close[-1]) / self.equity
        
        # 5. Penyesuaian untuk modal kecil (< $500)
        # Jika hasil proporsi terlalu kecil, broker sering menolak.
        # Kita pastikan size tidak melebihi max_exposure yang ditentukan.
        final_size = min(size_proportion, self.max_exposure)
        
        # Jika modal sangat kecil, pastikan tidak mencoba trading di bawah 1% equity
        # karena biasanya akan kena reject minimum order size di real exchange.
        if final_size < 0.01: 
            return 0
            
        return final_size


    def next(self):
        if len(self.data) < self.system2_entry:
            return

        price = self.data.Close[-1]
        atr = self.atr[-1]
        ma = self.ma50[-1]
        
        # Ambil nilai dari komponen DMI (ADX, Plus_DI, Minus_DI)
        adx = self.dmi[0][-1]
        plus_di = self.dmi[1][-1]
        minus_di = self.dmi[2][-1]
        current_regime = self.regime[-1]

        if np.isnan(atr) or np.isnan(ma):
            return

        if self.cooldown:
            self.cooldown = False
            return

        # ================= LOGIKA EXIT (UTAMA) =================
        if self.position:
            # 1. Exit Khusus Volatile (Take Profit Cepat)
            if current_regime == 2: # VOLATILE_SIDEWAYS
                if self.position.is_long and price > self.entry_price + (atr * self.tp_mult):
                    self.position.close()
                    self._reset()
                    return
                elif self.position.is_short and price < self.entry_price - (atr * self.tp_mult):
                    self.position.close()
                    self._reset()
                    return

            # 2. Exit Standar Turtle (Stop Loss & Donchian)
            stop = atr * self.stop_loss_atr
            if self.position.is_long:
                exit_lvl = self.sys1_exit_low[-2] if self.system_in_use == 'sys1' else self.sys2_exit_low[-2]
                if price <= max(exit_lvl, self.entry_price - stop):
                    self.position.close()
                    self._reset()
                    return
            else:
                exit_lvl = self.sys1_exit_high[-2] if self.system_in_use == 'sys1' else self.sys2_exit_high[-2]
                if price >= min(exit_lvl, self.entry_price + stop):
                    self.position.close()
                    self._reset()
                    return

            # 3. Pyramiding (Hanya jika TRENDING)
            if current_regime == 1: # TRENDING
                size = self.position_size(atr)
                self._pyramid(price, atr, size)
            return

        # ================= LOGIKA ENTRY (TERPUSAT) =================
        # Abaikan entry jika pasar Quiet
        if current_regime == 0: # QUIET_SIDEWAYS
            return

        size = self.position_size(atr)
        if size <= 0:
            return

        # Hanya Entry jika Trending dan DMI Konfirmasi
        if current_regime == 1: # TRENDING
            # System 1 Entry
            if self.system1_allowed:
                if price > self.sys1_high[-2] and price > ma and plus_di > minus_di:
                    if self.data.Volume[-1] > np.mean(self.data.Volume[-20:]):
                        self.buy(size=size)
                        self._init_trade(price, 'sys1')
                        return
                
                elif price < self.sys1_low[-2] and price < ma and minus_di > plus_di:
                    self.sell(size=size)
                    self._init_trade(price, 'sys1')
                    return

            # System 2 Entry (Breakout Lebih Jauh)
            if price > self.sys2_high[-2] and price > ma and plus_di > minus_di:
                self.buy(size=size)
                self._init_trade(price, 'sys2')
            elif price < self.sys2_low[-2] and price < ma and minus_di > plus_di:
                self.sell(size=size)
                self._init_trade(price, 'sys2')

    # === PYRAMIDING (STRICT) ===
    def _pyramid(self, price, atr, size):
        if self.units >= self.max_units:
            return

        step = atr * self.pyramid_atr
        exposure = abs(self.position.size * price / self.equity)

        if exposure >= self.max_exposure:
            return

        if self.position.is_long and price >= self.last_pyramid_price + step:
            self.buy(size=min(size, self.max_exposure - exposure))
            self.units += 1
            self.last_pyramid_price = price

        elif self.position.is_short and price <= self.last_pyramid_price - step:
            self.sell(size=min(size, self.max_exposure - exposure))
            self.units += 1
            self.last_pyramid_price = price

    def _init_trade(self, price, system):
        self.entry_price = price
        self.last_pyramid_price = price
        self.system_in_use = system
        self.units = 1

    def _reset(self):
        self.entry_price = None
        self.last_pyramid_price = None
        self.system_in_use = None
        self.units = 0
        self.cooldown = True

    def _dmi_components(self, high, low, close, period=14):
        h, l, c = pd.Series(high), pd.Series(low), pd.Series(close)
        
        # 1. TR & DM
        tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
        up_move = h.diff()
        down_move = l.diff()
        
        # Inisialisasi dengan float64 secara eksplisit untuk menghindari FutureWarning
        plus_dm = pd.Series(0.0, index=h.index, dtype='float64') 
        minus_dm = pd.Series(0.0, index=h.index, dtype='float64')
        
        plus_mask = (up_move > down_move.abs()) & (up_move > 0)
        minus_mask = (down_move.abs() > up_move) & (down_move.abs() > 0)
        
        plus_dm[plus_mask] = up_move[plus_mask]
        minus_dm[minus_mask] = down_move.abs()[minus_mask]

        # 2. Smoothing (Wilder's)
        atr_smooth = tr.ewm(alpha=1/period, min_periods=period).mean()
        plus_di = 100 * (plus_dm.ewm(alpha=1/period, min_periods=period).mean() / atr_smooth)
        minus_di = 100 * (minus_dm.ewm(alpha=1/period, min_periods=period).mean() / atr_smooth)
        
        # 3. ADX
        dx = (abs(plus_di - minus_di) / (plus_di + minus_di + 0.00001)) * 100
        adx = dx.ewm(alpha=1/period, min_periods=period).mean()
        
        return adx, plus_di, minus_di

    # Di dalam class TurtleTradingOptimized
    def _market_regime(self, close, adx, atr, lookback=20): # Tambahkan argumen 'close' jika digunakan
        adx_s = pd.Series(adx)
        atr_s = pd.Series(atr)
        
        atr_sma = atr_s.rolling(lookback * 2).mean()
        
        regimes = []
        for i in range(len(adx_s)):
            if adx_s.iloc[i] > 25:
                regimes.append(1) # TRENDING
            elif atr_s.iloc[i] > atr_sma.iloc[i]:
                regimes.append(2) # VOLATILE_SIDEWAYS
            else:
                regimes.append(0) # QUIET_SIDEWAYS
                
        return pd.Series(regimes, index=adx_s.index)

