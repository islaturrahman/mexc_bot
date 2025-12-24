from platforms.mexcspot.config import api_key, secret_key, mexc_host
from platforms.mexcspot import mexc_spot as mexc

# from platforms.mexcspot.ml_strategy import CPOStrategy

import time
import json
import os
import pandas as pd
import numpy as np
from datetime import datetime
import pytz
from decimal import Decimal, ROUND_DOWN

# Asumsi: library 'mexc_sdk' Anda
# import mexc_sdk as mexc 

class MexcTurtleBot():
    def __init__(self, symbol="BTCUSDT", interval="30m"):
        # --- API Setup ---
        self.market = mexc.mexc_market()
        self.trade = mexc.mexc_trade()
        self.symbol = symbol
        self.interval = interval
        
        # --- State Management File ---
        self.state_file = f"bot_state_{self.symbol}.json"
        
        # --- Strategy Parameters (Sesuai Prompt) ---
        self.sys1_entry = 20
        self.sys1_exit = 10  # Disesuaikan (Turtle asli 10, prompt 15/20 beda dikit, kita pakai konservatif)
        self.sys2_entry = 55
        self.sys2_exit = 20
        self.atr_period = 20
        self.adx_period = 14
        self.ma_period = 50 # Sesuai prompt ma50
        
        self.risk_per_trade = 0.02  # Risiko 2% per trade    
        self.stop_loss_atr = 2.0
        self.pyramid_atr = 1.0          
        self.max_units = 3
        self.tp_mult = 3.0
        
        # Load State Awal
        self.load_state()
        print(f"[INIT] Turtle Bot Loaded. State: {self.state}")

    
    # ================= DATA & INDICATORS =================
    def get_data(self):
        klines = self.market.get_kline(params={
            'symbol': self.symbol, 'interval': self.interval, 'limit': 200
            
        })
        
        # 1. Definisikan nama kolom sesuai struktur data response (8 kolom)
        col_names = ['time', 'open', 'high', 'low', 'close', 'vol', 'close_time', 'quote_vol']
        
        df = pd.DataFrame(klines, columns=col_names)
    
        # Konversi timestamp ke Datetime UTC dulu
        df['time'] = pd.to_datetime(df['time'], unit='ms', utc=True)
        
        # Konversi ke waktu lokal (Contoh: Makassar/WITA = +8)
        # Anda bisa mengganti 'Asia/Makassar' dengan 'Asia/Jakarta' untuk WIB
        local_tz = pytz.timezone('Asia/Makassar')
        df['time'] = df['time'].dt.tz_convert(local_tz)
        
        # Jika ingin membuang informasi timezone agar bersih saat diprint:
        df['time'] = df['time'].dt.tz_localize(None)

        # 2. Konversi tipe data string ke float untuk kolom harga/volume
        numeric_cols = ['open', 'high', 'low', 'close', 'vol', 'quote_vol']
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

        # print(f"Last close price: {df['close'].iloc[-1]}")

        # 3. Konversi timestamp (ms) ke datetime object (Agar terbaca manusia & memudahkan debugging)
        # df['time'] = pd.to_datetime(df['time'], unit='ms')
        # --- Hitung Indikator ---
        # 1. Donchian Channels (Rolling Max/Min)
        df['sys1_high'] = df['high'].rolling(self.sys1_entry).max().shift(1) # Shift 1 agar tidak lookahead bias
        df['sys1_low'] = df['low'].rolling(self.sys1_entry).min().shift(1)
        df['sys1_exit_high'] = df['high'].rolling(self.sys1_exit).max().shift(1)
        df['sys1_exit_low'] = df['low'].rolling(self.sys1_exit).min().shift(1)

        df['sys2_high'] = df['high'].rolling(self.sys2_entry).max().shift(1)
        df['sys2_low'] = df['low'].rolling(self.sys2_entry).min().shift(1)
        df['sys2_exit_high'] = df['high'].rolling(self.sys2_exit).max().shift(1)
        df['sys2_exit_low'] = df['low'].rolling(self.sys2_exit).min().shift(1)

        # 2. Moving Average
        df['ma50'] = df['close'].rolling(self.ma_period).mean()

        # 3. ATR
        df['tr1'] = df['high'] - df['low']
        df['tr2'] = (df['high'] - df['close'].shift(1)).abs()
        df['tr3'] = (df['low'] - df['close'].shift(1)).abs()
        df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
        df['atr'] = df['tr'].rolling(self.atr_period).mean()

        # 4. ADX & DMI (Simplified Vectorized)
        up = df['high'].diff()
        down = -df['low'].diff()
        
        # Menggunakan .copy() untuk menghindari SettingWithCopyWarning
        plus_dm = pd.Series(0.0, index=df.index)
        minus_dm = pd.Series(0.0, index=df.index)
        
        plus_dm[(up > down) & (up > 0)] = up
        minus_dm[(down > up) & (down > 0)] = down
        
        alpha = 1/self.adx_period
        df['plus_di'] = 100 * (plus_dm.ewm(alpha=alpha, min_periods=self.adx_period).mean() / df['atr'])
        df['minus_di'] = 100 * (minus_dm.ewm(alpha=alpha, min_periods=self.adx_period).mean() / df['atr'])
        
        dx = (abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'] + 1e-9)) * 100
        df['adx'] = dx.ewm(alpha=alpha, min_periods=self.adx_period).mean()

        # 5. Market Regime
        # 0: Quiet, 1: Trending, 2: Volatile
        atr_sma = df['atr'].rolling(40).mean() # SMA ATR jangka panjang
        
        conditions = [
            (df['adx'] > 25), # Trending
            (df['atr'] > atr_sma) & (df['adx'] <= 25) # Volatile Sideways
        ]
        choices = [1, 2]
        df['regime'] = np.select(conditions, choices, default=0) # 0 = Quiet

        return df
    
    # ================= STATE MANAGEMENT =================
    def load_state(self):
        """Memuat status trading terakhir agar tahan terhadap restart"""
        if os.path.exists(self.state_file):
            with open(self.state_file, 'r') as f:
                self.state = json.load(f)
        else:
            self.state = {
                'in_position': False,
                'position_side': None,  # 'LONG'
                'entry_price': 0.0,
                'units': 0,
                'last_pyramid_price': 0.0,
                'system_in_use': None,  # 'sys1' or 'sys2'
                'stop_loss_price': 0.0
            }


    def save_state(self):
        """Menyimpan status ke file JSON"""
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=4)


    # ================= POSITION FROM EXCHANGE =================
    def get_open_position_qty(self):
        """
        Hitung posisi REAL dari order history (SPOT)
        BUY  = +qty
        SELL = -qty
        """
        orders = self.trade.get_allorders(params={
            "symbol": self.symbol,
            "limit": 500
        })

        net_qty = Decimal("0")

        for o in orders:
            if o["status"] != "FILLED":
                continue

            qty = Decimal(o["executedQty"])
            if o["side"] == "BUY":
                net_qty += qty
            elif o["side"] == "SELL":
                net_qty -= qty

        return float(max(net_qty, Decimal("0")))


    # ================= LOGIKA ORDER =================
    def execute_trade(self, side, quantity):
        print(f"[{side}] Executing {quantity} {self.symbol}...")
        try:
            params = {
                'symbol': self.symbol,
                'side': side,
                'type': 'MARKET',
                'quantity': quantity
            }
            order = self.trade.post_order(params=params)
            print(f"> Order Executed: {order}")
            return True
        except Exception as e:
            print(f"> Order Failed: {e}")
            return False


    def adjust_qty(self, qty, symbol_info):
        step = Decimal(symbol_info["baseSizePrecision"])
        return float(
            Decimal(str(qty)).quantize(step, rounding=ROUND_DOWN)
        )


    def calculate_position_size(self, equity, atr):
        equity = Decimal(str(equity))
        atr = Decimal(str(atr))

        risk_amount = equity * Decimal(str(self.risk_per_trade))
        sl_distance = atr * Decimal(str(self.stop_loss_atr))

        if sl_distance <= 0:
            return 0.0

        raw_size = risk_amount / sl_distance

        exchange_info = self.market.get_exchangeInfo(params={"symbol": self.symbol})
        symbol_info = exchange_info["symbols"][0]

        qty = Decimal(raw_size).quantize(
            Decimal(symbol_info["baseSizePrecision"]),
            rounding=ROUND_DOWN
        )

        return float(qty)


    # ================= LOGIKA UTAMA =================
    def check_signals(self):
        print(self.get_open_position_qty())
        
        df = self.get_data()
        if df.empty:
            return

        curr = df.iloc[-1]
        prev = df.iloc[-2]

        price = curr['close']
        atr = prev['atr']
        regime = prev['regime']

        # ---- ACCOUNT INFO (HANYA UNTUK USDT) ----
        acc = self.trade.get_account_info()
        usdt = float(next((x['free'] for x in acc['balances'] if x['asset'] == 'USDT'), 0))

        open_qty = self.get_open_position_qty()
        total_equity = usdt + (open_qty * price)

        print(f"[INFO] USDT: {usdt} | OpenQty: {open_qty} | Equity: {total_equity}")

        # =====================================================
        # EXIT LOGIC
        # =====================================================
        if self.state['in_position']:
            is_long = self.state['position_side'] == 'LONG'
            entry = self.state['entry_price']

            should_close = False
            close_reason = ""

            # 1. Take Profit Volatile
            if regime == 2:
                tp_price = entry + (atr * self.tp_mult)
                if price >= tp_price:
                    should_close = True
                    close_reason = "TP Volatile"

            # 2. Stop Loss / Donchian Exit
            stop_price = self.state['stop_loss_price']
            exit_lvl = (
                prev['sys1_exit_low']
                if self.state['system_in_use'] == 'sys1'
                else prev['sys2_exit_low']
            )

            if price <= stop_price or price <= exit_lvl:
                should_close = True
                close_reason = "StopLoss / Donchian Exit"

            if should_close:
                print(f">>> SIGNAL CLOSE: {close_reason}")

                if open_qty <= 0:
                    print(">>> Position already closed on exchange")
                    self.state['in_position'] = False
                    self.save_state()
                    return

                if self.execute_trade('SELL', open_qty):
                    self.state = {
                        'in_position': False,
                        'position_side': None,
                        'entry_price': 0,
                        'units': 0,
                        'last_pyramid_price': 0,
                        'system_in_use': None,
                        'stop_loss_price': 0
                    }
                    self.save_state()
                return

            # 3. PYRAMIDING (TREND ONLY)
            if regime == 1 and self.state['units'] < self.max_units:
                step = atr * self.pyramid_atr
                if price >= self.state['last_pyramid_price'] + step:
                    print(">>> PYRAMID ADD")

                    size = self.calculate_position_size(total_equity, atr)
                    if size > 0 and self.execute_trade('BUY', size):
                        self.state['units'] += 1
                        self.state['last_pyramid_price'] = price
                        self.state['stop_loss_price'] += (0.5 * atr)
                        self.save_state()

        # =====================================================
        # ENTRY LOGIC
        # =====================================================
        else:
            if regime == 0:
                return

            size = self.calculate_position_size(usdt, atr)
            if size <= 0:
                return

            signal_found = False
            sys_used = None

            if (
                price > prev['sys1_high']
                and price > prev['ma50']
                and prev['plus_di'] > prev['minus_di']
            ):
                if prev['vol'] > df['vol'].rolling(20).mean().iloc[-2]:
                    signal_found = True
                    sys_used = 'sys1'

            elif (
                price > prev['sys2_high']
                and price > prev['ma50']
                and prev['plus_di'] > prev['minus_di']
            ):
                signal_found = True
                sys_used = 'sys2'

            if signal_found:
                print(f">>> SIGNAL ENTRY ({sys_used})")
                if self.execute_trade('BUY', size):
                    self.state['in_position'] = True
                    self.state['position_side'] = 'LONG'
                    self.state['entry_price'] = price
                    self.state['last_pyramid_price'] = price
                    self.state['units'] = 1
                    self.state['system_in_use'] = sys_used
                    self.state['stop_loss_price'] = price - (atr * self.stop_loss_atr)
                    self.save_state()


    def run(self):
        print("--- Turtle Bot Optimized Started ---")
        while True:
            try:
                self.check_signals()
                print(f"[ANALYSIS] Now {datetime.now().strftime('%H:%M:%S')} - Datatime : {self.get_data()['time'].iloc[-1]} |{self.symbol}| Price Now : {self.get_data()['close'].iloc[-1]}")
                time.sleep(5) # Cek setiap 1 menit (karena interval candle 30m, ini cukup responsif)
            except KeyboardInterrupt:
                print("Bot Stopped.")
                break
            except Exception as e:
                print(f"[ERROR] {e}")
                time.sleep(10)

# if __name__ == "__main__":
#     bot = MexcTurtleBot(symbol="BTCUSDT")
#     bot.run()