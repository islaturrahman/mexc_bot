import platforms.mexcspot.mexc_spot as mexc
import pandas as pd
import numpy as np
import json

import requests
from datetime import datetime, timedelta
import pytz
import time


from backtesting import Backtest
from backtesting.lib import crossover
from backtesting.test import SMA
from platforms.mexcspot.mexc_strategy import RSI,SimpleRSIStrategy,RSIStrategy



class BacktestMexc:
    def __init__(self, symbol, 
                 limit=1000, 
                 interval="30m",
                 days_ago=350,
                 cash=0, start_time = None, end_time = None,
                 commission = 0,
                 strategy= SimpleRSIStrategy
                 ):
        
        self.market = mexc.mexc_market()
        self.symbol = symbol
        self.limit = limit
        self.interval = interval
        self.days_ago = days_ago
        self.cash = cash
        self.start_time = start_time
        self.end_time = end_time
        self.commission = commission
        self.strategy = strategy

    def get_time_range(self, days_ago: int, end_time: datetime = None):
        """
        Mendapatkan range waktu dalam timestamp Unix (milliseconds)
        """
        wib = pytz.timezone('Asia/Jakarta')
        now = datetime.now(wib)
        end_datetime = end_time if end_time else now
        
        end = int(end_datetime.timestamp() * 1000)

        start_date = (end_datetime - timedelta(days=days_ago)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        start = int(start_date.timestamp() * 1000)
        
        return start, end

    def get_time_range(self, days_ago: int):
        # Menggunakan UTC secara internal jauh lebih aman untuk API Exchange
        now = datetime.now(pytz.utc)
        start_date = (now - timedelta(days=days_ago)).replace(hour=0, minute=0, second=0, microsecond=0)
        
        # Konversi ke milidetik
        start_ms = int(start_date.timestamp() * 1000)
        end_ms = int(now.timestamp() * 1000)
        
        return start_ms, end_ms
   

    def get_ohlcv_history(self, sleep=0.2):
        """
        Mengambil data dengan proteksi duplikasi dan chunking otomatis
        """
        current_start, total_end = self.get_time_range(self.days_ago)
        all_data = []
        
        print(f"[*] Memulai fetch dari: {datetime.fromtimestamp(current_start/1000)}")
        print(f"[*] Target berakhir di: {datetime.fromtimestamp(total_end/1000)}")

        while current_start < total_end:
            params = {
                "symbol": self.symbol,
                "interval": self.interval,
                "startTime": current_start,
                "endTime": total_end,
                "limit": 1000
            }

            try:
                # Menggunakan wrapper MEXC SDK Anda
                data = self.market.get_kline(params=params)
                
                # Validasi data: get_kline mengembalikan list of lists
                if not data or not isinstance(data, list):
                    print("[INFO] Selesai atau tidak ada data yang diterima.")
                    break

                all_data.extend(data)
                
                # Update start_time berdasarkan candle terakhir + 1ms
                last_timestamp = data[-1][0]
                new_start = last_timestamp + 1
                
                print(f"[PROGRESS] Fetched {len(data)} candles. Last date: {datetime.fromtimestamp(last_timestamp/1000)}")
                
                if new_start <= current_start:
                    break
                    
                current_start = new_start
                time.sleep(sleep)

            except Exception as e:
                print(f"[ERROR] Koneksi/API gagal: {e}")
                time.sleep(2)
                continue

        # Kembalikan list mentah agar diproses konsisten di save_ohlcv_to_csv
        return all_data

    def save_ohlcv_to_csv(self, filename="data/ohlcv_data.csv"):
        """
        Mengambil data mentah, membersihkan, mengonversi tipe data, dan simpan ke CSV
        """
        # 1. Panggil fungsi fetcher
        raw_data = self.get_ohlcv_history()

        if not raw_data:
            print("[ERROR] Tidak ada data yang berhasil diambil.")
            return None

        # 2. Konversi ke DataFrame dengan kolom yang sesuai contoh data MEXC Anda
        df = pd.DataFrame(raw_data, columns=[
            "open_time", "open", "high", "low", "close", 
            "volume", "close_time", "quote_volume"
        ])

        # 3. Konversi tipe data (WAJIB: karena MEXC mengembalikan string)
        # Gunakan unit='ms' untuk timestamp Unix
        # Konversi timestamp ke datetime UTC, lalu ubah ke Asia/Jakarta
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.tz_convert('Asia/Jakarta')

        # Penting: Hilangkan info timezone agar library backtesting.py tidak error
        df["open_time"] = df["open_time"].dt.tz_localize(None)

        df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")

        # Konversi UTC ke WIB agar jamnya sesuai dengan jam laptop Anda

        # Konversi kolom harga & volume dari string ke float
        numeric_cols = ["open", "high", "low", "close", "volume", "quote_volume"]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # 4. Bersihkan duplikat secara total
        initial_count = len(df)
        df = df.drop_duplicates(subset="open_time")
        df = df.sort_values("open_time").reset_index(drop=True)
        
        if initial_count > len(df):
            print(f"[INFO] Berhasil menghapus {initial_count - len(df)} baris duplikat.")

        # 5. Simpan ke CSV
        df.to_csv(filename, index=False)

        print(f"\n[OK] Berhasil menyimpan {len(df):,} baris ke {filename}")
        print(f"[INFO] Rentang Waktu: {df['open_time'].min()} s/d {df['open_time'].max()}")
        return df



    def validate_csv(self, filename, interval_minutes=1):
        """
        Validasi apakah ada gap/missing candle dalam data
        """
        df = pd.read_csv(filename, parse_dates=["open_time"])

        if len(df) == 0:
            print("[ERROR] CSV file is empty")
            return None

        expected = interval_minutes * 60  # dalam detik
        diffs = df["open_time"].diff().dt.total_seconds()

        # Skip baris pertama (NaN) dan cari yang tidak sesuai expected interval
        gaps = df[(diffs != expected) & (diffs.notna())]

        if len(gaps) > 0:
            print(f"\n[WARNING] Found {len(gaps)} gaps/irregularities")
            print("\nFirst 5 gaps:")
            print(gaps[["open_time", "close"]].head())
            return gaps

        print("\n[OK] No candle gap detected")
        return None

    def prepare_data_for_backtest(self, filename='data/ohlcv_data.csv'):
        """
        Prepare data dari CSV untuk backtesting
        """
        df = pd.read_csv(filename, parse_dates=['open_time'])
        
        # Rename kolom sesuai format backtesting.py (harus kapitalisasi)
        df = df.rename(columns={
            'open_time': 'Date',
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        })
        
        # Set Date sebagai index
        df = df.set_index('Date')
        
        # Ambil hanya kolom yang diperlukan
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        
        print(f"[INFO] Prepared {len(df)} rows for backtesting")
        print(f"[INFO] Date range: {df.index.min()} to {df.index.max()}")
        
        return df

    def check_rsi_signals(self, filename='data/ohlcv_data.csv'):
        """
        Debug: Cek berapa banyak sinyal RSI yang muncul di data
        """
        df = self.prepare_data_for_backtest(filename)
        
        # Hitung RSI
        rsi_values = RSI(df['Close'].values, 14)
        df['RSI'] = rsi_values
        
        # Hitung sinyal
        oversold = (df['RSI'] < 30).sum()
        overbought = (df['RSI'] > 70).sum()
        moderate = ((df['RSI'] >= 40) & (df['RSI'] <= 60)).sum()
        
        print("\n" + "="*60)
        print("RSI SIGNAL ANALYSIS")
        print("="*60)
        print(f"Total candles: {len(df)}")
        print(f"Valid RSI values: {(~np.isnan(df['RSI'])).sum()}")
        print(f"NaN RSI values: {np.isnan(df['RSI']).sum()}")
        print(f"\nOversold signals (RSI < 30): {oversold} ({oversold/len(df)*100:.2f}%)")
        print(f"Overbought signals (RSI > 70): {overbought} ({overbought/len(df)*100:.2f}%)")
        print(f"Moderate signals (40-60): {moderate} ({moderate/len(df)*100:.2f}%)")
        print(f"\nRSI range: {np.nanmin(df['RSI']):.2f} - {np.nanmax(df['RSI']):.2f}")
        print(f"RSI mean: {np.nanmean(df['RSI']):.2f}")
        print(f"RSI median: {np.nanmedian(df['RSI']):.2f}")
        print("="*60)
        
        # Tampilkan beberapa contoh sinyal
        if oversold > 0:
            print("\n✅ Contoh 5 sinyal OVERSOLD pertama:")
            print(df[df['RSI'] < 30][['Close', 'RSI']].head())
        else:
            print("\n❌ TIDAK ADA sinyal OVERSOLD di data!")
        
        if overbought > 0:
            print("\n✅ Contoh 5 sinyal OVERBOUGHT pertama:")
            print(df[df['RSI'] > 70][['Close', 'RSI']].head())
        else:
            print("\n❌ TIDAK ADA sinyal OVERBOUGHT di data!")
        
        # Cek 20 data pertama untuk debugging
        print("\n" + "="*60)
        print("FIRST 20 CANDLES (for debugging):")
        print("="*60)
        print(df[['Close', 'RSI']].head(20))
        
        return df

    def run_backtest(self, filename='data/ohlcv_data.csv'):
        """
        Jalankan backtest dengan strategi RSI
        """
        # Prepare data
        df = self.prepare_data_for_backtest(filename)
        
        # Setup backtest
        bt = Backtest(
            df, 
            self.strategy,
            cash=self.cash,
            commission=self.commission,
            exclusive_orders=True
        )
        
        # Run backtest
        print("\n[INFO] Running backtest...")
        stats = bt.run()
        
        # Print hasil
        print("\n" + "="*60)
        print("BACKTEST RESULTS")
        print("="*60)
        print(stats)
        print("="*60)
        
        # Plot hasil (optional)
        try:
            file_report_path = f'reports/{self.strategy.__name__}'
            bt.plot(filename=file_report_path)

        except Exception as e:
            print(f"[WARNING] Could not generate plot: {e}")
        
        return stats, bt
    
    def optimize_strategy(self, filename='data/ohlcv_data.csv'):
        """
        Optimasi parameter strategi RSI
        """
        df = self.prepare_data_for_backtest(filename)
        
        bt = Backtest(
            df,
            RSIStrategy,
            cash=self.cash,
            commission=self.commission,
            exclusive_orders=True
        )
        
        print("\n[INFO] Optimizing RSI parameters...")
        stats = bt.optimize(
            rsi_period=range(10, 21, 2),
            rsi_upper=range(60, 81, 5),
            rsi_lower=range(20, 41, 5),
            maximize='Return [%]',
            constraint=lambda p: p.rsi_lower < p.rsi_upper
        )
        
        print("\n" + "="*60)
        print("OPTIMIZATION RESULTS")
        print("="*60)
        print(stats)
        print("="*60)
        
        return stats


