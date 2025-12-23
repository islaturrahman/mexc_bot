"""
Enhanced DDQN Trading Strategy
File: platforms/mexcspot/ml_strategy.py

Copy this entire file to replace your ml_strategy.py
"""

import numpy as np
import pandas_ta as ta
import pandas as pd
import tensorflow as tf
from tf_keras.models import Sequential, Model
from tf_keras.layers import Dense, Dropout, BatchNormalization, Input, Lambda
from tf_keras.optimizers import Adam
from tf_keras.regularizers import l2    
import tf_keras.backend as K
from collections import deque
from backtesting import Strategy
from backtesting.lib import crossover


class DDQNAgent(Strategy):
    """
    Enhanced Double Deep Q-Network Trading Strategy
    
    Improvements:
    1. Dueling Network Architecture
    2. Prioritized Experience Replay
    3. Multi-step Returns
    4. Enhanced State Representation
    5. Dynamic Position Sizing
    6. Risk Management Layer
    """
    
    # ==================== HYPERPARAMETERS ====================
    # Network Architecture
    architecture = (512, 256, 128)
    learning_rate = 0.0001
    l2_reg = 1e-5
    
    # RL Parameters
    gamma = 0.995  # Increased for longer-term rewards
    tau = 500  # Target network update frequency
    batch_size = 128  # Reduced for more frequent updates
    replay_capacity = int(5e5)
    
    # Exploration
    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay_steps = 100000
    
    # Multi-step learning
    n_step = 5
    
    # Risk Management
    max_position_size = 0.95
    min_position_size = 0.1
    stop_loss_pct = 0.03  # 3% stop loss
    take_profit_pct = 0.06  # 6% take profit
    max_drawdown_threshold = 0.15  # Stop trading if DD > 15%
    
    # Prioritized Experience Replay
    per_alpha = 0.6
    per_beta_start = 0.4
    per_beta_frames = 100000
    per_epsilon = 1e-6

    def init(self):
        """Initialize the agent"""
        # State dimension calculation:
        # = data features + position features (4) + equity features (2)
        base_features = len(self.data.df.columns)
        position_features = 4  # is_long, is_short, pl_pct, position_bars
        equity_features = 2    # equity_drawdown, trade_count
        self.state_dim = base_features + position_features + equity_features
        self.num_actions = 3  # 0: Sell/Close, 1: Hold, 2: Buy/Long
        
        # Networks
        self.online_network = self.build_dueling_network()
        self.target_network = self.build_dueling_network()
        self.update_target()
        
        # Experience Replay with Priorities
        self.experience = deque([], maxlen=self.replay_capacity)
        self.priorities = deque([], maxlen=self.replay_capacity)
        
        # Tracking variables
        self.total_steps = 0
        self.training_steps = 0
        self.episode_rewards = []
        self.current_epsilon = self.epsilon_start
        
        # Risk tracking
        self.peak_equity = self.equity
        self.entry_price = None
        self.position_bars = 0
        
        # Multi-step buffer
        self.n_step_buffer = deque([], maxlen=self.n_step)
        
        print(f"Enhanced DDQN Agent Initialized")
        print(f"State Dimension: {self.state_dim}")
        print(f"Action Space: {self.num_actions}")

    def build_dueling_network(self):
        """
        Build Dueling DQN Architecture
        Separates state value and action advantages
        """
        inputs = Input(shape=(self.state_dim,))
        
        # Shared layers
        x = inputs
        for i, units in enumerate(self.architecture):
            x = Dense(units, activation='relu', 
                     kernel_regularizer=l2(self.l2_reg),
                     name=f'shared_{i}')(x)
            x = BatchNormalization()(x)
            if i < len(self.architecture) - 1:
                x = Dropout(0.2)(x)
        
        # Value stream
        value_stream = Dense(128, activation='relu', name='value_hidden')(x)
        value_stream = Dropout(0.2)(value_stream)
        value = Dense(1, name='value')(value_stream)
        
        # Advantage stream
        advantage_stream = Dense(128, activation='relu', name='advantage_hidden')(x)
        advantage_stream = Dropout(0.2)(advantage_stream)
        advantage = Dense(self.num_actions, name='advantage')(advantage_stream)
        
        # Combine streams: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        def dueling_layer(inputs):
            value, advantage = inputs
            # Subtract mean advantage for identifiability
            return value + (advantage - K.mean(advantage, axis=1, keepdims=True))
        
        q_values = Lambda(dueling_layer, name='q_values')([value, advantage])
        
        model = Model(inputs=inputs, outputs=q_values)
        model.compile(loss='huber', optimizer=Adam(learning_rate=self.learning_rate))
        
        return model

    def get_state(self):
        """Enhanced state representation"""
        # Get current bar features
        state = self.data.df.iloc[-1].values
        
        # Add position information
        position_state = np.array([
            1.0 if self.position and self.position.is_long else 0.0,
            1.0 if self.position and self.position.is_short else 0.0,
            self.position.pl_pct if self.position else 0.0,
            self.position_bars / 100.0 if self.position else 0.0
        ])
        
        # Add equity curve information
        equity_state = np.array([
            (self.equity - self.peak_equity) / self.peak_equity if self.peak_equity > 0 else 0.0,
            len(self.trades) / 1000.0  # Normalized trade count
        ])
        
        # Combine all states
        full_state = np.concatenate([state, position_state, equity_state])
        
        return full_state.reshape(1, -1)

    def get_action(self, state):
        """Epsilon-greedy action selection with decay"""
        # Update epsilon
        self.current_epsilon = max(
            self.epsilon_end,
            self.epsilon_start - (self.epsilon_start - self.epsilon_end) * 
            self.total_steps / self.epsilon_decay_steps
        )
        
        if np.random.rand() <= self.current_epsilon:
            return np.random.choice(self.num_actions)
        else:
            q_values = self.online_network.predict(state, verbose=0)
            return np.argmax(q_values[0])

    def calculate_position_size(self, confidence):
        """
        Dynamic position sizing based on Q-value confidence
        Kelly Criterion inspired approach
        """
        # Get Q-values for current state
        q_values = self.online_network.predict(self.get_state(), verbose=0)[0]
        max_q = np.max(q_values)
        
        # Calculate confidence (difference between best and second-best action)
        sorted_q = np.sort(q_values)
        confidence_score = (sorted_q[-1] - sorted_q[-2]) / (abs(sorted_q[-1]) + 1e-8)
        
        # Scale position size
        position_size = self.min_position_size + (self.max_position_size - self.min_position_size) * confidence_score
        position_size = np.clip(position_size, self.min_position_size, self.max_position_size)
        
        
        return position_size

    def check_risk_management(self):
        """Check stop loss, take profit, and max drawdown"""
        if not self.position:
            return False
            
        # Check drawdown
        current_dd = (self.peak_equity - self.equity) / self.peak_equity
        if current_dd > self.max_drawdown_threshold:
            return True  # Force close
        
        # Check position P&L
        if self.position.pl_pct <= -self.stop_loss_pct:
            return True  # Stop loss
        elif self.position.pl_pct >= self.take_profit_pct:
            return True  # Take profit
            
        return False

    def calculate_reward(self, prev_equity):
        """
        Enhanced reward function
        Considers: returns, risk-adjusted returns, trade efficiency
        """
        # Base reward: equity change
        equity_change = (self.equity - prev_equity) / prev_equity
        
        # Penalize frequent trading
        trade_cost = -0.001 if len(self.trades) > 0 and self.trades[-1].exit_bar == len(self.data) - 1 else 0
        
        # Reward staying in winning positions
        position_reward = 0
        if self.position:
            if self.position.pl_pct > 0:
                position_reward = 0.01 * self.position.pl_pct
        
        # Penalize excessive drawdown
        dd_penalty = 0
        current_dd = (self.peak_equity - self.equity) / self.peak_equity
        if current_dd > 0.05:
            dd_penalty = -current_dd * 0.1
        
        # Total reward
        reward = equity_change + trade_cost + position_reward + dd_penalty
        
        return reward * 100  # Scale for better learning

    def store_experience(self, state, action, reward, next_state, done):
        """Store experience with priority"""
        self.n_step_buffer.append((state, action, reward, next_state, done))
        
        if len(self.n_step_buffer) >= self.n_step:
            # Calculate n-step return
            n_step_return = sum([self.gamma**i * self.n_step_buffer[i][2] 
                                for i in range(self.n_step)])
            
            first_state, first_action = self.n_step_buffer[0][:2]
            last_next_state, last_done = self.n_step_buffer[-1][3:]
            
            # Calculate initial priority (TD error)
            q_current = self.online_network.predict(first_state, verbose=0)[0, first_action]
            q_next = np.max(self.target_network.predict(last_next_state, verbose=0)[0])
            td_error = abs(n_step_return + self.gamma**self.n_step * q_next * (1 - last_done) - q_current)
            
            priority = (td_error + self.per_epsilon) ** self.per_alpha
            
            self.experience.append((first_state, first_action, n_step_return, last_next_state, last_done))
            self.priorities.append(priority)

    def train(self):
        """Train the network using prioritized experience replay"""
        if len(self.experience) < self.batch_size:
            return
        
        # Calculate sampling probabilities
        priorities_array = np.array(self.priorities)
        probs = priorities_array / priorities_array.sum()
        
        # Sample batch
        indices = np.random.choice(len(self.experience), self.batch_size, p=probs, replace=False)
        
        # Calculate importance sampling weights
        per_beta = min(1.0, self.per_beta_start + self.training_steps * 
                      (1.0 - self.per_beta_start) / self.per_beta_frames)
        weights = (len(self.experience) * probs[indices]) ** (-per_beta)
        weights /= weights.max()
        
        # Prepare batch
        states = np.vstack([self.experience[i][0] for i in indices])
        actions = np.array([self.experience[i][1] for i in indices])
        rewards = np.array([self.experience[i][2] for i in indices])
        next_states = np.vstack([self.experience[i][3] for i in indices])
        dones = np.array([self.experience[i][4] for i in indices])
        
        # Double DQN: use online network to select action, target network to evaluate
        next_q_online = self.online_network.predict(next_states, verbose=0)
        next_actions = np.argmax(next_q_online, axis=1)
        next_q_target = self.target_network.predict(next_states, verbose=0)
        next_q_values = next_q_target[np.arange(self.batch_size), next_actions]
        
        # Calculate targets
        targets = self.online_network.predict(states, verbose=0)
        targets[np.arange(self.batch_size), actions] = rewards + self.gamma**self.n_step * next_q_values * (1 - dones)
        
        # Train with importance sampling weights
        self.online_network.fit(states, targets, sample_weight=weights, verbose=0, epochs=1, batch_size=self.batch_size)
        
        # Update priorities
        new_q_values = self.online_network.predict(states, verbose=0)
        td_errors = np.abs(targets[np.arange(self.batch_size), actions] - new_q_values[np.arange(self.batch_size), actions])
        new_priorities = (td_errors + self.per_epsilon) ** self.per_alpha
        
        for i, idx in enumerate(indices):
            self.priorities[idx] = new_priorities[i]
        
        self.training_steps += 1

    def update_target(self):
        """Update target network"""
        self.target_network.set_weights(self.online_network.get_weights())

    def next(self):
        """Main trading logic - called every bar"""
        self.total_steps += 1
        self.position_bars += 1 if self.position else 0
        
        # Update peak equity
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity
        
        # Get state and action
        prev_equity = self.equity
        state = self.get_state()
        action = self.get_action(state)
        
        # Risk management check
        force_close = self.check_risk_management()
        
        # Execute action
        if force_close:
            if self.position:
                self.position.close()
                self.position_bars = 0
        else:
            position_size = self.calculate_position_size(confidence=0.5)
            
            if action == 2:  # BUY
                if not self.position or self.position.is_short:
                    self.buy(size=position_size)
                    self.entry_price = self.data.Close[-1]
                    self.position_bars = 0
            elif action == 0:  # SELL
                if not self.position or self.position.is_long:
                    self.sell(size=position_size)
                    self.entry_price = self.data.Close[-1]
                    self.position_bars = 0
            # action == 1 is HOLD
        
        # Calculate reward and store experience
        reward = self.calculate_reward(prev_equity)
        next_state = self.get_state()
        done = False  # Set to True at end of episode if needed
        
        self.store_experience(state, action, reward, next_state, done)
        
        # Train periodically
        if self.total_steps % 4 == 0:
            self.train()
        
        # Update target network
        if self.total_steps % self.tau == 0:
            self.update_target()
            print(f"Step {self.total_steps}: Updated target network | Epsilon: {self.current_epsilon:.4f} | Equity: {self.equity:.2f}")

class LTSM_MA_Strategy(Strategy):

    n1 = 50          # MA Cepat
    n2 = 100         # MA Lambat
    adx_threshold = 25
    
    def init(self):
        # 1. Hitung Indikator Teknikal menggunakan pandas_ta
        close = pd.Series(self.data.Close)
        high = pd.Series(self.data.High)
        low = pd.Series(self.data.Low)

        self.ma50 = self.I(ta.sma, close, self.n1)
        self.ma100 = self.I(ta.sma, close, self.n2)
        
        # ADX
        adx_data = ta.adx(high, low, close, length=14)
        self.adx = self.I(lambda: adx_data['ADX_14'])

        # Supertrend
        st_data = ta.supertrend(high, low, close, length=7, multiplier=3.0)
        self.st_val = self.I(lambda: st_data['SUPERT_7_3.0'])
        self.st_dir = self.I(lambda: st_data['SUPERTd_7_3.0'])

        # 2. Load Model LSTM (Asumsi model sudah di-train di luar class)
        # Kita simpan prediksi dalam array agar bisa diakses per bar
        self.predictions = self.I(lambda: self.data.preds) 

    def next(self):
        price = self.data.Close[-1]
        
        # Logika Trailing Stop: Jika ada posisi terbuka, update SL ke Supertrend
        if self.position:
            # Update SL mengikuti nilai Supertrend (Trailing)
            # Supertrend secara otomatis berfungsi sebagai pelindung profit
            self.position.sl = self.st_val[-1]
        
        if not self.position:
            if (self.predictions[-1] > price and 
                self.ma50[-1] > self.ma100[-1] and 
                self.adx[-1] > self.adx_threshold and 
                self.st_dir[-1] == 1):
                
                # Hitung SL awal dan TP (RR 1:2)
                sl_level = self.st_val[-1]
                risk = price - sl_level
                tp_level = price + (risk * 2)
                
                # Eksekusi Buy
                self.buy(sl=sl_level, tp=tp_level)



class CPOStrategy(Strategy):
    
    # Parameters
    # atr_period = 14
    # adx_period = 14
    # rsi_period = 14
    # ema_fast = 50
    # ema_slow = 100
    # bb_period = 20
    # bb_std = 2
    # volume_ma_period = 20

    atr_period = 14

    adx_period = 14
    adx_threshold = 20

    rsi_period = 7

    ema_fast = 20
    ema_slow = 50

    bb_period = 20
    bb_std = 2  # opsional

    volume_ma_period = 20
    volume_threshold = 1.2

    
    def init(self):
        # Convert OHLCV to pandas DataFrame
        df = pd.DataFrame({
            'open': self.data.Open,
            'high': self.data.High,
            'low': self.data.Low,
            'close': self.data.Close,
            'volume': self.data.Volume
        })
        
        # ==========================================
        # INDICATORS
        # ==========================================
        
        # ATR
        atr_series = ta.atr(df['high'], df['low'], df['close'], length=self.atr_period)
        self.atr = self.I(lambda x=atr_series: x, name='ATR')
        
        # ADX
        adx_df = ta.adx(df['high'], df['low'], df['close'], length=self.adx_period)
        self.adx = self.I(lambda x=adx_df: x.iloc[:, 0], name='ADX')
        self.di_plus = self.I(lambda x=adx_df: x.iloc[:, 1], name='DI+')
        self.di_minus = self.I(lambda x=adx_df: x.iloc[:, 2], name='DI-')
        
        # RSI
        rsi_series = ta.rsi(df['close'], length=self.rsi_period)
        self.rsi = self.I(lambda x=rsi_series: x, name='RSI')
        
        # EMA
        ema_fast_series = ta.ema(df['close'], length=self.ema_fast)
        ema_slow_series = ta.ema(df['close'], length=self.ema_slow)
        self.ema_f = self.I(lambda x=ema_fast_series: x, name='EMA_Fast')
        self.ema_s = self.I(lambda x=ema_slow_series: x, name='EMA_Slow')
        
        # Bollinger Bands
        bbands = ta.bbands(df['close'], length=self.bb_period, std=self.bb_std)
        self.bb_upper = self.I(lambda x=bbands: x.iloc[:, 0], name='BB_Upper')
        self.bb_middle = self.I(lambda x=bbands: x.iloc[:, 1], name='BB_Middle')
        self.bb_lower = self.I(lambda x=bbands: x.iloc[:, 2], name='BB_Lower')
        
        # Volume SMA
        volume_sma = ta.sma(df['volume'], length=self.volume_ma_period)
        self.volume_ma = self.I(lambda x=volume_sma: x, name='Volume_MA')
        
        # MACD
        macd_df = ta.macd(df['close'])
        self.macd = self.I(lambda x=macd_df: x.iloc[:, 0], name='MACD')
        self.macd_signal = self.I(lambda x=macd_df: x.iloc[:, 1], name='MACD_Signal')
        self.macd_hist = self.I(lambda x=macd_df: x.iloc[:, 2], name='MACD_Hist')
        
        # Stochastic
        stoch_df = ta.stoch(df['high'], df['low'], df['close'])
        self.stoch_k = self.I(lambda x=stoch_df: x.iloc[:, 0], name='Stoch_K')
        self.stoch_d = self.I(lambda x=stoch_df: x.iloc[:, 1], name='Stoch_D')
        
        self.current_regime = None
        
    def next(self):
        # Skip jika sudah ada posisi
        if self.position:
            self.manage_position()
            return
        
        # Skip jika data belum cukup (warmup period)
        if len(self.data.Close) < 50:
            return
        
        # Detect market regime
        regime = self.detect_market_regime()
        self.current_regime = regime
        
        # Entry logic
        if regime == "trending":
            self.trending_entry()
        elif regime == "ranging":
            self.ranging_entry()
        elif regime == "volatile":
            self.volatile_entry()
    
    # ==========================================
    # REGIME DETECTION
    # ==========================================
    
    def detect_market_regime(self):
        try:
            adx = self.adx[-1]
            atr = self.atr[-1]
            
            if pd.isna(adx) or pd.isna(atr):
                return "neutral"
            
            atr_ma = np.nanmean(self.atr[-20:])
            
            if adx > 25:
                return "trending"
            elif atr > atr_ma * 1.5:
                return "volatile"
            elif adx < 20:
                return "ranging"
            else:
                return "neutral"
        except:
            return "neutral"
    
    # ==========================================
    # ENTRY SIGNALS
    # ==========================================
    def trending_entry(self):
        try:
            price = self.data.Close[-1]
            
            if any(pd.isna(x[-1]) for x in [self.ema_f, self.ema_s, self.di_plus, self.di_minus, self.rsi]):
                return
            
            uptrend = (self.ema_f[-1] > self.ema_s[-1] and 
                       self.di_plus[-1] > self.di_minus[-1])
            downtrend = (self.ema_f[-1] < self.ema_s[-1] and 
                         self.di_minus[-1] > self.di_plus[-1])
            
            if len(self.ema_f) < 3:
                return
                
            ema_cross_up = (self.ema_f[-1] > self.ema_s[-1] and 
                            self.ema_f[-3] <= self.ema_s[-3])
            ema_cross_down = (self.ema_f[-1] < self.ema_s[-1] and 
                              self.ema_f[-3] >= self.ema_s[-3])
            
            macd_bullish = self.macd[-1] > self.macd_signal[-1] and self.macd_hist[-1] > 0
            macd_bearish = self.macd[-1] < self.macd_signal[-1] and self.macd_hist[-1] < 0
            
            volume_surge = self.data.Volume[-1] > self.volume_ma[-1] * 1.2
            
            # LONG Entry
            if uptrend and (ema_cross_up or macd_bullish) and volume_surge:
                if price <= self.ema_f[-1] * 1.015 and self.rsi[-1] < 70:
                    self.enter_long("trending")
            
            # SHORT Entry
            elif downtrend and (ema_cross_down or macd_bearish) and volume_surge:
                if price >= self.ema_f[-1] * 0.985 and self.rsi[-1] > 30:
                    self.enter_short("trending")
        except Exception as e:
            print(f"Error in trending_entry: {e}")
    
    def ranging_entry(self):
        try:
            price = self.data.Close[-1]
            open_curr = self.data.Open[-1]
            
            if any(pd.isna(x[-1]) for x in [self.rsi, self.stoch_k, self.bb_lower, self.bb_upper]):
                return
            
            rsi_oversold = self.rsi[-1] < 30
            stoch_oversold = self.stoch_k[-1] < 20
            price_below_bb = price < self.bb_lower[-1]
            
            bullish_rejection = (self.data.Low[-1] <= self.bb_lower[-1] and 
                                price > self.bb_lower[-1] and
                                price > open_curr)
            
            rsi_overbought = self.rsi[-1] > 70
            stoch_overbought = self.stoch_k[-1] > 80
            price_above_bb = price > self.bb_upper[-1]
            
            bearish_rejection = (self.data.High[-1] >= self.bb_upper[-1] and 
                                price < self.bb_upper[-1] and
                                price < open_curr)
            
            if len(self.stoch_k) < 2:
                return
                
            stoch_cross_up = self.stoch_k[-1] > self.stoch_d[-1] and self.stoch_k[-2] <= self.stoch_d[-2]
            stoch_cross_down = self.stoch_k[-1] < self.stoch_d[-1] and self.stoch_k[-2] >= self.stoch_d[-2]
            
            # LONG Entry
            if (rsi_oversold and stoch_oversold and price_below_bb) or bullish_rejection:
                if stoch_cross_up or bullish_rejection:
                    self.enter_long("ranging")
            
            # SHORT Entry
            elif (rsi_overbought and stoch_overbought and price_above_bb) or bearish_rejection:
                if stoch_cross_down or bearish_rejection:
                    self.enter_short("ranging")
        except Exception as e:
            print(f"Error in ranging_entry: {e}")
    
    def volatile_entry(self):
        try:
            price = self.data.Close[-1]
            
            if len(self.data.Close) < 21:
                return
            
            high_20 = max([self.data.High[i] for i in range(-20, 0)])
            low_20 = min([self.data.Low[i] for i in range(-20, 0)])
            
            volume_surge = self.data.Volume[-1] > self.volume_ma[-1] * 1.5
            
            if pd.isna(self.rsi[-1]) or pd.isna(self.macd[-1]):
                return
            
            strong_rsi_up = self.rsi[-1] > 60
            strong_rsi_down = self.rsi[-1] < 40
            
            if len(self.macd_hist) < 2:
                return
                
            macd_momentum_up = (self.macd_hist[-1] > self.macd_hist[-2] and 
                               self.macd[-1] > self.macd_signal[-1])
            macd_momentum_down = (self.macd_hist[-1] < self.macd_hist[-2] and 
                                 self.macd[-1] < self.macd_signal[-1])
            
            price_surge_up = price > self.data.Close[-2] * 1.015
            price_surge_down = price < self.data.Close[-2] * 0.985
            
            # LONG Entry
            if (price > high_20 and volume_surge and strong_rsi_up and 
                macd_momentum_up and price_surge_up):
                if self.data.Close[-2] < high_20:
                    self.enter_long("volatile")
            
            # SHORT Entry
            elif (price < low_20 and volume_surge and strong_rsi_down and 
                  macd_momentum_down and price_surge_down):
                if self.data.Close[-2] > low_20:
                    self.enter_short("volatile")
        except Exception as e:
            print(f"Error in volatile_entry: {e}")
    
    # ==========================================
    # ENTRY EXECUTION - FIX POSITION SIZING
    # ==========================================
    
    def enter_long(self, regime):
        try:
            # Calculate stop loss price first
            sl_distance = self.stop_loss(regime)
            sl_price = self.data.Close[-1] - sl_distance
            
            # Calculate position size
            size = self.position_sizing(regime, sl_distance)
            
            # Validate size
            if size <= 0 or pd.isna(size):
                print(f"❌ Invalid size for LONG: {size}")
                return
            
            # Calculate take profit
            tp_multiplier = self.take_profit(regime)
            tp_price = self.data.Close[-1] + (sl_distance * tp_multiplier) if tp_multiplier else None
            
            # Execute
            self.buy(size=size, sl=sl_price, tp=tp_price)
            
            print(f"🟢 LONG [{regime}] @ {self.data.Close[-1]:.2f} | "
                  f"Size: {size:.6f} | SL: {sl_price:.2f} | TP: {tp_price}")
        except Exception as e:
            print(f"Error in enter_long: {e}")
            import traceback
            traceback.print_exc()
    
    def enter_short(self, regime):
        try:
            sl_distance = self.stop_loss(regime)
            sl_price = self.data.Close[-1] + sl_distance
            
            size = self.position_sizing(regime, sl_distance)
            
            if size <= 0 or pd.isna(size):
                print(f"❌ Invalid size for SHORT: {size}")
                return
            
            tp_multiplier = self.take_profit(regime)
            tp_price = self.data.Close[-1] - (sl_distance * tp_multiplier) if tp_multiplier else None
            
            self.sell(size=size, sl=sl_price, tp=tp_price)
            
            print(f"🔴 SHORT [{regime}] @ {self.data.Close[-1]:.2f} | "
                  f"Size: {size:.6f} | SL: {sl_price:.2f} | TP: {tp_price}")
        except Exception as e:
            print(f"Error in enter_short: {e}")
            import traceback
            traceback.print_exc()
    
    # ==========================================
    # POSITION MANAGEMENT
    # ==========================================
    
    def manage_position(self):
        """Manage open positions dengan trailing stop"""
        try:
            if not self.position:
                return
            
            # Check if position exists and has trades
            if not self.trades:
                return
            
            regime = self.current_regime if self.current_regime else "neutral"
            trail_distance = self.trailing_stop(regime)
            
            current_price = self.data.Close[-1]
            
            # Get current trade (last opened trade)
            trade = self.trades[-1]
            
            # Update trailing stop untuk LONG positions
            if self.position.is_long:
                # Calculate new stop loss
                new_sl = current_price - trail_distance
                
                # Only update if new SL is higher (trailing up)
                if hasattr(trade, 'sl') and trade.sl is not None:
                    if new_sl > trade.sl:
                        trade.sl = new_sl
                        print(f"📈 Trailing LONG SL updated: {new_sl:.2f}")
                else:
                    # Set initial SL if not set
                    trade.sl = new_sl
            
            # Update trailing stop untuk SHORT positions
            elif self.position.is_short:
                # Calculate new stop loss
                new_sl = current_price + trail_distance
                
                # Only update if new SL is lower (trailing down)
                if hasattr(trade, 'sl') and trade.sl is not None:
                    if new_sl < trade.sl:
                        trade.sl = new_sl
                        print(f"📉 Trailing SHORT SL updated: {new_sl:.2f}")
                else:
                    # Set initial SL if not set
                    trade.sl = new_sl
                    
        except Exception as e:
            print(f"Error in manage_position: {e}")
            import traceback
            traceback.print_exc()
    
    # ==========================================
    # RISK MANAGEMENT - FIXED
    # ==========================================
    
    def position_sizing(self, regime, sl_distance):
        """Fixed position sizing - return fraction of equity"""
        try:
            base_risk_pct = 0.015  # 1.5% risk per trade
            
            # Regime adjustment
            regime_multipliers = {
                "volatile": 0.5,    # 1% risk
                "ranging": 1.0,     # 2% risk
                "trending": 1.25,   # 2.5% risk
                "neutral": 0.75     # 1.5% risk
            }
            regime_mult = regime_multipliers.get(regime, 0.75)
            
            # Drawdown adjustment
            dd = self.drawdown_detection()
            if dd > 0.15:
                dd_mult = 0.5
            elif dd > 0.10:
                dd_mult = 0.75
            else:
                dd_mult = 1.0
            
            # Final risk percentage (max capital to risk)
            final_risk_pct = base_risk_pct * regime_mult * dd_mult
            
            # Calculate position size as FRACTION OF EQUITY
            # Formula: (risk% / (SL% from entry))
            price = self.data.Close[-1]
            sl_pct = sl_distance / price  # SL as percentage of price
            
            if sl_pct <= 0 or sl_pct > 0.5:  # Max 50% SL
                print(f"❌ Invalid SL%: {sl_pct:.4f}")
                return 0
            
            # Position size as fraction of equity
            # If we risk 2% of equity with 3% SL, position = 2/3 = 0.67 = 67% of equity
            position_fraction = final_risk_pct / sl_pct
            
            # Cap at 95% of equity (leave some buffer)
            position_fraction = min(position_fraction, 0.95)
            
            # Validate
            if position_fraction <= 0 or position_fraction > 1 or pd.isna(position_fraction):
                print(f"❌ Invalid position fraction: {position_fraction}")
                return 0
            
            print(f"📊 Position Sizing: Price={price:.2f}, SL%={sl_pct*100:.2f}%, "
                f"Risk={final_risk_pct*100:.2f}%, Fraction={position_fraction:.4f} ({position_fraction*100:.1f}%)")
            
            return position_fraction
            
        except Exception as e:
            print(f"Error in position_sizing: {e}")
            import traceback
            traceback.print_exc()
            return 0
    
    def stop_loss(self, regime):
        """Calculate stop loss distance in price units"""
        try:
            atr = self.atr[-1]
            
            # Fallback if ATR is invalid
            if pd.isna(atr) or atr <= 0:
                atr = self.data.Close[-1] * 0.02  # 2% of price
                print(f"⚠️  Using fallback ATR: {atr}")
            
            multipliers = {
                "trending": 2.5,
                "ranging": 1.5,
                "volatile": 3.0,
                "neutral": 2.0
            }
            
            sl_distance = multipliers.get(regime, 2.0) * atr
            
            print(f"🛡️  SL Distance [{regime}]: {sl_distance:.2f} (ATR: {atr:.2f})")
            
            return sl_distance
            
        except Exception as e:
            print(f"Error in stop_loss: {e}")
            # Absolute fallback
            return self.data.Close[-1] * 0.02
    
    def take_profit(self, regime):
        """R:R multiplier"""
        if regime == "trending":
            return None  # Let it run
        elif regime == "ranging":
            return 1.5
        elif regime == "volatile":
            return 2.0
        else:
            return 2.0
    
    def trailing_stop(self, regime):
        """Trailing stop distance"""
        try:
            atr = self.atr[-1]
            
            if pd.isna(atr) or atr <= 0:
                atr = self.data.Close[-1] * 0.02
            
            multipliers = {
                "trending": 2.0,
                "ranging": 1.0,
                "volatile": 2.5,
                "neutral": 1.5
            }
            
            return multipliers.get(regime, 1.5) * atr
            
        except:
            return self.data.Close[-1] * 0.015
    
    def kelly_fraction(self):
        """Kelly Criterion - simplified"""
        try:
            if len(self.closed_trades) < 10:
                return 0.02
            
            wins = [t.pl for t in self.closed_trades if t.pl > 0]
            losses = [abs(t.pl) for t in self.closed_trades if t.pl < 0]
            
            if not wins or not losses:
                return 0.02
            
            win_rate = len(wins) / len(self.closed_trades)
            avg_win = np.mean(wins)
            avg_loss = np.mean(losses)
            
            kelly = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
            
            # Cap between 1-5%
            return max(0.01, min(kelly, 0.05))
        except:
            return 0.02
    
    def drawdown_detection(self):
        """
        Calculate current drawdown based on true equity curve.
        Return value between 0.0 - 1.0
        """
        try:
            # 1. Bangun equity curve dari MODAL AWAL
            equity = self.equity  # Equity saat in
            peak_equity = self.equity

            # 2. Iterasi trade tertutup secara kronologis
            for trade in self.closed_trades:
                equity += trade.pl

                if equity > peak_equity:
                    peak_equity = equity

            # 3. Tambahkan floating P/L dari open trades
            floating_pl = 0.0
            for trade in self.trades.index:
                floating_pl += trade

            current_equity = equity + floating_pl

            # 4. Update peak jika floating membuat equity ATH
            if current_equity > peak_equity:
                peak_equity = current_equity

            # 5. Hitung drawdown
            if peak_equity <= 0:
                return 0.0

            drawdown = (peak_equity - current_equity) / peak_equity

            return max(0.0, drawdown)

        except Exception as e:
            # Jangan silent fail
            print(f"[Drawdown Error] {e}")
            return 0.0
