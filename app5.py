from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from datetime import datetime
import sys
import os
import json
import time
import requests

app = Flask(__name__)
CORS(app)

# =============================================================================
# REMOTE API CONFIGURATION
# =============================================================================

# Your Render API URL (update this after deployment)
TRADING_API_URL = os.getenv('TRADING_API_URL', 'http://localhost:8000')
API_KEY = os.getenv('TRADING_API_KEY', None)  # Optional: if you add authentication

# Global variables
T_indicators = ['HA_signal', 'MACD_signal', 'MA_signal', 'OBV_signal']
MR_indicators = ['RSI_signal', 'STOCH_signal', 'BBANDS_signal', 'CCI_signal']
continuous_features = [
    'RSI', 'CMF', 'VWAP', 'ATR', 'VOLATILITY', 
    'PARKINSON', 'dist_from_high', 'dist_from_low', 'PRICE_ACTION',
    'Taker Buy Quote', 'Taker Buy Base', 'Number of Trades', 'Quote Asset Volume'
]


# =============================================================================
# REMOTE API CLIENT
# =============================================================================

class RemoteTradingAPIClient:
    """Client for interacting with the remote Trading Environment API"""
    
    def __init__(self, base_url, api_key=None):
        self.base_url = base_url.rstrip('/')
        self.headers = {'Content-Type': 'application/json'}
        if api_key:
            self.headers['X-API-Key'] = api_key
    
    def health_check(self):
        """Check if remote API is running"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def create_environment(self, csv_file_path, initial_balance=10000):
        """Create a new trading environment on remote server"""
        config = {
            "T_indicators": T_indicators,
            "MR_indicators": MR_indicators,
            "continuous_features": continuous_features,
            "initial_balance": initial_balance
        }
        
        with open(csv_file_path, 'rb') as f:
            files = {'file': f}
            data = {'config': json.dumps(config)}
            response = requests.post(
                f"{self.base_url}/create_environment",
                files=files,
                data=data,
                headers={'X-API-Key': self.headers.get('X-API-Key', '')} if 'X-API-Key' in self.headers else {}
            )
        
        response.raise_for_status()
        result = response.json()
        return result['env_id']
    
    def reset_environment(self, env_id, seed=None):
        """Reset environment to initial state"""
        payload = {"env_id": env_id}
        if seed is not None:
            payload["seed"] = seed
        
        response = requests.post(
            f"{self.base_url}/reset",
            json=payload,
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()
    
    def step(self, env_id, action):
        """Execute one step in the environment"""
        payload = {
            "env_id": env_id,
            "action": action
        }
        
        response = requests.post(
            f"{self.base_url}/step",
            json=payload,
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()
    
    def get_environment_info(self, env_id):
        """Get detailed information about an environment"""
        response = requests.get(
            f"{self.base_url}/environment/{env_id}/info",
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()
    
    def get_positions(self, env_id):
        """Get current positions in the environment"""
        response = requests.post(
            f"{self.base_url}/environment/{env_id}/positions",
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()
    
    def delete_environment(self, env_id):
        """Delete an environment"""
        response = requests.delete(
            f"{self.base_url}/environment/{env_id}",
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()


# =============================================================================
# TRADING ANALYZER
# =============================================================================

class TradingAnalyzer:
    """Collects trading data and calculates metrics"""
    
    def __init__(self, initial_balance=10000):
        self.initial_balance = initial_balance
        self.history = []
    
    def record_step(self, step, timestamp, info, reward):
        """Record step data"""
        self.history.append({
            'step': step,
            'timestamp': timestamp,
            'info': info,
            'reward': reward
        })
    
    def extract_trades_from_history(self):
        """Extract completed trades from history"""
        trades = []
        
        for record in self.history:
            info = record['info']
            timestamp = record['timestamp']
            
            if info and 'position_changes' in info and info['position_changes']:
                position_changes = info['position_changes']
                
                if 'closed' in position_changes and position_changes['closed']:
                    for closed_pos in position_changes['closed']:
                        trade = {
                            'timestamp': timestamp,
                            'step': record['step'],
                            'entry_price': closed_pos.get('entry_price', 0),
                            'exit_price': closed_pos.get('exit_price', 0),
                            'quantity': closed_pos.get('quantity', 0),
                            'pnl': closed_pos.get('pnl', 0),
                            'pnl_percent': closed_pos.get('pnl_percent', 0),
                            'holding_period': closed_pos.get('holding_period', 0),
                            'close_reason': closed_pos.get('close_reason', 'unknown'),
                            'portfolio_value': info.get('portfolio_value', 0)
                        }
                        trades.append(trade)
        
        return trades
    
    def save_trades_to_csv(self, trades, save_dir='backtest_results'):
        """Save trades to CSV"""
        if not trades:
            return None
        
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = os.path.join(save_dir, f'trades_{timestamp}.csv')
        
        try:
            trades_df = pd.DataFrame(trades)
            trades_df.insert(0, 'trade_number', range(1, len(trades_df) + 1))
            trades_df['win_loss'] = trades_df['pnl'].apply(
                lambda x: 'WIN' if x > 0 else 'LOSS' if x < 0 else 'BREAKEVEN'
            )
            
            trades_df.to_csv(filename, index=False)
            print(f"✓ Trades saved to: {filename}")
            return filename
        except Exception as e:
            print(f"❌ Error saving trades: {str(e)}")
            return None
    
    def calculate_metrics(self):
        """Calculate trading metrics"""
        portfolio_values = [h['info']['portfolio_value'] for h in self.history if h['info']]
        
        final_value = portfolio_values[-1] if portfolio_values else self.initial_balance
        total_return = final_value - self.initial_balance
        total_return_pct = (total_return / self.initial_balance) * 100
        
        trades = self.extract_trades_from_history()
        num_trades = len(trades)
        winning_trades = [t for t in trades if t['pnl'] > 0]
        losing_trades = [t for t in trades if t['pnl'] < 0]
        
        win_rate = (len(winning_trades) / num_trades * 100) if num_trades > 0 else 0.0
        total_pnl = sum(t['pnl'] for t in trades) if trades else 0.0
        avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0.0
        avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0.0
        
        total_wins = sum(t['pnl'] for t in winning_trades) if winning_trades else 0.0
        total_losses = abs(sum(t['pnl'] for t in losing_trades)) if losing_trades else 0.0
        profit_factor = (total_wins / total_losses) if total_losses > 0 else 0.0
        
        expectancy = (win_rate/100 * avg_win) + ((1 - win_rate/100) * avg_loss)
        total_reward = sum(h['reward'] for h in self.history)
        
        metrics = {
            'Initial Balance': f'${self.initial_balance:,.2f}',
            'Final Balance': f'${final_value:,.2f}',
            'Total Return': f'${total_return:,.2f}',
            'Total Return (%)': f'{total_return_pct:.2f}%',
            'Total Trades': f'{num_trades:,}',
            'Winning Trades': f'{len(winning_trades):,}',
            'Losing Trades': f'{len(losing_trades):,}',
            'Win Rate': f'{win_rate:.2f}%',
            'Total PnL': f'${total_pnl:,.2f}',
            'Avg Win': f'${avg_win:.2f}',
            'Avg Loss': f'${avg_loss:.2f}',
            'Profit Factor': f'{profit_factor:.3f}',
            'Expectancy': f'${expectancy:.2f}',
            'Total Steps': f'{len(self.history):,}',
            'Total Reward': f'{total_reward:.2f}'
        }
        
        return metrics, trades


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def live_indicators(csv_path):
    """Load and process the CSV file with technical indicators"""
    from indicators import TechnicalIndicators
    
    ti = TechnicalIndicators()
    df = ti.calculate_from_csv(csv_path)
    return df


def save_filtered_csv(df, save_path='temp_backtest_data.csv'):
    """Save filtered dataframe temporarily for API upload"""
    df.to_csv(save_path, index=False)
    return save_path


# =============================================================================
# FLASK ROUTES
# =============================================================================

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    # Also check if remote API is available
    client = RemoteTradingAPIClient(TRADING_API_URL, API_KEY)
    remote_status = client.health_check()
    
    return jsonify({
        'status': 'ok', 
        'message': 'Flask backend is running',
        'remote_api_status': 'connected' if remote_status else 'disconnected',
        'remote_api_url': TRADING_API_URL
    })


@app.route('/api/cryptocurrencies', methods=['GET'])
def get_cryptocurrencies():
    """Return list of available cryptocurrencies"""
    try:
        cryptos = ['XRPJPY', 'LINKJPY', 'AVAXTRY', 'ADAJPY']
        return jsonify({'cryptocurrencies': cryptos})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/timerange', methods=['POST'])
def get_timerange():
    """Get min and max timestamps for selected cryptocurrency"""
    try:
        data = request.json
        crypto = data.get('crypto')
        
        if not crypto:
            return jsonify({'error': 'Cryptocurrency not specified'}), 400
        
        df = pd.read_csv('best_cluster_similar_price.csv')
        df_crypto = df[df['cryptocoin'] == crypto].copy()
        
        if df_crypto.empty:
            return jsonify({'error': f'No data found for {crypto}'}), 404
        
        df_crypto['Open Time'] = pd.to_datetime(df_crypto['Open Time'])
        min_time = df_crypto['Open Time'].min()
        max_time = df_crypto['Open Time'].max()
        
        return jsonify({
            'crypto': crypto,
            'min_timestamp': min_time.strftime('%Y-%m-%d %H:%M:%S'),
            'max_timestamp': max_time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_rows': len(df_crypto)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/backtest/stream', methods=['POST'])
def run_backtest_stream():
    """Run backtest using REMOTE API with streaming updates"""
    def generate():
        client = RemoteTradingAPIClient(TRADING_API_URL, API_KEY)
        env_id = None
        analyzer = None
        
        try:
            data = request.json
            crypto = data.get('crypto')
            start_time = data.get('start_time')
            end_time = data.get('end_time')
            
            if not all([crypto, start_time, end_time]):
                yield f"data: {json.dumps({'type': 'error', 'message': 'Missing required parameters'})}\n\n"
                return
            
            # Check remote API connection
            yield f"data: {json.dumps({'type': 'info', 'message': 'Checking remote API connection...'})}\n\n"
            
            if not client.health_check():
                yield f"data: {json.dumps({'type': 'error', 'message': f'Cannot connect to remote API at {TRADING_API_URL}'})}\n\n"
                return
            
            yield f"data: {json.dumps({'type': 'info', 'message': f'Connected to remote API: {TRADING_API_URL}'})}\n\n"
            yield f"data: {json.dumps({'type': 'info', 'message': 'Loading data...'})}\n\n"
            
            # Load and filter data
            df = live_indicators('best_cluster_similar_price.csv')
            df.dropna(inplace=True)
            df_crypto = df[df['cryptocoin'] == crypto].copy()
            
            if df_crypto.empty:
                yield f"data: {json.dumps({'type': 'error', 'message': f'No data found for {crypto}'})}\n\n"
                return
            
            df_crypto['Open Time'] = pd.to_datetime(df_crypto['Open Time'])
            start_dt = pd.to_datetime(start_time)
            end_dt = pd.to_datetime(end_time)
            
            df_test = df_crypto[(df_crypto['Open Time'] >= start_dt) & 
                               (df_crypto['Open Time'] <= end_dt)].copy()
            
            if df_test.empty:
                yield f"data: {json.dumps({'type': 'error', 'message': 'No data in selected time range'})}\n\n"
                return
            
            if len(df_test) < 100:
                yield f"data: {json.dumps({'type': 'error', 'message': 'Insufficient data points (need at least 100)'})}\n\n"
                return
            
            df_test = df_test.reset_index(drop=True)
            total_steps = len(df_test)
            
            yield f"data: {json.dumps({'type': 'init', 'message': 'Initializing backtest...', 'total_steps': total_steps})}\n\n"
            
            # Save filtered data temporarily
            temp_csv = save_filtered_csv(df_test)
            
            # Create environment on remote server
            yield f"data: {json.dumps({'type': 'info', 'message': 'Creating remote trading environment...'})}\n\n"
            
            initial_balance = 10000
            env_id = client.create_environment(temp_csv, initial_balance=initial_balance)
            
            yield f"data: {json.dumps({'type': 'info', 'message': f'Environment created: {env_id[:8]}...'})}\n\n"
            
            # Initialize analyzer
            analyzer = TradingAnalyzer(initial_balance=initial_balance)
            
            # Reset environment
            yield f"data: {json.dumps({'type': 'info', 'message': 'Resetting environment...'})}\n\n"
            reset_result = client.reset_environment(env_id)
            obs = reset_result['observation']
            
            # Load model
            yield f"data: {json.dumps({'type': 'info', 'message': 'Loading AI model...'})}\n\n"
            model = PPO.load("ppo_trading_bot_enhanced.zip")
            
            yield f"data: {json.dumps({'type': 'info', 'message': 'Starting backtest...'})}\n\n"
            
            done = False
            step_count = 0
            
            # Run backtest using remote API
            while not done and step_count < total_steps:
                # Predict action locally
                action, _ = model.predict(np.array(obs), deterministic=False)
                
                # Send action to remote environment
                step_result = client.step(env_id, action.tolist())
                
                obs = step_result['observation']
                reward = step_result['reward']
                done = step_result['terminated'] or step_result['truncated']
                info = step_result['info']
                
                # Get current timestamp
                current_timestamp = df_test.iloc[min(step_count, len(df_test) - 1)]['Open Time'].strftime('%Y-%m-%d %H:%M:%S')
                
                # Record step
                analyzer.record_step(
                    step=step_count,
                    timestamp=current_timestamp,
                    info=info,
                    reward=reward
                )
                
                step_count += 1
                
                # Stream progress every 10 steps
                if step_count % 10 == 0:
                    current_value = info['portfolio_value']
                    current_pnl = current_value - initial_balance
                    
                    step_data = {
                        'type': 'step',
                        'step': step_count,
                        'total_steps': total_steps,
                        'portfolio_value': float(current_value),
                        'initial_balance': float(initial_balance),
                        'pnl': float(current_pnl),
                        'timestamp': current_timestamp,
                        'reward': float(reward)
                    }
                    yield f"data: {json.dumps(step_data)}\n\n"
            
            # Calculate metrics
            yield f"data: {json.dumps({'type': 'info', 'message': 'Calculating metrics...'})}\n\n"
            
            final_metrics, trades = analyzer.calculate_metrics()
            
            # Save trades to CSV
            trades_csv_file = None
            if len(trades) > 0:
                yield f"data: {json.dumps({'type': 'info', 'message': f'Saving {len(trades)} trades to CSV...'})}\n\n"
                trades_csv_file = analyzer.save_trades_to_csv(trades)
            
            # Prepare completion data
            final_value = analyzer.history[-1]['info']['portfolio_value'] if analyzer.history else initial_balance
            total_return = ((final_value - initial_balance) / initial_balance) * 100
            total_reward = sum(h['reward'] for h in analyzer.history)
            
            completion_data = {
                'type': 'complete',
                'results': {
                    'crypto': crypto,
                    'start_time': start_time,
                    'end_time': end_time,
                    'steps': step_count,
                    'initial_balance': float(initial_balance),
                    'final_balance': float(final_value),
                    'total_pnl': float(final_value - initial_balance),
                    'total_return': round(total_return, 2),
                    'total_reward': round(total_reward, 2),
                    'num_trades': len(trades),
                    'win_rate': round((len([t for t in trades if t['pnl'] > 0]) / len(trades) * 100) if trades else 0, 2),
                    'trades_csv_saved': trades_csv_file,
                    'env_id': env_id
                }
            }
            yield f"data: {json.dumps(completion_data)}\n\n"
            
            # Cleanup
            if os.path.exists(temp_csv):
                os.remove(temp_csv)
            
            # Delete remote environment
            yield f"data: {json.dumps({'type': 'info', 'message': 'Cleaning up remote environment...'})}\n\n"
            client.delete_environment(env_id)
        
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            print(error_trace)
            
            # Cleanup on error
            if env_id:
                try:
                    client.delete_environment(env_id)
                except:
                    pass
            
            error_data = {
                'type': 'error',
                'message': str(e),
                'trace': error_trace
            }
            yield f"data: {json.dumps(error_data)}\n\n"
    
    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
            'Connection': 'keep-alive'
        }
    )


if __name__ == '__main__':
    # Check environment variables
    print(f"Trading API URL: {TRADING_API_URL}")
    if API_KEY:
        print(f"API Key configured: Yes")
    else:
        print(f"API Key configured: No (optional)")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
