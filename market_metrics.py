import pandas as pd
import numpy as np
import json
from datetime import datetime

class MetricsRecalculator:
    """Recalculate trading metrics correctly from saved data"""
    
    def __init__(self, csv_path=None, json_path=None):
        """
        Load data from either CSV or JSON file
        csv_path: Path to CSV with columns: timestamp, portfolio_value, action, reward, positions
        json_path: Path to JSON with analyzer data
        """
        if csv_path:
            self.df = pd.read_csv(csv_path)
            self.portfolio_values = self.df['portfolio_value'].values
            self.actions = self.df['action'].values
            self.rewards = self.df['reward'].values
            self.initial_balance = self.portfolio_values[0]
        elif json_path:
            with open(json_path, 'r') as f:
                data = json.load(f)
            self.portfolio_values = np.array(data['portfolio_values'])
            self.actions = np.array(data['actions'])
            self.rewards = np.array(data['rewards'])
            self.initial_balance = data.get('initial_balance', 10000)
        else:
            raise ValueError("Provide either csv_path or json_path")
    
    def identify_actual_trades(self):
        """Identify actual trades (when position changes)"""
        trades = []
        in_position = False
        entry_price = 0
        entry_idx = 0
        
        for i in range(len(self.actions)):
            action = self.actions[i]
            portfolio_value = self.portfolio_values[i]
            
            # Action 0 = hold, 1 = buy, 2 = sell (adjust based on your action space)
            if action == 1 and not in_position:  # Buy/Enter
                in_position = True
                entry_price = portfolio_value
                entry_idx = i
            elif action == 2 and in_position:  # Sell/Exit
                exit_price = portfolio_value
                pnl = exit_price - entry_price
                pnl_pct = (pnl / entry_price) * 100
                
                trades.append({
                    'entry_idx': entry_idx,
                    'exit_idx': i,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'win': pnl > 0
                })
                in_position = False
        
        return pd.DataFrame(trades) if trades else pd.DataFrame()
    
    def calculate_correct_metrics(self):
        """Calculate metrics based on actual trades"""
        portfolio_array = np.array(self.portfolio_values)
        final_value = portfolio_array[-1]
        
        # Overall portfolio metrics
        total_return = ((final_value - self.initial_balance) / self.initial_balance) * 100
        
        # Calculate returns (step-by-step for Sharpe/Sortino)
        returns = np.diff(portfolio_array) / portfolio_array[:-1]
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        # Sharpe Ratio (annualized, assuming 365 periods per year)
        sharpe_ratio = (mean_return / std_return) * np.sqrt(365) if std_return > 0 else 0
        
        # Maximum Drawdown
        cumulative = portfolio_array / self.initial_balance
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown_pct = np.min(drawdown) * 100
        
        # Volatility
        volatility = std_return * np.sqrt(365) * 100
        
        # Trade-based metrics
        trades_df = self.identify_actual_trades()
        
        if not trades_df.empty:
            num_trades = len(trades_df)
            winning_trades = trades_df[trades_df['win'] == True]
            losing_trades = trades_df[trades_df['win'] == False]
            
            num_wins = len(winning_trades)
            num_losses = len(losing_trades)
            win_rate = (num_wins / num_trades * 100) if num_trades > 0 else 0
            
            avg_win = winning_trades['pnl_pct'].mean() if num_wins > 0 else 0
            avg_loss = losing_trades['pnl_pct'].mean() if num_losses > 0 else 0
            
            total_win_amount = winning_trades['pnl'].sum() if num_wins > 0 else 0
            total_loss_amount = abs(losing_trades['pnl'].sum()) if num_losses > 0 else 0
            profit_factor = total_win_amount / total_loss_amount if total_loss_amount > 0 else 0
            
            # Expectancy
            expectancy = (win_rate/100 * avg_win) - ((1 - win_rate/100) * abs(avg_loss))
        else:
            num_trades = 0
            num_wins = 0
            num_losses = 0
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
            expectancy = 0
        
        # Total reward
        total_reward = np.sum(self.rewards)
        
        metrics = {
            'Initial Balance': f'${self.initial_balance:,.2f}',
            'Final Balance': f'${final_value:,.2f}',
            'Total Return': f'${final_value - self.initial_balance:,.2f}',
            'Total Return (%)': f'{total_return:.2f}%',
            'Sharpe Ratio': f'{sharpe_ratio:.3f}',
            'Max Drawdown': f'{max_drawdown_pct:.2f}%',
            'Volatility (Annual)': f'{volatility:.2f}%',
            '---': '---',
            'Total Trades': f'{num_trades}',
            'Winning Trades': f'{num_wins}',
            'Losing Trades': f'{num_losses}',
            'Win Rate': f'{win_rate:.2f}%',
            'Avg Win': f'{avg_win:.2f}%',
            'Avg Loss': f'{avg_loss:.2f}%',
            'Profit Factor': f'{profit_factor:.3f}',
            'Expectancy': f'{expectancy:.2f}%',
            'Total Reward': f'{total_reward:,.2f}',
            'Total Steps': f'{len(self.portfolio_values):,}'
        }
        
        return metrics, trades_df

# USAGE EXAMPLES:

# If you have CSV file:
# recalc = MetricsRecalculator(csv_path='backtest_results.csv')

# If you have JSON file:
# recalc = MetricsRecalculator(json_path='backtest_results.json')

# Calculate corrected metrics
# metrics, trades = recalc.calculate_correct_metrics()

# Print metrics
# for key, value in metrics.items():
#     print(f"{key:.<40} {value}")

# Print trade details
# if not trades.empty:
#     print("\nTrade Details:")
#     print(trades.to_string(index=False))