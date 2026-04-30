# Week 7 — Reinforcement Learning Intro (Ch. 17)

## Objective

RL is the most direct application of deep learning to trading decisions. You'll go from the theory of MDPs and Q-learning to building your first DQN trading agent that takes long/flat/short positions on a real stock.

## Schedule

| Day | Activity |
|-----|----------|
| Mon-Tue | Ch. 17 — RL intro: MDP, value iteration, Q-learning |
| Wed-Thu | Implement a Q-learning agent on a custom trading environment |
| Fri | Project: DQN on a single stock |

## Project: DQN Trading Agent

**Build a DQN agent that trades a single stock:**
- **State**: last N days of returns, rolling volatility, position (long/flat/short)
- **Actions**: {0: go flat, 1: go long, 2: go short}
- **Reward**: daily P&L minus transaction costs
- **Environment**: custom `gym`-style class wrapping real historical data

## Deliverable

Notebook: **"My First RL Trading Agent"**

- Custom trading environment
- DQN implementation with replay buffer and target network
- Training curves (reward over episodes)
- Backtest: cumulative returns vs buy-and-hold
- Honest analysis: what works, what doesn't, what would make it better

## Key Concepts

- **MDP**: Markov Decision Process — state, action, reward, transition
- **Q-function**: Q(s, a) = expected cumulative reward from state s taking action a
- **Bellman equation**: Q(s,a) = r + γ · max_{a'} Q(s', a')
- **DQN improvements**: experience replay, target network (both are mandatory)
- **Exploration vs exploitation**: epsilon-greedy policy

## Custom Trading Environment

```python
class TradingEnv:
    def __init__(self, returns, window=20, transaction_cost=0.001):
        self.returns = returns
        self.window = window
        self.cost = transaction_cost
        self.reset()

    def reset(self):
        self.t = self.window
        self.position = 0  # -1: short, 0: flat, 1: long
        return self._get_state()

    def step(self, action):
        new_position = action - 1  # map {0,1,2} -> {-1,0,1}
        trade = abs(new_position - self.position)
        pnl = new_position * self.returns[self.t] - trade * self.cost
        self.position = new_position
        self.t += 1
        done = self.t >= len(self.returns)
        return self._get_state(), pnl, done

    def _get_state(self):
        return np.concatenate([
            self.returns[self.t - self.window:self.t],
            [self.position]
        ])
```

## DQN Architecture

```python
class DQN(nn.Module):
    def __init__(self, state_dim, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, n_actions)
        )

    def forward(self, x):
        return self.net(x)
```

## Data Sources

- `yfinance`: pick a volatile stock (TSLA, BTC-USD, or a sector ETF)
- Use at least 5 years of daily data

## Tips

- **Transaction costs are critical**: without them, the agent will overtrade and fake alpha
- Start with a **very simple state** — RL with financial data overfits easily
- The agent should be compared to buy-and-hold AND to a simple momentum strategy
- RL in finance is hard — temper expectations, the goal is understanding the framework
- **This project impresses quant recruiters**: very few candidates have built an RL trading agent
- Read the original DQN paper (Mnih et al. 2015) — you'll likely be asked about it
