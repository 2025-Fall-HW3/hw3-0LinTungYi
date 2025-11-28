"""
Package Import
"""
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import quantstats as qs
import gurobipy as gp
import warnings
import argparse
import sys

"""
Project Setup
"""
warnings.simplefilter(action="ignore", category=FutureWarning)

assets = [
    "SPY",
    "XLB",
    "XLC",
    "XLE",
    "XLF",
    "XLI",
    "XLK",
    "XLP",
    "XLRE",
    "XLU",
    "XLV",
    "XLY",
]

# Initialize Bdf and df
Bdf = pd.DataFrame()
for asset in assets:
    raw = yf.download(asset, start="2012-01-01", end="2024-04-01", auto_adjust = False)
    Bdf[asset] = raw['Adj Close']

df = Bdf.loc["2019-01-01":"2024-04-01"]

"""
Strategy Creation

Create your own strategy, you can add parameter but please remain "price" and "exclude" unchanged
"""




class MyPortfolio:
    """
    NOTE: You can modify the initialization function
    """

    def __init__(self, price, exclude, lookback=50, gamma=0):
        self.price = price
        self.returns = price.pct_change().fillna(0)
        self.exclude = exclude
        self.lookback = lookback
        self.gamma = gamma

    def calculate_weights(self):
        # Get the assets by excluding the specified column
        assets = self.price.columns[self.price.columns != self.exclude]

        # Calculate the portfolio weights
        self.portfolio_weights = pd.DataFrame(
            index=self.price.index, columns=self.price.columns
        )

        """
        TODO: Complete Task 4 Below
        """
        
        # 1) 先設定 baseline：全期間等權重
        n_assets = len(assets)
        self.portfolio_weights[assets] = 1.0 / n_assets

        # 方便取用 index
        date_index = self.portfolio_weights.index
        total_len = len(self.portfolio_weights)

        # 從大約 lookback/5 之後才開始用自己的策略
        start_day = self.lookback // 5
        if start_day < 1:
            start_day = 1

        # 目前持有的權重（先用等權重）
        current_w = np.ones(n_assets) / n_assets

        for i in range(start_day, total_len):
            # 實際用的 lookback 天數：一開始資料還不夠就用 i 天
            lookback = min(self.lookback, i)
            # 用「日報酬」作為歷史資料
            window_returns = self.returns.copy()[assets].iloc[i - lookback : i]

            # 每 10 天重算一次；第一天也要算一次
            if i % 10 == 0 or i == start_day:
                sol = self.get_solution(window_returns, no_short=False)
                # 正規化確保權重和為 1
                sol = sol / sol.sum()
                current_w = sol.values if isinstance(sol, pd.Series) else sol

            # 寫入當天的權重（即使今天沒重算，也沿用 current_w）
            self.portfolio_weights.loc[date_index[i], assets] = current_w
        
        """
        TODO: Complete Task 4 Above
        """

        self.portfolio_weights.ffill(inplace=True)
        self.portfolio_weights.fillna(0, inplace=True)

    def get_solution(self, window_returns: pd.DataFrame, no_short: bool = True):
        # 參考 https://medium.com/@carlolepelaars/towards-a-more-profitable-s-p500-with-portfolio-optimization-c43a37d9078f
        """
        給定一段歷史「日報酬」資料，做 minimum variance 配置

        window_returns: rows = 天數, cols = 資產
        no_short      : True → 把負權重砍成 0（不做空）；False → 可有負權重
        """
        # 這裡的 mean 其實沒用到，只是留著可以擴充
        mu = window_returns.mean()
        sigma = window_returns.cov()

        # 逆 covariance，用 pseudo-inverse 比較穩定
        inv_sigma = np.linalg.pinv(sigma.values)
        ones = np.ones(len(mu))

        # global minimum variance 的解析式解：w ∝ Σ^{-1} 1
        inv_dot_ones = inv_sigma @ ones
        weights = inv_dot_ones / (ones @ inv_dot_ones)

        # 轉成 Series 比較好對應 assets
        weights = pd.Series(weights, index=window_returns.columns)

        # 如果不允許做空，就把 <=0 的砍成 0，再重正規化
        if no_short:
            weights[weights <= 0] = 0.0
            if weights.sum() == 0:
                # 極端情況全部被砍掉，退回等權重
                weights[:] = 1.0 / len(weights)
            else:
                weights = weights / weights.sum()

        return weights
    
    def calculate_portfolio_returns(self):
        # Ensure weights are calculated
        if not hasattr(self, "portfolio_weights"):
            self.calculate_weights()

        # Calculate the portfolio returns
        self.portfolio_returns = self.returns.copy()
        assets = self.price.columns[self.price.columns != self.exclude]
        self.portfolio_returns["Portfolio"] = (
            self.portfolio_returns[assets]
            .mul(self.portfolio_weights[assets])
            .sum(axis=1)
        )

    def get_results(self):
        # Ensure portfolio returns are calculated
        if not hasattr(self, "portfolio_returns"):
            self.calculate_portfolio_returns()

        return self.portfolio_weights, self.portfolio_returns


if __name__ == "__main__":
    # Import grading system (protected file in GitHub Classroom)
    from grader_2 import AssignmentJudge
    
    parser = argparse.ArgumentParser(
        description="Introduction to Fintech Assignment 3 Part 12"
    )

    parser.add_argument(
        "--score",
        action="append",
        help="Score for assignment",
    )

    parser.add_argument(
        "--allocation",
        action="append",
        help="Allocation for asset",
    )

    parser.add_argument(
        "--performance",
        action="append",
        help="Performance for portfolio",
    )

    parser.add_argument(
        "--report", action="append", help="Report for evaluation metric"
    )

    parser.add_argument(
        "--cumulative", action="append", help="Cumulative product result"
    )

    args = parser.parse_args()

    judge = AssignmentJudge()
    
    # All grading logic is protected in grader_2.py
    judge.run_grading(args)
