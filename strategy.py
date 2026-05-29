import numpy as np
import pandas as pd

def gen_position(df: pd.DataFrame,
                  ema_period: int   = 20,
                  atr_period: int   = 10,
                  mult:       float = 2.0) -> pd.DataFrame:
    """
    Trend Following — Keltner Channel Breakout
    Vào LONG  : Close vượt Upper band
    Vào SHORT : Close phá Lower band
    Thoát     : Close về phía đường giữa (EMA)
    """
    df = df.copy()

    #  Keltner Channel 
    ema = df["Close"].ewm(span=ema_period, adjust=False).mean()

    high, low, close = df["High"], df["Low"], df["Close"]
    tr  = pd.concat([high - low,
                     (high - close.shift(1)).abs(),
                     (low  - close.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.ewm(span=atr_period, adjust=False).mean()

    upper  = (ema + mult * atr).shift(1).values  # shift tránh look-ahead
    lower  = (ema - mult * atr).shift(1).values
    middle = ema.shift(1).values
    close_vals = close.values

    n         = len(df)
    positions = np.zeros(n, dtype=int)
    state     = 0

    for i in range(ema_period + atr_period, n):
        c = close_vals[i]

        if state == 0:
            if   c > upper[i]:  state = 1
            elif c < lower[i]:  state = -1

        elif state == 1:
            # Thoát long khi giá về dưới EMA
            if c < middle[i]:   state = 0

        elif state == -1:
            # Thoát short khi giá về trên EMA
            if c > middle[i]:   state = 0

        positions[i] = state

    df["position"] = positions
    return df
