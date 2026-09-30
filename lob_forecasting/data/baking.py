def investor_raw_columns():
    return [f"{side}_{kind}{grp}" for grp in ["11","12","21","22"] for side,kind in [("BUY","QTY"),("SELL","QTY"),("BUY","PRC"),("SELL","PRC")]]
def lag_investor_columns(df, lag_steps):
    if lag_steps <= 0: return df
    cols=[c for c in investor_raw_columns() if c in df.columns]
    if cols: df.loc[:,cols]=df.loc[:,cols].shift(lag_steps).fillna(0)
    return df
