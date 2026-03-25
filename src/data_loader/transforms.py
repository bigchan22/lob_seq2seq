import torch
import numpy as np

class LOBFeatureTransformer:
    def __init__(self, col_map, include_investor=True, epsilon=1e-9):
        self.col_map = col_map
        self.include_investor = include_investor
        self.eps = epsilon

    def __call__(self, x):
        # x shape: (38, N, F_raw)
        device = x.device
        
        # 1. Calculate Midprice
        ask1 = x[..., self.col_map['ASK_STEP1_BSTORD_PRC']]
        bid1 = x[..., self.col_map['BID_STEP1_BSTORD_PRC']]
        midprice = (ask1 + bid1) / 2.0
        
        features_list = []

        # --- CATEGORY A: Prices (REDUCED) ---
        # Log Midprice (The anchor)
        features_list.append(torch.log(midprice + self.eps).unsqueeze(-1))
        
        # ONLY Level 1 Relative Prices (Spread info)
        # We skip range(2, 6) here to avoid the -inf issues you found
        features_list.append(torch.log((ask1 / (midprice + self.eps)) + self.eps).unsqueeze(-1))
        features_list.append(torch.log((bid1 / (midprice + self.eps)) + self.eps).unsqueeze(-1))

        # --- CATEGORY B: Quantities (KEPT 1-5) ---
        # Quantities are log1p-transformed (safe from log(0) -> -inf)
        for i in range(1, 6):
            q_ask = x[..., self.col_map[f'ASK_STEP{i}_BSTORD_RQTY']]
            q_bid = x[..., self.col_map[f'BID_STEP{i}_BSTORD_RQTY']]
            features_list.append(torch.log1p(q_ask).unsqueeze(-1))
            features_list.append(torch.log1p(q_bid).unsqueeze(-1))

        # --- CATEGORY C: Investor Data ---
        if self.include_investor:
            inst_buy_vol, inst_sell_vol = 0, 0
            ret_buy_vol, ret_sell_vol = 0, 0

            for grp in ['11', '12', '21', '22']:
                b_qty = x[..., self.col_map[f'BUY_QTY{grp}']]
                s_qty = x[..., self.col_map[f'SELL_QTY{grp}']]
                b_prc = x[..., self.col_map[f'BUY_PRC{grp}']]
                s_prc = x[..., self.col_map[f'SELL_PRC{grp}']]

                # Order Imbalance (OIB)
                features_list.append(((b_qty - s_qty) / (b_qty + s_qty + self.eps)).unsqueeze(-1))
                
                # Net Flow (Log-scaled magnitude)
                net_flow = torch.sign(b_qty - s_qty) * torch.log1p(torch.abs(b_qty - s_qty))
                features_list.append(net_flow.unsqueeze(-1))
                
                # Relative Investor Prices (using nan_to_num as it's investor specific)
                features_list.append(torch.log((b_prc / (midprice + self.eps)) + self.eps).nan_to_num(0).unsqueeze(-1))
                features_list.append(torch.log((s_prc / (midprice + self.eps)) + self.eps).nan_to_num(0).unsqueeze(-1))

                if grp in ['11', '12']:
                    inst_buy_vol += b_qty; inst_sell_vol += s_qty
                else:
                    ret_buy_vol += b_qty; ret_sell_vol += s_qty

            # Ratio
            ratio = (inst_buy_vol + inst_sell_vol - (ret_buy_vol + ret_sell_vol)) / \
                    (inst_buy_vol + inst_sell_vol + ret_buy_vol + ret_sell_vol + self.eps)
            features_list.append(ratio.unsqueeze(-1))

        return torch.cat(features_list, dim=-1)