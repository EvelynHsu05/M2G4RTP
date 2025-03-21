# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# class lstm_eta_decoder(nn.Module):
#     def __init__(self, state_size, hidden_size, seq_len=20):
#         super(lstm_eta_decoder, self).__init__()
#         self.lstm = nn.LSTM(input_size=state_size,
#                             hidden_size=hidden_size,
#                             num_layers=2, batch_first=True)

#         self.output_layer = nn.Sequential(
#             nn.Linear(hidden_size, hidden_size // 2),
#             nn.ReLU(),
#             nn.Linear(hidden_size // 2, 1),
#         )
#         self.seq_len = seq_len

#     def forward(self, hidden_state, unpick_len, pred_idx, pred_score=None):
#         pred_idx[pred_idx == -1] = self.seq_len - 1
#         state_sort_idx = pred_idx.unsqueeze(-1).expand(pred_idx.shape[0], pred_idx.shape[1], hidden_state.shape[-1])
#         sorted_state = hidden_state.gather(1, state_sort_idx.to(torch.int64))

#         pack_state = nn.utils.rnn.pack_padded_sequence(sorted_state,
#                                                        unpick_len.cpu().to(torch.int64),
#                                                        batch_first=True,
#                                                        enforce_sorted=False)
#         output_state, (_, _) = self.lstm(pack_state)
#         output_state, _ = nn.utils.rnn.pad_packed_sequence(output_state, batch_first=True)
#         output_state = nn.functional.pad(output_state,
#                                          [0, 0, 0, self.seq_len - output_state.shape[1], 0, 0],
#                                          mode="constant",
#                                          value=0)

#         pred_eta = self.output_layer(output_state)
#         pred_eta = pred_eta.squeeze()

#         resort_index = torch.argsort(pred_idx, dim=1)
#         resorted_pred_eta = pred_eta.gather(1, resort_index.to(torch.int64))
#         return resorted_pred_eta


# 載入 torch 模組，提供張量操作與數值運算功能
import torch
# 載入 torch.nn 模組，提供神經網路相關的層與工具
import torch.nn as nn
# 載入 torch.nn.functional 模組，提供函數式 API，如激活函數、卷積等
import torch.nn.functional as F


# 定義一個基於 LSTM 的 ETA (Estimated Time of Arrival, 預計到達時間) 解碼器
class lstm_eta_decoder(nn.Module):
    # 定義初始化方法，參數 state_size 為輸入特徵維度，hidden_size 為 LSTM 隱藏狀態大小，
    # seq_len 為序列最大長度，預設值為 20
    def __init__(self, state_size, hidden_size, seq_len=20):
        # 呼叫父類別 nn.Module 的初始化方法
        super(lstm_eta_decoder, self).__init__()
        # 定義一個 LSTM 模組，輸入維度為 state_size，隱藏維度為 hidden_size，
        # 設定 LSTM 層數為 2 且 batch_first=True，表示輸入與輸出形狀為 (batch, seq, feature)
        self.lstm = nn.LSTM(input_size=state_size,
                            hidden_size=hidden_size,
                            num_layers=2, batch_first=True)

        # 定義一個全連接層組成的輸出層，用於將 LSTM 輸出映射到單個標量 (ETA)
        self.output_layer = nn.Sequential(
            # 第一個全連接層：將隱藏狀態從 hidden_size 映射到 hidden_size 的一半
            nn.Linear(hidden_size, hidden_size // 2),
            # 使用 ReLU 激活函數引入非線性
            nn.ReLU(),
            # 第二個全連接層：將隱藏狀態從 hidden_size//2 映射到 1 維 (ETA 預測值)
            nn.Linear(hidden_size // 2, 1),
        )
        # 將傳入的序列最大長度存入實例變數，用於後續補零操作
        self.seq_len = seq_len

    # 定義前向傳播函式
    # hidden_state: [batch_size, seq_len, feature_dim]，代表每個位置的隱藏表示
    # unpick_len: 每個樣本有效的序列長度 (未填充部分)
    # pred_idx: 預測的排序索引，代表 decoder 預測出的順序
    # pred_score: 可選參數，預測分數 (此處未使用)
    def forward(self, hidden_state, unpick_len, pred_idx, pred_score=None):
        # 將 pred_idx 中所有值為 -1 的位置替換為 (seq_len - 1)，避免無效索引出現
        pred_idx[pred_idx == -1] = self.seq_len - 1
        # 將 pred_idx 維度從 [batch_size, seq_len] 擴展為 [batch_size, seq_len, feature_dim]
        # 以便後續從 hidden_state 中根據索引擷取對應特徵
        state_sort_idx = pred_idx.unsqueeze(-1).expand(pred_idx.shape[0], pred_idx.shape[1], hidden_state.shape[-1])
        # 根據 state_sort_idx 從 hidden_state 中擷取排序後的狀態表示，結果形狀仍為 [batch_size, seq_len, feature_dim]
        sorted_state = hidden_state.gather(1, state_sort_idx.to(torch.int64))

        # 將排序後的狀態表示根據 unpick_len 打包成 PackedSequence
        # 使用 pack_padded_sequence 使 LSTM 只處理有效長度部分，enforce_sorted=False 表示不要求排序
        pack_state = nn.utils.rnn.pack_padded_sequence(sorted_state,
                                                       unpick_len.cpu().to(torch.int64),
                                                       batch_first=True,
                                                       enforce_sorted=False)
        # 將打包的序列輸入到 LSTM 中，output_state 為打包後的輸出，(_, _) 分別代表最終隱藏狀態與細胞狀態（此處未使用）
        output_state, (_, _) = self.lstm(pack_state)
        # 將 LSTM 輸出的 PackedSequence 轉換回填充後的張量，形狀為 [batch_size, padded_seq_len, hidden_size]
        output_state, _ = nn.utils.rnn.pad_packed_sequence(output_state, batch_first=True)
        # 如果 output_state 序列長度不足 seq_len，則用 0 對其進行補零，使得序列長度達到 seq_len
        # pad 的參數 [0, 0, 0, self.seq_len - output_state.shape[1], 0, 0] 表示僅在序列維度 (第二維) 補零
        output_state = nn.functional.pad(output_state,
                                         [0, 0, 0, self.seq_len - output_state.shape[1], 0, 0],
                                         mode="constant",
                                         value=0)

        # 將補零後的 LSTM 輸出通過全連接的 output_layer 得到預測的 ETA 值，結果形狀為 [batch_size, seq_len, 1]
        pred_eta = self.output_layer(output_state)
        # 利用 squeeze 壓縮最後一個維度，最終 pred_eta 的形狀變為 [batch_size, seq_len]
        pred_eta = pred_eta.squeeze()

        # 根據 pred_idx 的值，對每個樣本內的索引進行 argsort 排序，獲得重新排序的索引 resort_index
        resort_index = torch.argsort(pred_idx, dim=1)
        # 根據 resort_index 將 pred_eta 的預測結果重新排列，恢復原始順序
        resorted_pred_eta = pred_eta.gather(1, resort_index.to(torch.int64))
        # 返回重新排序後的 ETA 預測結果，形狀為 [batch_size, seq_len]
        return resorted_pred_eta
