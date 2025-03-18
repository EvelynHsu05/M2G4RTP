# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# class GAT_layer_multihead(nn.Module):
#     """
#     https://arxiv.org/pdf/1710.10903.pdf
#     """
#     def __init__(self,
#                  node_in_dim,
#                  edge_in_dim,
#                  node_out_dim,
#                  edge_out_dim,
#                  nheads=8,
#                  drop=0.5,
#                  leaky=0.2,
#                  is_mix_attention=True,
#                  is_update_edge=True):
#         super(GAT_layer_multihead, self).__init__()
#         # 多头计算
#         self.nheads = nheads

#         self.node_out_dim = node_out_dim
#         self.edge_out_dim = edge_out_dim

#         self.is_mix_attention = is_mix_attention
#         self.is_update_edge = is_update_edge
#         if is_mix_attention:
#             self.a_edge_init = nn.Linear(edge_in_dim * 1, 1 * nheads)
#             self.a_edge = nn.Linear(edge_in_dim * nheads, 1 * nheads)
#             if is_update_edge:
#                 assert edge_out_dim == node_out_dim
#                 self.W_edge = nn.Linear(edge_in_dim * nheads, edge_out_dim * nheads)
#                 self.W_od = nn.Linear(node_in_dim * nheads, node_out_dim * nheads)
#                 self.W_edge_init = nn.Linear(edge_in_dim * 1, edge_out_dim * nheads)
#                 self.W_od_init = nn.Linear(node_in_dim * 1, node_out_dim * nheads)
#         self.W_init = nn.Linear(node_in_dim * 1, node_out_dim * nheads)
#         self.W = nn.Linear(node_in_dim * nheads, node_out_dim * nheads)
#         self.a1 = nn.Linear(node_out_dim * nheads, 1 * nheads, bias=False)
#         self.a2 = nn.Linear(node_out_dim * nheads, 1 * nheads, bias=False)
#         self.leakyrelu = nn.LeakyReLU(leaky)
#         self.dropout = nn.Dropout(p=drop)

#     def update_edge(self, node_fea, edge_fea, first_layer=False):
#         # node_fea: [B, N, nheads * node_in_dim]
#         # edge_fea: [B, N, N, nheads * node_in_dim]

#         if not first_layer:
#             edge = self.W_edge(edge_fea)
#             od = self.W_od(node_fea)
#         else:
#             edge = self.W_edge_init(edge_fea)
#             od = self.W_od_init(node_fea)
#         o = od.unsqueeze(2)
#         d = od.unsqueeze(1)
#         return o + d + edge

#     @staticmethod
#     def mask(x, adj):
#         if adj is None:
#             return x
#         adj = adj.unsqueeze(-1).expand(adj.shape[0], adj.shape[1], adj.shape[2], x.shape[-1])
#         x = torch.where(adj > 0, x, -9e15 * torch.ones_like(x))
#         return x

#     def forward(self, node_fea, edge_fea, adj):
#         first_layer = False
#         B, H, N, _ = node_fea.shape
#         # node_fea: [B, nhead, N, node_in_dim]
#         # edge_fea: [B, nhead, N, N, edge_in_dim]

#         # node_fea_new: [B, N, nheads * node_in_dim]
#         node_fea_new = node_fea.permute(0, 2, 1, 3).reshape(B, N, -1)
#         # edge_fea_new: [B, N, N, nheads * node_in_dim]
#         edge_fea_new = edge_fea.permute(0, 2, 3, 1, 4).reshape(B, N, N, -1)
#         if H == self.nheads:
#             # Wh: [B, N, nheads * node_out_dim]
#             Wh = self.W(node_fea_new)
#             # e_edge: [B, N, N, nheads * 1]
#             e_edge = self.a_edge(edge_fea_new)
#         else:
#             first_layer = True
#             # Wh: [B, N, nheads * node_out_dim]
#             Wh = self.W_init(node_fea_new)
#             # e_edge: [B, N, N, nheads * 1]
#             # try:
#             e_edge = self.a_edge_init(edge_fea_new)
#             # except:
#             #     print(self.a_edge_init.weight.shape)
#             #     print(edge_fea_new.shape)

#         # Whi&j: [B, N, nheads * 1]
#         Whi = self.a1(Wh)
#         Whj = self.a2(Wh)

#         if self.is_mix_attention:
#             # e: [B, N, N, nheads * 1]
#             e = Whi.unsqueeze(2) + Whj.unsqueeze(1)
#             e = self.leakyrelu(e + e_edge)
#             e = self.mask(e, adj)
#             if self.is_update_edge:
#                 # edge_fea_new: [B, N, N, nhead * edge_out_dim]
#                 edge_fea_new = self.update_edge(node_fea_new, edge_fea_new, first_layer)
#                 # edge_fea_new: [B, N, N, nhead * edge_out_dim]
#                 #               -> [B, N, N, nhead, edge_out_dim]
#                 #               -> [B, nhead, N, N, edge_out_dim]
#                 edge_fea_new = edge_fea_new.reshape(B, N, N, self.nheads, -1).permute(0, 3, 1, 2, 4)
#             else:
#                 edge_fea_new = edge_fea
#         else:
#             # e: [B, N, N, nheads * 1]
#             e = Whi.unsqueeze(2) + Whj.unsqueeze(1)
#             e = self.leakyrelu(e)
#             e = self.mask(e, adj)
#             edge_fea_new = edge_fea

#         # attention: [B, nheads, N, N]
#         attention = e.permute(0, 3, 1, 2)
#         attention = F.softmax(attention, dim=-1)
#         attention = self.dropout(attention)

#         # Wh: [B, nheads, N, node_out_dim]
#         Wh = Wh.contiguous().view(B, N, self.nheads, self.node_out_dim).permute(0, 2, 1, 3)
#         node_fea_new = torch.matmul(attention, Wh) + Wh

#         # node_fea_new: [B, N, node_out_dim]
#         # node_fea_new = node_fea_new.mean(dim=1)

#         return node_fea_new, edge_fea_new


# class GAT_layer(nn.Module):
#     def __init__(self, node_size, edge_size, hidden_size, nhead=6, is_mix_attention=False, is_update_edge=True):
#         super(GAT_layer, self).__init__()
#         self.is_mix_attention = is_mix_attention
#         self.is_update_edge = is_update_edge
#         if is_mix_attention:
#             self.We = nn.Linear(edge_size, hidden_size)
#             if is_update_edge:
#                 self.W_od = nn.Linear(node_size, hidden_size)
#         self.W = nn.Linear(node_size, hidden_size)
#         self.a1 = nn.Linear(node_size, 1, bias=False)
#         self.a2 = nn.Linear(node_size, 1, bias=False)
#         self.leakyrelu = nn.LeakyReLU(0.2)
#         self.dropout = nn.Dropout(p=0.5)

#     def update_edge(self, node_fea, edge_fea):
#         od = self.W_od(node_fea)
#         o = od.unsqueeze(2)
#         d = od.unsqueeze(1)
#         return edge_fea + o + d

#     def forward(self, node_fea, edge_fea, adj):
#         # node_fea: [B, N, node_size]
#         Wh = self.W(node_fea)

#         # Whi&j: [B, N, 1]
#         Whi = self.a1(Wh)
#         Whj = self.a2(Wh)
#         # e: [B, N, N]
#         if self.is_mix_attention:
#             e_node = Whi + Whj.T
#             Wh_edge = self.We(edge_fea)
#             e = self.leakyrelu(e_node + Wh_edge)
#             if adj is not None:
#                 e = torch.where(adj > 0, e, -9e15 * torch.ones_like(e))
#             if self.is_update_edge:
#                 edge_fea_new = self.update_edge(node_fea, Wh_edge)
#             else:
#                 edge_fea_new = edge_fea
#         else:
#             e = self.leakyrelu(Whi + Whj.T)
#             e = torch.where(adj > 0, e, -9e15 * torch.ones_like(e))
#             edge_fea_new = edge_fea

#         # attention: [B, N, N]
#         attention = F.softmax(e)
#         attention = self.dropout(attention)

#         node_fea_new = torch.matmul(attention, Wh) + Wh

#         return node_fea_new, edge_fea_new


# class GAT_encoder(nn.Module):
#     def __init__(self, node_size, edge_size, hidden_size,
#                  num_layers=3, nheads=4, is_mix_attention=True, is_update_edge=True, num_node=20):
#         super(GAT_encoder, self).__init__()
#         self.num_layers = num_layers
#         self.gat = nn.ModuleList()
#         self.gat.append(GAT_layer_multihead(node_in_dim=node_size,
#                                             edge_in_dim=edge_size,
#                                             node_out_dim=hidden_size,
#                                             edge_out_dim=hidden_size,
#                                             nheads=nheads,
#                                             is_mix_attention=is_mix_attention,
#                                             is_update_edge=is_update_edge
#                                             ))
#         for i in range(1, num_layers):
#             self.gat.append(GAT_layer_multihead(node_in_dim=hidden_size,
#                                                 edge_in_dim=hidden_size,
#                                                 node_out_dim=hidden_size,
#                                                 edge_out_dim=hidden_size,
#                                                 nheads=nheads,
#                                                 is_mix_attention=is_mix_attention,
#                                                 is_update_edge=is_update_edge
#                                                 ))

#     def forward(self, node_fea, edge_fea, adj=None):
#         node_fea = node_fea.unsqueeze(1)
#         edge_fea = edge_fea.unsqueeze(1)
#         for i in range(self.num_layers):
#             node_fea, edge_fea = self.gat[i](node_fea, edge_fea, adj)
#             if i == self.num_layers - 1:
#                 # node_fea_new: [B, nhead, N, node_out_dim] -> [B, N, node_out_dim]
#                 node_fea = node_fea.mean(dim=1)
#             else:
#                 node_fea = F.relu(node_fea)
#         return node_fea, edge_fea

import torch                     # 匯入 PyTorch 核心庫，提供張量運算和自動微分功能
import torch.nn as nn            # 從 PyTorch 中匯入神經網絡模組，用於定義各種網絡層
import torch.nn.functional as F  # 匯入 torch.nn.functional 模組，提供激活函數、softmax、dropout 等常用函數


class GAT_layer_multihead(nn.Module):
    """
    參考文獻：https://arxiv.org/pdf/1710.10903.pdf
    此類實現多頭圖注意力層（GAT），並結合邊特徵更新機制（GAT-e）。
    論文中提到的 Multi-level GAT-e Encoding Module 正是利用這種機制，
    將 AOI 級與地點級圖的節點和邊特徵進行融合，提取高層次表示。
    """
    def __init__(self,
                 node_in_dim,     # 輸入節點特徵的維度，例如包含地理位置、時間、其他離散或連續特徵
                 edge_in_dim,     # 輸入邊特徵的維度，例如兩點間距離、時間間隔、連通性等信息
                 node_out_dim,    # 輸出節點特徵的維度，經過此層編碼後的節點表示將作為下游任務輸入
                 edge_out_dim,    # 輸出邊特徵的維度，通常要求與 node_out_dim 相同，便於後續融合
                 nheads=8,        # 多頭注意力的頭數，能夠捕捉多個子空間中的特徵
                 drop=0.5,        # dropout 機率，用於防止過擬合
                 leaky=0.2,       # LeakyReLU 激活函數的負斜率，用以引入非線性
                 is_mix_attention=True,  # 是否使用混合注意力，即將邊特徵參與注意力得分的計算
                 is_update_edge=True):   # 是否根據節點信息動態更新邊特徵
        super(GAT_layer_multihead, self).__init__()  # 呼叫父類 nn.Module 的初始化方法

        # 保存多頭注意力的頭數，後續多頭運算均依此數量進行
        self.nheads = nheads

        # 保存輸出節點與邊特徵的維度，這兩者在啟用邊更新時必須一致
        self.node_out_dim = node_out_dim
        self.edge_out_dim = edge_out_dim

        # 保存是否使用混合注意力及是否更新邊特徵的標誌
        self.is_mix_attention = is_mix_attention
        self.is_update_edge = is_update_edge
        if is_mix_attention:
            # 對原始邊特徵進行線性變換，使用單頭線性層將 edge_in_dim 映射到 1*nheads 維度
            self.a_edge_init = nn.Linear(edge_in_dim * 1, 1 * nheads)
            # 對多頭化後的邊特徵進行線性變換，將 edge_in_dim*nheads 映射到 1*nheads 維度
            self.a_edge = nn.Linear(edge_in_dim * nheads, 1 * nheads)
            if is_update_edge:
                # 當啟用邊更新時，要求輸出邊特徵維度必須與節點輸出維度一致（便於後續相加融合）
                assert edge_out_dim == node_out_dim
                # 定義一個線性層，用於更新多頭邊特徵：
                # 將 edge_in_dim*nheads 映射到 edge_out_dim*nheads，從而學習邊特徵的新表示
                self.W_edge = nn.Linear(edge_in_dim * nheads, edge_out_dim * nheads)
                # 定義一個線性層，從節點特徵中獲取更新邊特徵所需的信息
                # 將 node_in_dim*nheads 映射到 node_out_dim*nheads
                self.W_od = nn.Linear(node_in_dim * nheads, node_out_dim * nheads)
                # 第一層專用：使用單頭線性層對邊特徵進行初始映射，再擴展至多頭
                self.W_edge_init = nn.Linear(edge_in_dim * 1, edge_out_dim * nheads)
                # 第一層專用：使用單頭線性層對節點特徵進行初始映射，再擴展至多頭
                self.W_od_init = nn.Linear(node_in_dim * 1, node_out_dim * nheads)
        # 定義對節點特徵的初始線性變換（單頭版本），將輸入特徵從 node_in_dim 映射到 node_out_dim*nheads
        self.W_init = nn.Linear(node_in_dim * 1, node_out_dim * nheads)
        # 定義對多頭化後的節點特徵進行進一步線性變換，將 node_in_dim*nheads 映射到 node_out_dim*nheads
        self.W = nn.Linear(node_in_dim * nheads, node_out_dim * nheads)
        # 定義計算注意力得分的線性層 a1，處理節點作為來源（source）的表示，無偏置
        self.a1 = nn.Linear(node_out_dim * nheads, 1 * nheads, bias=False)
        # 定義計算注意力得分的線性層 a2，處理節點作為目標（target）的表示，無偏置
        self.a2 = nn.Linear(node_out_dim * nheads, 1 * nheads, bias=False)
        # 定義 LeakyReLU 激活函數，使用指定的負斜率 leaky
        self.leakyrelu = nn.LeakyReLU(leaky)
        # 定義 dropout 層，隨機丟棄一部分注意力值以防止過擬合，機率由 drop 指定
        self.dropout = nn.Dropout(p=drop)

    def update_edge(self, node_fea, edge_fea, first_layer=False):
        # node_fea: [B, N, nheads * node_in_dim]，B 為批次大小，N 為節點數
        # edge_fea: [B, N, N, nheads * node_in_dim]，每對節點間的原始邊特徵

        if not first_layer:
            # 非第一層時，使用多頭線性層更新邊特徵
            edge = self.W_edge(edge_fea)   # 將邊特徵從原始空間映射到更新後的空間，形狀變為 [B, N, N, nheads * edge_out_dim]
            od = self.W_od(node_fea)         # 將節點特徵映射，獲得用於更新邊特徵的中間表示，形狀 [B, N, nheads * node_out_dim]
        else:
            # 第一層時，使用初始線性層進行映射（單頭版本，再擴展至多頭）
            edge = self.W_edge_init(edge_fea)   # 初始邊特徵線性變換
            od = self.W_od_init(node_fea)         # 初始節點特徵線性變換
        # 將映射後的節點特徵 od 擴展，構造來源和目的節點的信息
        o = od.unsqueeze(2)  # 將 od 從 [B, N, dim] 擴展為 [B, N, 1, dim]，作為來源節點表示
        d = od.unsqueeze(1)  # 將 od 從 [B, N, dim] 擴展為 [B, 1, N, dim]，作為目的節點表示
        # 更新後的邊特徵由來源信息、目的信息與原始（線性變換後的）邊特徵相加得到
        return o + d + edge

    @staticmethod
    def mask(x, adj):
        # 若鄰接矩陣 adj 為 None，則不進行 mask，直接返回 x
        if adj is None:
            return x
        # 將鄰接矩陣 adj 擴展最後一個維度，使其形狀變為 [B, N, N, 1]，以匹配 x 的最後一維
        adj = adj.unsqueeze(-1).expand(adj.shape[0], adj.shape[1], adj.shape[2], x.shape[-1])
        # 將鄰接矩陣中非連接位置（adj <= 0）對應的 x 值設為極小值 -9e15，
        # 這樣在後續 softmax 時，這些位置的概率將趨近於 0
        x = torch.where(adj > 0, x, -9e15 * torch.ones_like(x))
        return x

    def forward(self, node_fea, edge_fea, adj):
        """
        前向傳播過程：
        - node_fea: [B, nheads, N, node_in_dim]，批次中每個節點的多頭特徵
        - edge_fea: [B, nheads, N, N, edge_in_dim]，批次中每對節點間的多頭邊特徵
        - adj: 鄰接矩陣，用於指示圖中節點之間是否有連接（1 表示連接，0 表示無連接）
        返回更新後的節點特徵與邊特徵
        """
        first_layer = False  # 初始預設非第一層，稍後根據頭數判斷是否使用初始線性層
        B, H, N, _ = node_fea.shape  # 取得批次大小 B、頭數 H 和節點數 N

        # 將節點特徵從形狀 [B, nheads, N, node_in_dim] 轉換為 [B, N, nheads * node_in_dim]
        node_fea_new = node_fea.permute(0, 2, 1, 3).reshape(B, N, -1)
        # 將邊特徵從形狀 [B, nheads, N, N, edge_in_dim] 轉換為 [B, N, N, nheads * edge_in_dim]
        edge_fea_new = edge_fea.permute(0, 2, 3, 1, 4).reshape(B, N, N, -1)

        if H == self.nheads:
            # 當目前多頭數 H 與設定 nheads 相等，使用線性層 W 對合併後的節點特徵進行變換
            Wh = self.W(node_fea_new)  # Wh 形狀變為 [B, N, nheads * node_out_dim]
            # 同時對邊特徵使用線性層 a_edge 進行變換，得到 e_edge，形狀為 [B, N, N, nheads * 1]
            e_edge = self.a_edge(edge_fea_new)
        else:
            first_layer = True  # 如果多頭數不一致，認定當前為第一層
            # 使用初始線性層 W_init 對節點特徵進行變換，得到 Wh
            Wh = self.W_init(node_fea_new)
            # 以下 try/except 區段原本用於捕捉形狀不匹配錯誤，目前保留註解掉以供日後調試參考
            # try:
            e_edge = self.a_edge_init(edge_fea_new)  # 使用初始邊特徵線性變換，得到 e_edge
            # except:
            #     print(self.a_edge_init.weight.shape)
            #     print(edge_fea_new.shape)

        # 分別計算每個節點作為來源（source）與目標（target）的注意力投影
        # a1 將 Wh 映射為 [B, N, nheads * 1]，作為來源注意力投影
        Whi = self.a1(Wh)
        # a2 將 Wh 映射為 [B, N, nheads * 1]，作為目標注意力投影
        Whj = self.a2(Wh)

        if self.is_mix_attention:
            # 若啟用混合注意力，則計算注意力得分矩陣
            # 將來源注意力投影 Whi 擴展為 [B, N, 1, nheads * 1]
            # 將目標注意力投影 Whj 擴展為 [B, 1, N, nheads * 1]
            # 兩者相加得到初步注意力得分矩陣 e，形狀為 [B, N, N, nheads * 1]
            e = Whi.unsqueeze(2) + Whj.unsqueeze(1)
            # 將邊特徵投影 e_edge 加入，並使用 LeakyReLU 激活後進行非線性變換
            e = self.leakyrelu(e + e_edge)
            # 根據鄰接矩陣對非連接位置進行 mask 處理，將這些位置設為極小值，避免參與 softmax
            e = self.mask(e, adj)
            if self.is_update_edge:
                # 若啟用邊更新，則根據節點與原始邊特徵更新邊特徵
                edge_fea_new = self.update_edge(node_fea_new, edge_fea_new, first_layer)
                # 將更新後的邊特徵 reshape：
                # 原始形狀為 [B, N, N, nheads * edge_out_dim]，先 reshape 為 [B, N, N, nheads, edge_out_dim]
                # 再進行維度置換，變為 [B, nheads, N, N, edge_out_dim]
                edge_fea_new = edge_fea_new.reshape(B, N, N, self.nheads, -1).permute(0, 3, 1, 2, 4)
            else:
                # 若不更新邊，則保留原始邊特徵
                edge_fea_new = edge_fea
        else:
            # 若不使用混合注意力，僅計算節點間的注意力得分，不結合邊特徵
            e = Whi.unsqueeze(2) + Whj.unsqueeze(1)
            e = self.leakyrelu(e)
            e = self.mask(e, adj)
            edge_fea_new = edge_fea

        # 調整注意力得分矩陣的維度：
        # 將 e 從形狀 [B, N, N, nheads * 1] 轉換為 [B, nheads, N, N]
        attention = e.permute(0, 3, 1, 2)
        # 對每個頭的注意力得分進行 softmax 正規化，使得每個節點對其所有鄰居的注意力和為 1
        attention = F.softmax(attention, dim=-1)
        # 將 dropout 應用於注意力矩陣，防止過擬合
        attention = self.dropout(attention)

        # 將變換後的節點表示 Wh 重塑為 [B, N, nheads, node_out_dim]
        # 然後調整維度為 [B, nheads, N, node_out_dim] 以便與注意力矩陣相乘
        Wh = Wh.contiguous().view(B, N, self.nheads, self.node_out_dim).permute(0, 2, 1, 3)
        # 利用注意力矩陣對節點表示進行加權聚合：
        # torch.matmul(attention, Wh) 結果形狀為 [B, nheads, N, node_out_dim]
        # 並加上原始 Wh（殘差連接），得到更新後的節點表示
        node_fea_new = torch.matmul(attention, Wh) + Wh

        # 返回更新後的節點特徵和邊特徵
        return node_fea_new, edge_fea_new


class GAT_layer(nn.Module):
    def __init__(self, node_size, edge_size, hidden_size, nhead=6, is_mix_attention=False, is_update_edge=True):
        super(GAT_layer, self).__init__()  # 初始化父類 nn.Module
        self.is_mix_attention = is_mix_attention  # 保存是否使用混合注意力的標誌
        self.is_update_edge = is_update_edge      # 保存是否更新邊特徵的標誌
        if is_mix_attention:
            # 若使用混合注意力，定義線性層 We 對邊特徵進行映射，將 edge_size 映射到 hidden_size
            self.We = nn.Linear(edge_size, hidden_size)
            if is_update_edge:
                # 同時定義線性層 W_od 對節點特徵進行映射，生成更新邊特徵所需的信息
                self.W_od = nn.Linear(node_size, hidden_size)
        # 定義節點基本映射線性層，將 node_size 映射到 hidden_size
        self.W = nn.Linear(node_size, hidden_size)
        # 定義計算注意力的線性層 a1，對節點作為來源的表示進行投影（無偏置）
        self.a1 = nn.Linear(node_size, 1, bias=False)
        # 定義計算注意力的線性層 a2，對節點作為目標的表示進行投影（無偏置）
        self.a2 = nn.Linear(node_size, 1, bias=False)
        # 定義 LeakyReLU 激活函數，負斜率設為 0.2
        self.leakyrelu = nn.LeakyReLU(0.2)
        # 定義 dropout 層，機率設為 0.5
        self.dropout = nn.Dropout(p=0.5)

    def update_edge(self, node_fea, edge_fea):
        # 對節點特徵使用線性層 W_od 進行映射，生成更新邊特徵所需的信息
        od = self.W_od(node_fea)
        # 將 od 擴展為來源節點表示 o：形狀 [B, N, 1, hidden_size]
        o = od.unsqueeze(2)
        # 將 od 擴展為目的節點表示 d：形狀 [B, 1, N, hidden_size]
        d = od.unsqueeze(1)
        # 返回更新後的邊特徵：原始邊特徵加上來源和目的節點表示
        return edge_fea + o + d

    def forward(self, node_fea, edge_fea, adj):
        # node_fea: [B, N, node_size]，B 為批次大小，N 為節點數
        Wh = self.W(node_fea)  # 將節點特徵映射到隱藏空間，形狀變為 [B, N, hidden_size]

        # 分別計算節點作為來源和目標的注意力投影
        Whi = self.a1(Wh)  # 形狀 [B, N, 1]
        Whj = self.a2(Wh)  # 形狀 [B, N, 1]
        # 根據是否使用混合注意力計算注意力得分矩陣
        if self.is_mix_attention:
            # 將來源與目標投影相加：Whi + Whj.T 得到 [B, N, N]
            e_node = Whi + Whj.T
            # 對邊特徵使用線性層 We 進行映射，形狀 [B, N, N, hidden_size]
            Wh_edge = self.We(edge_fea)
            # 將邊特徵加入來源與目標投影後，使用 LeakyReLU 激活得到 e
            e = self.leakyrelu(e_node + Wh_edge)
            if adj is not None:
                # 根據鄰接矩陣對 e 中非連接位置設置極小值，保證 softmax 時這些位置近似 0
                e = torch.where(adj > 0, e, -9e15 * torch.ones_like(e))
            if self.is_update_edge:
                edge_fea_new = self.update_edge(node_fea, Wh_edge)
            else:
                edge_fea_new = edge_fea
        else:
            # 若不使用混合注意力，僅計算 Whi + Whj.T
            e = self.leakyrelu(Whi + Whj.T)
            # 根據鄰接矩陣對非連接位置設置極小值
            e = torch.where(adj > 0, e, -9e15 * torch.ones_like(e))
            edge_fea_new = edge_fea

        # 對注意力得分 e 進行 softmax 正規化
        attention = F.softmax(e)
        # 將 dropout 應用於注意力矩陣，防止過擬合
        attention = self.dropout(attention)

        # 利用注意力矩陣對節點特徵 Wh 進行加權聚合，並加上殘差連接
        node_fea_new = torch.matmul(attention, Wh) + Wh

        return node_fea_new, edge_fea_new


class GAT_encoder(nn.Module):
    def __init__(self, node_size, edge_size, hidden_size,
                 num_layers=3, nheads=4, is_mix_attention=True, is_update_edge=True, num_node=20):
        super(GAT_encoder, self).__init__()  # 初始化父類 nn.Module
        self.num_layers = num_layers         # 保存總層數
        self.gat = nn.ModuleList()            # 使用 ModuleList 存放多層 GAT 層
        # 第一層：使用原始節點與邊特徵，通過多頭 GAT_layer_multihead 進行編碼
        self.gat.append(GAT_layer_multihead(node_in_dim=node_size,
                                            edge_in_dim=edge_size,
                                            node_out_dim=hidden_size,
                                            edge_out_dim=hidden_size,
                                            nheads=nheads,
                                            is_mix_attention=is_mix_attention,
                                            is_update_edge=is_update_edge
                                            ))
        # 後續層：使用前一層輸出的隱藏表示作為輸入，堆疊多層以提取更高階的特徵
        for i in range(1, num_layers):
            self.gat.append(GAT_layer_multihead(node_in_dim=hidden_size,
                                                edge_in_dim=hidden_size,
                                                node_out_dim=hidden_size,
                                                edge_out_dim=hidden_size,
                                                nheads=nheads,
                                                is_mix_attention=is_mix_attention,
                                                is_update_edge=is_update_edge
                                                ))

    def forward(self, node_fea, edge_fea, adj=None):
        # 將原始節點特徵從形狀 [B, N, node_size] 擴展一個頭維度以符合多頭格式，變為 [B, 1, N, node_size]
        node_fea = node_fea.unsqueeze(1)
        # 將原始邊特徵從形狀 [B, N, N, edge_size] 擴展為 [B, 1, N, N, edge_size]
        edge_fea = edge_fea.unsqueeze(1)
        # 依次通過每一層 GAT 層進行編碼，逐層更新節點與邊特徵
        for i in range(self.num_layers):
            node_fea, edge_fea = self.gat[i](node_fea, edge_fea, adj)
            if i == self.num_layers - 1:
                # 在最後一層，將多頭結果取平均：
                # 原本 node_fea 的形狀為 [B, nheads, N, node_out_dim]，取平均後變為 [B, N, node_out_dim]
                node_fea = node_fea.mean(dim=1)
            else:
                # 其他層使用 ReLU 激活，增加非線性
                node_fea = F.relu(node_fea)
        # 返回最終更新後的節點特徵與邊特徵（邊特徵可用於後續任務，如路徑約束）
        return node_fea, edge_fea
