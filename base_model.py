import torch.nn as nn
from multimodel_hierar import MultiModel_hierar
from my_utils.utils import *


class Model(nn.Module):
    def __init__(self, config, params):
        super(Model, self).__init__()
        self.config = config
        # self.use_poi = config['model']['use_aoi']
        self.method = config['model']['method'].split('_')[-1]
        self.seq_len = config['model']['max_len']
        self.device = get_device(config)
        print(f'device: {self.device}')
        self.params = params

        # 離散特徵處理
        self.gps_embedding = nn.Linear(2, params['gps_embedding_size'])
        self.user_embedding = nn.Embedding(params['max_courier'] + 1, params['user_embedding_size'])
        self.weekday_embedding = nn.Embedding(8, params['weekday_embedding_size'])
        self.aoi_embedding = nn.Embedding(params['max_aoi'] + 1, params['aoi_embedding_size'])
        self.aoi_type_embedding = nn.Embedding(params['max_aoi_type'] + 1, params['aoi_type_embedding_size'])

        # 連續特徵處理
        # 連續特徵映射後的維度定為 16，即所有連續特徵最終都被轉換為 16 維的向量
        self.conti_fea_size = 16 
        # global feature中的3個連續特徵：working hours, driving speed, attendance
        self.global_conti_embedding = nn.Linear(3, self.conti_fea_size)
        # node feature的6個特徵：地理位置、快遞員與目的地的距離、截止時間、接單時間、距離截止時間剩餘時間、？？(no AOI ID、AOI種類)
        self.unpick_conti_embedding = nn.Linear(6, self.conti_fea_size)
        self.aoi_conti_embedding = nn.Linear(params['aoi_conti_size'], self.conti_fea_size)

        self.use_dipan = params['use_dipan']
        
        # 計算全局特徵的最終維度，通過拼接全局連續特徵（16 維）、快遞員嵌入向量（維度由 params['user_embedding_size'] 決定）和星期嵌入向量（由 params['weekday_embedding_size'] 決定）
        self.global_size = self.conti_fea_size + params['user_embedding_size'] + \
                               params['weekday_embedding_size']
        if self.use_dipan:
            self.dipan_embedding = nn.Embedding(params['max_dipan'] + 1, params['dipan_embedding_size'])
            self.global_size += params['dipan_embedding_size']

        # 快遞員特徵來自兩部分GPS信息(location & AOI)和AOI相關(AOI類型、AOI ID的嵌入維度)的離散特徵 
        self.courier_size = params['gps_embedding_size'] * 2 + params['aoi_type_embedding_size'] + \
                                params['aoi_embedding_size']

        # 未處理訂單的特徵維度：連續特徵+GPS信息(為什麼沒有AOI？)+AOI相關(AOI類型、AOI ID的嵌入維度)離散特徵+全局訊息
        self.unpick_size = self.conti_fea_size + params['gps_embedding_size'] + params['aoi_type_embedding_size'] + \
                           params['aoi_embedding_size'] + self.global_size

        # 論文中edge feature只有3，此處為5，其餘2項未知
        self.edge_size = 5
        # 此參數通常用於模型最後某一層的輸出尺寸，或作為 SortLSTM 等模塊的輸入尺寸，具體作用在 MultiModel_hierar 中使用。
        self.last_size = 8

        # AOI的特徵維度：AOI自身的連續特徵映射+ AOI的GPS + AOI的類型與ID嵌入 + 全局特徵
        self.aoi_fea_size = self.conti_fea_size + params['gps_embedding_size'] + params['aoi_type_embedding_size'] + \
                            params['aoi_embedding_size'] + self.global_size

        # 與論文中對 AOI 間基本邊特徵（例如距離、截止時間差、連通性）的描述是一致的
        self.aoi_edge_size = 3

        unpick_input_size = self.unpick_size
        self.model = MultiModel_hierar(params, self.last_size, unpick_input_size, self.edge_size,
                                       self.aoi_fea_size, self.aoi_edge_size, self.courier_size, self.device)

        # 將內部 MultiModel_hierar 模型的名稱保存到 self.model_name 中，用於後續識別或模型保存。
        self.model_name = self.model.model_name

        # 遍歷所有模型參數，對於維度大於 1 的參數採用 Xavier 均勻初始化，以促進梯度流動和模型穩定訓練。
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

    def forward(self, *data, is_train=True, label_eta=None, aoi_eta=None):
        unpick_fea, edge_fea, unpick_len, last_fea, last_len, global_fea, idx, pos, \
            aoi_index, aoi_fea, aoi_edge, aoi_len, aoi_idx, aoi_pos = data

        # 從 global_fea 的第一列提取快遞員 ID，轉換為整數型後通過 user_embedding 層得到快遞員的嵌入表示。
        user_embed = self.user_embedding(global_fea[:, 0].long())
        
        # 從 global_fea 的第4列（索引 3）提取星期信息，通過 weekday_embedding 得到其嵌入表示。
        weekday_embed = self.weekday_embedding(global_fea[:, 3].long())

        global_conti_fea = torch.cat([global_fea[:, 1:3], global_fea[:, 4:5]], dim=1)
        global_conti_fea = self.global_conti_embedding(global_conti_fea)

        # 從 global_fea 的第4列（索引 3）提取星期信息，通過 weekday_embedding 得到其嵌入表示。
        global_fea_new = torch.cat([global_conti_fea, user_embed, weekday_embed], dim=1)
        dipan_embed = self.dipan_embedding(global_fea[:, 11].long())
        global_fea_new = torch.cat([global_fea_new, dipan_embed], dim=1)

        # 第一行從 global_fea 提取第6-7列（索引 5:7）的 GPS 坐標（當前位置），通過 gps_embedding 映射得到表示。
        now_gps_embed = self.gps_embedding(global_fea[:, 5:7])

        # 第二行提取第10-11列（索引 9:11），作為所屬 AOI 的中心位置 GPS，同樣通過 gps_embedding 處理。
        now_aoi_gps_embed = self.gps_embedding(global_fea[:, 9:11])

        # 從 global_fea 的第8列（索引 7）提取 AOI ID，並通過 aoi_embedding 得到其嵌入表示。
        now_aoi_id_embed = self.aoi_embedding(global_fea[:, 7].long())
        
        # 從第9列（索引 8）提取 AOI 類型，通過 aoi_type_embedding 得到表示
        now_aoi_type_embed = self.aoi_type_embedding(global_fea[:, 8].long())

        # 將快遞相關的所有信息（當前位置 GPS、AOI 中心位置、AOI ID 及類型）拼接成一個完整的快遞特徵向量。
        courier_fea = torch.cat([now_gps_embed, now_aoi_gps_embed,
                                 now_aoi_id_embed, now_aoi_type_embed], dim=1)

        # 對於每個未處理訂單（unpick_fea 為三維張量，形狀為 [batch, seq, feature_dim]），提取第3列（索引 2）作為訂單所屬 AOI ID，並通過 aoi_embedding 映射
        
        # 提取第4列（索引 3）作為 AOI 類型，通過 aoi_type_embedding 映射
        unpick_aoi_embed = self.aoi_embedding(unpick_fea[:, :, 2].long())
        unpick_aoi_type_embed = self.aoi_type_embedding(unpick_fea[:, :, 3].long())

        # 提取每個未處理訂單的前兩列作為 GPS 坐標，通過 gps_embedding 映射，獲得其空間表示
        unpick_gps_embed = self.gps_embedding(unpick_fea[:, :, 0:2])
        
        # 從每個未處理訂單中提取從第5列開始的連續特徵（共 6 項，根據前面的設置），並通過 unpick_conti_embedding 映射到隱藏空間
        unpick_conti_embed = self.unpick_conti_embedding(unpick_fea[:, :, 4:])
        
        # 將未處理訂單的各部分特徵（GPS、AOI ID 嵌入、AOI 類型嵌入和連續特徵映射結果）拼接在一起，形成統一的訂單特徵表示
        unpick_fea_new = torch.cat([unpick_gps_embed, unpick_aoi_embed, unpick_aoi_type_embed, unpick_conti_embed], dim=2)

        # 從 AOI 特徵張量中提取第一列作為 AOI ID，並通過 aoi_embedding 映射
        aoi_id_embed = self.aoi_embedding(aoi_fea[:, :, 0].long())
        
        # 提取第二列作為 AOI 類型，通過 aoi_type_embedding 映射
        aoi_tpye_embed = self.aoi_type_embedding(aoi_fea[:, :, 1].long())

        # 從 AOI 特徵中提取第3-4列作為 AOI 的 GPS 坐標，映射到隱藏表示。
        aoi_gps_embed = self.gps_embedding(aoi_fea[:, :, 2:4])

        # 提取 AOI 特徵中從第5列開始的連續特徵，通過 aoi_conti_embedding 映射到統一的隱藏維度
        aoi_conti_embed = self.aoi_conti_embedding(aoi_fea[:, :, 4:])

        # 將 AOI 的所有部分特徵（GPS、ID 嵌入、類型嵌入、連續特徵映射）拼接起來，形成完整的 AOI 特徵表示
        aoi_fea = torch.cat([aoi_gps_embed, aoi_id_embed, aoi_tpye_embed, aoi_conti_embed], dim=2)

        return self.model(unpick_fea_new, edge_fea, unpick_len, courier_fea, global_fea_new, idx, pos,
                          aoi_index, aoi_fea, aoi_edge, aoi_len, aoi_idx, aoi_pos,
                          is_train, label_eta, aoi_eta)


if __name__ == '__main__':
    pass
