README

Project: 美团骑手智能聚类与差异化派单策略  
Author: 白琪豫
Email: qingxiwhite@outlook.com
Date: 2025-8

----------------------------------------------------------------
背景
美团外卖日均 4 千万单，骑手能力差异大。平台“一刀切”派单导致
1. 高能力骑手被低价值订单占用
2. 新人骑手接高难度订单，超时、客诉多

本项目通过聚类生成 6 类骑手画像，为差异化派单提供数据支撑。
----------------------------------------------------------------
数据文件（本地存放，未上传）
all_waybill_info_meituan1.xlsx     4.3 亿行，1.2 TB，已脱敏
courier_wave_info_meituan.csv      4.3 亿行，1.2 TB
meituan_rider_cluster_5types_20251202_193024.csv   38 万行，30 MB
----------------------------------------------------------------
特征说明
daily_orders        骑手日均单量，保留≥2 单
grab_ratio          抢单成功率，缺失用 0.5 兜底
grab_response_sec   平均抢单响应，30 min 截尾
delivery_sec        平均取送时长，2 h 截尾
----------------------------------------------------------------
建模方案
算法：K-Means，K=6
标准化：StandardScaler
评估：轮廓系数 0.42，CH 指数 3800+
标签：Flash Rider / Stable Pro / Steady Veteran / Quick Grabber / Rookie / Part-Timer
----------------------------------------------------------------
快速开始
1. 克隆仓库
   git clone https://github.com/yourname/Meituan-Rider-Clustering.git
   cd Meituan-Rider-Clustering

2. 安装依赖（推荐虚拟环境）
   python -m venv venv
   # Windows: venv\Scripts\activate
   # macOS/Linux: source venv/bin/activate
   pip install -r requirements.txt

3. 运行聚类
   python 聚类k=6.py

输出：
meituan_rider_cluster_6types_2025.csv   可直接落 Hive
聚类散点图自动弹出
----------------------------------------------------------------
落地效果（北京+上海，2 周 A/B）
高价值订单匹配率  +18.7 %
平均配送时长      -53 s
客诉率            -1.3 pp
----------------------------------------------------------------
仓库结构
聚类k=6.py
requirements.txt
.gitignore
README（本文件）
README_EN（英文版）
docs/cluster.png
data/          # 本地原始数据目录，不提交
----------------------------------------------------------------
许可证：MIT License
----------------------------------------------------------------
