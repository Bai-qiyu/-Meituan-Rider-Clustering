# Meituan-Rider-Clustering  
**Language**: 中文 | [English](README_EN.md)  

## 1. 项目背景
美团外卖日均 4 千万单，骑手能力差异大，平台派单「一刀切」导致高能力骑手被低价值订单占用。  
本项目通过聚类生成 6 类骑手画像，为差异化派单提供数据支撑。

## 2. 数据说明
- 订单表 `all_waybill_info_meituan1.xlsx`（4.3 亿行，1.2 TB，已脱敏）  
- 骑手宽表 `meituan_rider_cluster_5types_20251202_193024.csv`（38 万行）

## 3. 特征工程
| 特征 | 含义 | 分箱/异常处理 |
|---|---|---|
| daily_orders | 骑手日均单量 | 宽松过滤 ≥2 单 |
| grab_ratio | 抢单成功率 | 缺失用 0.5 兜底 |
| grab_response_sec | 平均抢单响应 | 30 min 截尾 |
| delivery_sec | 取送时长 | 2 h 截尾 |

## 4. 建模
- 算法：K-Means，K=6（轮廓系数 0.42）  
- 标准化：StandardScaler  
- 聚类标签：Flash Rider / Stable Pro / Steady Veteran / Quick Grabber / Rookie / Part-Timer

## 5. 快速开始
```bash
# 1. 克隆仓库
git clone https://github.com/yourname/Meituan-Rider-Clustering.git
cd Meituan-Rider-Clustering

# 2. 创建虚拟环境（可选）
python -m venv venv
source venv/bin/activate  # Windows 用 venv\Scripts\activate

# 3. 安装依赖
pip install -r requirements.txt

# 4. 运行聚类
python "聚类k=6.py"
