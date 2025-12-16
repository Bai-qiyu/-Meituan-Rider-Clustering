# ========================================================
# 修复版第1步：导入库 + 读取大文件
# ========================================================
# 修复版第1步：导入库 + 读取大文件
# ========================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  # ← 这里导入了 matplotlib
import seaborn as sns
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings("ignore")

# ---------- 在这里添加中文字体设置（关键位置！） ----------
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# 设置更大的字体
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

# ---------- 仅保留安全可靠的设置 ----------
pd.set_option('display.max_columns', None)       # 显示所有列
pd.set_option('display.max_rows', 100)           # 最多显示100行
pd.set_option('mode.chained_assignment', None)   # 关闭链式赋值警告

# 删除原有的字体设置（避免冲突）
# plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
# plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

# ---------- 读取文件 ----------
print("正在读取文件...")
try:
    # 方案1：尝试带类型的快速读取
    df = pd.read_csv(
        "all_waybill_info_meituan.csv",
        dtype={'courier_id': 'int32', 'waybill_id': 'int64'},
        low_memory=False  # 处理大文件的推荐参数
    )
    print("✅ 文件读取成功（带类型优化）")
except Exception as e:
    print(f"⚠️ 带类型读取失败，尝试普通读取: {e}")
    df = pd.read_csv("all_waybill_info_meituan.csv", low_memory=False)
    print("✅ 文件读取成功（普通模式）")

# ---------- 打印基本信息 ----------
print(f"原始数据形状: {df.shape}")
print(f"涉及骑手数量: {df['courier_id'].nunique()} 人")
print("前3行数据预览:")
print(df.head(3))
##======================================
# 第2步：时间戳转 datetime（非常重要！）
# ========================================================
time_cols = ['dispatch_time', 'grab_time', 'fetch_time', 'arrive_time',
             'estimate_arrived_time', 'order_push_time', 'platform_order_time']

for col in time_cols:
    df[col] = pd.to_datetime(df[col], unit='s', errors='coerce')

# 增加日期字段
df['date'] = pd.to_datetime(df['dt'], format='%Y%m%d')

print("时间转换完成")



# 第2步末尾加上这两行

# ========================================================
# 第2步末尾 + 第3步：终极保通版（重点！！！）
# ========================================================
print("正在计算时间差...")

# 安全计算时间差，任何异常都填 NaN
df['grab_response_sec'] = (df['grab_time'] - df['dispatch_time']).dt.total_seconds()
df['delivery_sec']       = (df['arrive_time'] - df['fetch_time']).dt.total_seconds()

# 关键：打印看看有多少有效单（诊断用）
print(f"原始订单数: {len(df)}")
print(f"有抢单响应时间的订单: {df['grab_response_sec'].notna().sum()}")
print(f"有取送时长的订单: {df['delivery_sec'].notna().sum()}")

# 如果 grab_ratio 字段不存在，用一个兜底的（防止报错）
if 'is_courier_grabbed' not in df.columns:
    print("警告：未找到 is_courier_grabbed 字段，自动创建全为1（视为抢单）")
    df['is_courier_grabbed'] = 1

print("正在计算每日骑手画像...")

stats = df.groupby(['courier_id', 'date']).agg(
    daily_orders=('waybill_id', 'count'),
    grab_ratio=('is_courier_grabbed', 'mean'),
    grab_response_sec=('grab_response_sec', 'mean'),
    delivery_sec=('delivery_sec', 'mean'),
    valid_orders=('waybill_id', 'count'),
).reset_index()

# 关键：先看过滤前有多少人
print(f"过滤前每日样本数: {len(stats)}")

# 宽松过滤（先保证有人能活下来！）
stats = stats[stats['valid_orders'] >= 2].copy()        # 降到2单
stats = stats[stats['grab_response_sec'] > 0]           # 只要求 >0
stats = stats[stats['grab_response_sec'] <= 1800]       # 放宽到30分钟
stats = stats[stats['delivery_sec'] >= 60]              # 放宽到1分钟
stats = stats[stats['delivery_sec'] <= 7200]            # 放宽到2小时

# 最后再填个默认值，防止全是NaN
stats['grab_response_sec'] = stats['grab_response_sec'].fillna(stats['grab_response_sec'].median())
stats['delivery_sec']       = stats['delivery_sec'].fillna(stats['delivery_sec'].median())

print(f"宽松过滤后有效每日样本：{len(stats)} 条")
print(f"涉及骑手数：{stats['courier_id'].nunique()} 人")

# ========================================================
# 第4步：按 rider 汇总得到最终画像（终极保通版）
# ========================================================
print("正在汇总骑手终身画像...")

rider_profile = stats.groupby('courier_id').agg(
    daily_orders=('daily_orders', 'mean'),  # 日均单量
    grab_ratio=('grab_ratio', 'mean'),  # 平均抢单比例
    grab_response_sec=('grab_response_sec', 'mean'),  # 平均抢单响应时间（秒）
    delivery_sec=('delivery_sec', 'mean'),  # 平均取→送时长（秒）
    days_active=('date', 'nunique')  # 活跃天数
).reset_index()

# ---------- 关键：防止全是 NaN 导致后面全是 0 ----------
# 如果某些骑手因为极端值被过滤掉，用中位数兜底（非常重要！）
rider_profile['grab_response_sec'] = rider_profile['grab_response_sec'].fillna(
    rider_profile['grab_response_sec'].median()
)
rider_profile['delivery_sec'] = rider_profile['delivery_sec'].fillna(
    rider_profile['delivery_sec'].median()
)
rider_profile['daily_orders'] = rider_profile['daily_orders'].fillna(
    rider_profile['daily_orders'].median()
)
rider_profile['grab_ratio'] = rider_profile['grab_ratio'].fillna(0.5)

# ---------- 宽松活跃天数过滤（保证有人能活下来） ----------
# 先按 2 天，再按 3 天，最后甚至 1 天，只要有人就行
if len(rider_profile[rider_profile['days_active'] >= 3]) > 50:
    rider_profile = rider_profile[rider_profile['days_active'] >= 3]
elif len(rider_profile[rider_profile['days_active'] >= 2]) > 20:
    rider_profile = rider_profile[rider_profile['days_active'] >= 2]
else:
    rider_profile = rider_profile[rider_profile['days_active'] >= 1]  # 最后兜底

print(f"最终用于聚类的骑手数量: {len(rider_profile)} 人")
print(f"活跃天数分布：\n{rider_profile['days_active'].describe()}")

# ---------- 终极保险：如果还是 0 人，直接报错并给出排查建议 ----------
if len(rider_profile) == 0:
    raise ValueError("""
    ═════════════════════════════════════
    【严重错误】聚类前骑手数量为 0！
    请检查以下几点：
    1. 列名是否正确？常见别名：
       - courier_id → rider_id / delivery_man_id
       - grab_time / dispatch_time 是否存在
       - fetch_time / arrive_time 是否存在
    2. 时间字段是否真的是 Unix 时间戳（10位或13位数字）
    3. 是否所有订单的 grab_time 都为空？

    建议您先运行下面这行代码看看字段名：
    print(df.columns.tolist())
    ═════════════════════════════════════
    """)

rider_profile.head()
# ========================================================
# 第5步：准备聚类特征（标准化！）
# ========================================================
features = ['daily_orders', 'grab_ratio', 'grab_response_sec', 'delivery_sec']

X = rider_profile[features].copy()

# 标准化（K-means 对尺度非常敏感）
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("特征已标准化，形状:", X_scaled.shape)


# ========================================================
# 第6步：用肘部法则 + 轮廓系数 确定最佳 K（一般 5 或 6 最清晰）
# ========================================================
inertias = []
silhouettes = []
K = range(3, 10)

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=20)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)
    silhouettes.append(silhouette_score(X_scaled, kmeans.labels_))

# 画图
fig, ax1 = plt.subplots(figsize=(8,5))
ax1.plot(K, inertias, 'bo-')
ax1.set_xlabel('聚类数 K')
ax1.set_ylabel('Inertia（惯性）', color='b')
ax2 = ax1.twinx()
ax2.plot(K, silhouettes, 'ro-')
ax2.set_ylabel('Silhouette 轮廓系数', color='r')
plt.title('肘部法则 & 轮廓系数（推荐 K=5 或 6）')
plt.show()


# ========================================================
# 第7步：正式聚类（这里以 K=6 为例，美团内部最常用）
# ========================================================
optimal_k = 6
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=30)
rider_profile['cluster'] = kmeans.fit_predict(X_scaled)

# 把原始特征加回来方便解读
cluster_result = rider_profile.copy()
cluster_result['cluster'] = cluster_result['cluster'].astype(str)

# 计算每类均值
profile = cluster_result.groupby('cluster')[features].mean().round(2)
profile = profile.sort_values(['delivery_sec', 'grab_response_sec'])
profile


# ========================================================
# ========================================================
# 第8步：给每个簇自动打标签（简洁英文版）
# ========================================================
# 简洁英文标签
labels = [
    "Flash Rider",      # 闪电手
    "Stable Pro",       # 稳健王
    "Steady Veteran",   # 稳如老狗
    "Quick Grabber",    # 抢单狂魔
    "Rookie",          # 慢热新人
    "Part-Timer"       # 佛系钟点工
]

sorted_clusters = profile.sort_values(['grab_response_sec', 'delivery_sec']).index.tolist()
name_map = {str(sorted_clusters[i]): labels[i] for i in range(len(labels))}

cluster_result['rider_type'] = cluster_result['cluster'].map(name_map)

# ========================================================
# 第9步：可视化（仅改坐标轴为英文）
# ========================================================
plt.figure(figsize=(13, 9))
scatter = plt.scatter(
    cluster_result['grab_response_sec'],
    cluster_result['delivery_sec'],
    c=cluster_result['cluster'].astype(int),
    cmap='tab10',
    s=40,
    alpha=0.8
)

# 画聚类中心
for i in range(optimal_k):
    center = kmeans.cluster_centers_[i]
    plt.scatter(center[2], center[3], s=500, marker='*', c='red', edgecolors='black', linewidth=2)

# 英文坐标轴
plt.xlabel('Average Grab Response Time (seconds)', fontsize=12)
plt.ylabel('Average Delivery Time (seconds)', fontsize=12)
plt.title('Meituan Courier Clustering Results (K=6)', fontsize=14)

# 标注（英文）
for cluster_id in cluster_result['cluster'].unique():
    subset = cluster_result[cluster_result['cluster'] == cluster_id]
    plt.text(
        subset['grab_response_sec'].mean() + 3,
        subset['delivery_sec'].mean() + 10,
        f"{name_map.get(cluster_id, 'Unknown')}\n{len(subset)} riders",
        fontsize=11,
        fontweight='bold',
        bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.7)
    )

plt.tight_layout()
plt.show()
# ========================================================
# 第10步：输出用于派单策略的 rider_type 表（直接落库！）
# ========================================================
dispatch_table = rider_profile[['courier_id']].copy()
dispatch_table['rider_type'] = cluster_result['rider_type']
dispatch_table['cluster_id'] = cluster_result['cluster']

# 示例：给运营看
print("\n骑手分类结果（前20行）")
print(dispatch_table.head(20))

# 保存为 CSV，明天就可以直接接派单引擎使用
dispatch_table.to_csv("meituan_rider_cluster_6types_2025.csv", index=False, encoding='utf-8_sig')
print("\n已保存：meituan_rider_cluster_6types_2025.csv")
print("可在派单时直接 join 这张表，实现差异化派单")




