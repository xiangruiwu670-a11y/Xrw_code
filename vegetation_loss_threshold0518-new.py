# -*- coding: utf-8 -*-
"""
复合干热事件植被损失触发阈值分析系统 v7.0 (Copula方法 — 全面修正版)
═══════════════════════════════════════════════════════════════════════
相较 v6.0 的修正清单：

Bug 1 ★★★ [fit_gamma_marginal / gamma_cdf / gamma_ppf]
  原因：fit_gamma_marginal 对偏移后的 x_pos 拟合，但 gamma_cdf 传入原始 x，
        导致 CDF 值不对应拟合分布，非正值区域 u≈0，整个概率转换失效。
  修正：params 元组增加第四元素 offset；gamma_cdf / gamma_ppf 统一施加偏移。

Bug 2 ★★★ [_solve_threshold_for_feature → _solve_threshold_trivariate_copula]
  原因：混淆 P(U2≤u2|U1=u1)（Copula条件CDF）与 P(损失超标|特征超标），
        用二变量条件CDF加权损失标记不等价于条件损失概率。
  修正：引入三变量高斯Copula，将损失变量纳入联合建模；
        以多元正态条件分布公式计算 P(U_loss > u_thresh | U_self=u*, U_other=u_med)；
        样本充足时以经验条件概率补充/替代解析值（混合策略）。

Bug 3 ★★★ [process_pixel 像元筛选]
  原因：np.sum(np.isnan(spei_series)) > 0 — 任意单个 NaN 即跳过有效像元。
  修正：改为 > N_MONTHS * 0.5，允许最多 50% 缺失。

Bug 4 ★★ [SPEI_SUM_THRESHOLD 方向]
  原因：SPEI_SUM_THRESHOLD=-500，条件 sum < -500 过滤掉最严重干旱像元。
  修正：改为 sum > -130（干旱信号不足时才跳过），与 v6.0 MK 版保持一致。

Bug 5 ★★ [run_theory_builtin 默认阈值]
  原因：run_theory_builtin(threshold=-1) 与全局 DROUGHT_THRESHOLD=-0.8 不符。
  修正：函数签名改为 threshold=DROUGHT_THRESHOLD（延迟绑定形式见代码）。

Bug 6 ★ ['time' 特征语义]
  原因：'time' 被赋值为事件序号 event_idx+1，无干旱物理意义。
  修正：改为 start_idx（事件起始月份绝对索引），表征干旱发生的季节时机。

Bug 7 ◎ [identify_top_loss_drivers 回退死代码]
  原因：回退 for 循环遍历 drought_features 查找 not in seen_features，
        但此时 seen_features 已包含 top_features 全部成员，循环永不新增。
  修正：改为直接从 feature_importance_df 按重要性顺序补齐第二特征。
═══════════════════════════════════════════════════════════════════════
"""

import numpy as np
import pandas as pd
import rasterio
import warnings
import os
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from tqdm import tqdm
import pickle

from scipy.stats import gamma as gamma_dist
from scipy.stats import norm

warnings.filterwarnings('ignore')

# ─── 外部 Run Theory 模块 ──────────────────────────────────────────────────
run_theory_path = "F:/Research_Paper/Thesis_WGM/Thesis-WGM/Code"
if run_theory_path not in sys.path:
    sys.path.insert(0, run_theory_path)

try:
    from run_theory import run_theory
    RUN_THEORY_IMPORTED = True
except ImportError as e:
    print(f"警告: 无法导入外部 run_theory 模块: {e}")
    RUN_THEORY_IMPORTED = False

# ─── 路径配置 ────────────────────────────────────────────────────────────────
inputpath_spei = "F:/Research_Paper/FSY_CDHEs/data/CDHEs3_ca_1982-2022.tif"
inputpath_ndvi = "F:/Research_Paper/Thesis_WGM/Thesis-WGM/data/NDVI/NDVI_1982_2022_monthly_ca.tif"
outpath        = "F:/Research_Paper/FSY_CDHEs/2021051418174-FSY/Result/Threshold"

# ─── 关键参数 ────────────────────────────────────────────────────────────────
INVALID_VALUE_THRESHOLD = -10000
NDVI_MIN_THRESHOLD      = -100
DROUGHT_THRESHOLD       = -0.8
MIN_DURATION            = 2
REVSA_METHOD            = "mean"
START_YEAR              = 1982
START_MONTH             = 1
TRIGGER_PROBABILITIES   = [0.3, 0.5, 0.7]
DROUGHT_FEATURES        = ['duration', 'severity', 'intensity', 'peak',
                            'interarrival', 'time']

# ★ Bug 4 修正：将筛选方向由"sum < -500（排除严重干旱）"
#   改为"sum > -130（排除干旱信号不足）"
SPEI_SUM_THRESHOLD_LOW  = -130   # 低于此值才认为干旱信号充足

# ★ Bug 3 修正：允许最多 50% 缺失（不再要求零缺失）
MAX_NAN_RATIO           = 0.5

SAVE_INTERVAL           = 500
CHECKPOINT_FILE         = os.path.join(outpath, "checkpoint_v7.pkl")
DEBUG_MODE              = True

# Copula 数值参数
COPULA_N_GRID           = 200   # 搜索网格点数
COPULA_MIN_SAMPLES      = 8     # Copula拟合最小样本量
COPULA_EMP_MIN_SUBSET   = 5     # 使用经验概率所需的最小子集样本量
COPULA_FALLBACK_EMPIRICAL = True


# ════════════════════════════════════════════════════════════════════════════
#  辅助工具
# ════════════════════════════════════════════════════════════════════════════

def log_message(message, level="INFO"):
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    symbols = {"INFO": "ℹ", "SUCCESS": "✓", "WARNING": "⚠",
               "ERROR": "✗", "DEBUG": "►"}
    print(f"[{ts}] {symbols.get(level,'•')} {message}")


def validate_input_data(spei_data, ndvi_data):
    errors, warnings_list = [], []
    if spei_data.shape != ndvi_data.shape:
        errors.append(f"SPEI与NDVI维度不匹配: {spei_data.shape} vs {ndvi_data.shape}")
    spei_valid = spei_data[(spei_data > INVALID_VALUE_THRESHOLD) & (spei_data < 10000)]
    if len(spei_valid) == 0:
        errors.append("SPEI数据全部无效")
    ndvi_valid = ndvi_data[(ndvi_data > INVALID_VALUE_THRESHOLD) & (ndvi_data < 10000)]
    if len(ndvi_valid) == 0:
        errors.append("NDVI数据全部无效")
    return errors, warnings_list


# ════════════════════════════════════════════════════════════════════════════
#  干旱事件识别
# ════════════════════════════════════════════════════════════════════════════

# ★ Bug 5 修正：内置版本的默认阈值改为全局常量，与外部模块行为一致
def run_theory_builtin(time_series, threshold=None):
    """
    内置游程理论干旱事件识别。
    threshold 默认使用全局 DROUGHT_THRESHOLD（= -0.8），
    修正了 v6.0 中写死 threshold=-1 导致的阈值不一致问题。
    """
    if threshold is None:
        threshold = DROUGHT_THRESHOLD   # Bug 5 修正

    events = []
    in_event = False
    start_idx = None
    event_values = []

    for i, val in enumerate(time_series):
        if np.isnan(val):
            if in_event and len(event_values) >= MIN_DURATION:
                _append_event(events, start_idx, i - 1, event_values)
            in_event = False
            start_idx = None
            event_values = []
            continue

        if val <= threshold:
            if not in_event:
                in_event = True
                start_idx = i
                event_values = [val]
            else:
                event_values.append(val)
        else:
            if in_event:
                if len(event_values) >= MIN_DURATION:
                    _append_event(events, start_idx, i - 1, event_values)
                in_event = False
                event_values = []

    if in_event and len(event_values) >= MIN_DURATION:
        _append_event(events, start_idx, len(time_series) - 1, event_values)

    return pd.DataFrame(events)


def _append_event(events, start_idx, end_idx, values):
    duration  = len(values)
    severity  = float(abs(np.sum(values)))
    intensity = severity / duration
    events.append({
        'Date_Ini_Ev': start_idx,
        'Date_Fin_Ev': end_idx,
        'Duration':    duration,
        'Severity':    severity,
        'Intesity':    intensity,
    })


def identify_drought_events(time_series, threshold=None):
    if threshold is None:
        threshold = DROUGHT_THRESHOLD
    if RUN_THEORY_IMPORTED:
        try:
            result = run_theory(time_series, threshold)
            return result if result is not None else pd.DataFrame()
        except Exception as e:
            if DEBUG_MODE:
                log_message(f"外部 run_theory 调用失败，使用内置版本: {e}", "WARNING")
    return run_theory_builtin(time_series, threshold)


# ════════════════════════════════════════════════════════════════════════════
#  植被损失（REVSA）计算
# ════════════════════════════════════════════════════════════════════════════

def calc_revsa(anomalies, method="mean"):
    negative = anomalies[anomalies < 0]
    if len(negative) == 0:
        return 0.0
    methods = {
        "min":  lambda n: np.min(n),
        "sum":  lambda n: np.sum(n),
        "mean": lambda n: np.mean(n),
        "zsum": lambda n: np.sum(n) / (np.std(anomalies) or 1),
    }
    return float(methods[method](negative))


def calculate_monthly_climatology(ndvi_series, start_month=1):
    n_months = len(ndvi_series)
    month_means = np.full(12, np.nan)
    for m in range(12):
        offset = (m - start_month + 1) % 12
        indices = np.arange(offset, n_months, 12)
        vals = ndvi_series[indices]
        vals = vals[~np.isnan(vals)]
        if len(vals) > 0:
            month_means[m] = np.mean(vals)
    return month_means


# ════════════════════════════════════════════════════════════════════════════
#  Bug 1 修正：Gamma 边际分布（含 offset 四元组参数）
# ════════════════════════════════════════════════════════════════════════════

def fit_gamma_marginal(x):
    """
    对一维正实数数组 x 拟合 Gamma 分布。

    ★ Bug 1 修正：
      v6.0 中 fit 时对 x_pos = x - min(x) + 1e-6 做了偏移，
      但 gamma_cdf() 传入原始 x，导致 CDF 与拟合参数不对应。
      修正方案：将 offset 存入参数元组第四位，
      gamma_cdf / gamma_ppf 统一施加同一偏移。

    返回
    ----
    (alpha, loc, beta, offset) 四元组，失败返回 None
    """
    x = np.asarray(x, dtype=float)
    x_min = np.nanmin(x)

    # 计算使所有值为正的最小偏移量
    offset = max(0.0, -x_min + 1e-6) if x_min <= 0 else 0.0
    x_pos  = x + offset   # 所有元素均 > 0

    try:
        alpha, loc, beta = gamma_dist.fit(x_pos, floc=0)
        if alpha <= 0 or beta <= 0:
            return None
        return (alpha, loc, beta, offset)   # ★ 返回四元组
    except Exception:
        return None


def gamma_cdf(x_val, params):
    """
    Gamma CDF：F(x) = P(X ≤ x)。
    ★ Bug 1 修正：施加拟合时相同的 offset。
    """
    alpha, loc, beta, offset = params
    x_shifted = np.asarray(x_val, dtype=float) + offset
    return gamma_dist.cdf(x_shifted, a=alpha, loc=loc, scale=beta)


def gamma_ppf(u, params):
    """
    Gamma 分位函数（CDF逆函数）。
    ★ Bug 1 修正：减去 offset 还原到原始空间。
    """
    alpha, loc, beta, offset = params
    return gamma_dist.ppf(u, a=alpha, loc=loc, scale=beta) - offset


# ════════════════════════════════════════════════════════════════════════════
#  三变量高斯 Copula（Bug 2 修正核心）
# ════════════════════════════════════════════════════════════════════════════

def estimate_gaussian_copula_matrix(u_list):
    """
    估计 k 变量高斯 Copula 的相关矩阵（最大似然估计）。

    参数
    ----
    u_list : list of k arrays，每个元素均已转换到 (0,1)

    返回
    ----
    rho_matrix : (k×k) numpy array，相关系数矩阵
    """
    eps = 1e-6
    Z   = np.column_stack([
        norm.ppf(np.clip(u, eps, 1 - eps)) for u in u_list
    ])
    rho = np.corrcoef(Z.T)

    # 对角线保持为 1，非对角线裁剪至 [-0.999, 0.999]
    k = rho.shape[0]
    for i in range(k):
        for j in range(k):
            if i != j:
                rho[i, j] = np.clip(rho[i, j], -0.999, 0.999)
            else:
                rho[i, j] = 1.0

    return rho


def _copula_conditional_loss_prob(u_self, u_other, u_loss_thresh, rho_matrix):
    """
    ★ Bug 2 修正核心函数

    计算三变量高斯 Copula 的条件概率：
        P(U_loss > u_loss_thresh | U_self = u_self, U_other = u_other)

    推导过程（多元正态条件分布）：
      设 Z_i = Φ^{-1}(U_i)，则 (Z_self, Z_other, Z_loss) ~ MVN(0, Σ)
      条件分布：
        Z_loss | Z_self=z1, Z_other=z2  ~  N(μ_c, σ²_c)
        μ_c    = Σ_{loss,cond} · Σ_{cond,cond}^{-1} · [z1, z2]^T
        σ²_c   = 1 - Σ_{loss,cond} · Σ_{cond,cond}^{-1} · Σ_{cond,loss}
      故：
        P(U_loss > u_thresh | U_self, U_other)
          = 1 - Φ( (z_thresh - μ_c) / σ_c )

    参数
    ----
    rho_matrix : 3×3 相关矩阵，顺序 = [self(0), other(1), loss(2)]

    返回
    ----
    float：P(损失超标 | 两特征条件)
    """
    eps = 1e-6
    z_self   = float(norm.ppf(np.clip(u_self,        eps, 1 - eps)))
    z_other  = float(norm.ppf(np.clip(u_other,       eps, 1 - eps)))
    z_thresh = float(norm.ppf(np.clip(u_loss_thresh, eps, 1 - eps)))

    # 分块矩阵：条件变量 = [0,1]，目标变量 = [2]
    Sigma_cc = rho_matrix[:2, :2]      # 2×2
    Sigma_ct = rho_matrix[:2,  2]     # 长度2的向量（cov of conditioning with target）
    Sigma_tt = rho_matrix[ 2,  2]     # 标量 = 1.0

    try:
        Sigma_cc_inv = np.linalg.inv(Sigma_cc)
    except np.linalg.LinAlgError:
        return 0.5   # 数值退化时回退到 0.5

    z_cond       = np.array([z_self, z_other])
    mu_c         = float(Sigma_ct @ Sigma_cc_inv @ z_cond)
    sigma2_c     = float(Sigma_tt - Sigma_ct @ Sigma_cc_inv @ Sigma_ct)
    sigma_c      = float(np.sqrt(max(sigma2_c, 1e-10)))

    # P(Z_loss > z_thresh | Z_self, Z_other)  =  1 - Φ(...)
    return float(1.0 - norm.cdf((z_thresh - mu_c) / sigma_c))


def _empirical_cdf_values(data):
    """
    计算数组各元素的经验 CDF 值（Hazen 公式，避免 0/1 端点）。
    返回与 data 等长的 u 数组，均在 (0,1) 内。
    """
    n    = len(data)
    rank = np.argsort(np.argsort(data)) + 1   # 1-based rank
    return (rank - 0.5) / n


# ════════════════════════════════════════════════════════════════════════════
#  自适应随机森林参数
# ════════════════════════════════════════════════════════════════════════════

def get_adaptive_rf_params(n_samples, n_features):
    base = {'max_features': 'sqrt', 'random_state': 42, 'n_jobs': 1}
    if n_samples < 20:
        return {**base, 'n_estimators': 30, 'max_depth': 3,
                'min_samples_split': 5, 'min_samples_leaf': 2}
    elif n_samples < 50:
        return {**base, 'n_estimators': 50, 'max_depth': 4,
                'min_samples_split': 4, 'min_samples_leaf': 2}
    elif n_samples < 100:
        return {**base, 'n_estimators': 80, 'max_depth': 5,
                'min_samples_split': 3, 'min_samples_leaf': 2,
                'oob_score': True}
    else:
        return {**base, 'n_estimators': 100, 'max_depth': 6,
                'min_samples_split': 3, 'min_samples_leaf': 2,
                'oob_score': True}


# ════════════════════════════════════════════════════════════════════════════
#  核心函数1：随机森林识别主导特征（含 Bug 7 修正）
# ════════════════════════════════════════════════════════════════════════════

def identify_top_loss_drivers(event_data, drought_features,
                               min_samples=10, pixel_id=None):
    """
    ★ Bug 7 修正：
      v6.0 回退逻辑遍历 drought_features 查找 not in seen_features，
      但 seen_features 已包含全部 top_features，for 循环永远不新增。
      修正：改为从已排序的 feature_importance_df 按重要性顺序补齐。
    """
    y = (event_data['revsa'] < 0).astype(int).values
    X = event_data[drought_features].values

    if len(y) < min_samples or np.unique(y).size < 2:
        return None, []

    X_scaled = StandardScaler().fit_transform(X)
    params   = get_adaptive_rf_params(len(X), X.shape[1])
    rf       = RandomForestClassifier(**params)
    rf.fit(X_scaled, y)

    if DEBUG_MODE and pixel_id is not None and pixel_id % 1000 == 0:
        acc = np.mean(y == rf.predict(X_scaled))
        msg = f"Pixel {pixel_id}: n={len(y)}, Acc={acc:.3f}"
        if hasattr(rf, 'oob_score_'):
            msg += f", OOB={rf.oob_score_:.3f}"
        log_message(msg, "DEBUG")

    importance_df = pd.DataFrame({
        'feature':    drought_features,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False).reset_index(drop=True)

    # ★ Bug 7 修正：直接从已排序表中顺序取前两个不重复特征
    top_features = importance_df['feature'].tolist()[:2]

    return importance_df, top_features


# ════════════════════════════════════════════════════════════════════════════
#  核心函数2：三变量 Copula 触发阈值计算（Bug 1 + Bug 2 完全重写）
# ════════════════════════════════════════════════════════════════════════════

def calculate_trigger_threshold_copula(event_data, target_features,
                                       probabilities=None,
                                       min_samples=COPULA_MIN_SAMPLES):
    """
    基于三变量高斯 Copula 计算干热事件触发阈值。

    完整流程（对应论文 §2.4.7）：
    ─────────────────────────────────────────────────────────────────────
    Step 1  提取两个关键特征 x1, x2 及植被损失指标 revsa
    Step 2  Gamma分布拟合 x1, x2 的边际分布（Bug 1修正：含 offset）
            revsa 使用经验CDF（Hazen公式）转换为均匀边际
    Step 3  将 x1, x2, revsa 全部转换为均匀边际 u1, u2, u_loss
    Step 4  估计三变量高斯 Copula 相关矩阵 Σ（3×3）
    Step 5  对每个目标概率 α，在特征网格上找最小 x* 满足：
              P(loss > median | feat_self ≥ x*, feat_other ≥ x_other_med) ≥ α
            ► 子集样本 ≥ COPULA_EMP_MIN_SUBSET 时：用经验条件概率
            ► 子集样本不足时：用三变量 Copula 解析条件概率（多元正态）
    ─────────────────────────────────────────────────────────────────────
    """
    if probabilities is None:
        probabilities = TRIGGER_PROBABILITIES

    # 初始化空结果
    nan_result = {
        f'prob_{int(p * 100)}pct': {f: np.nan for f in target_features}
        for p in probabilities
    }
    nan_result.update({'copula_rho': np.nan, 'method': 'failed'})

    n = len(event_data)
    if n < min_samples:
        return (_empirical_threshold_fallback(event_data, target_features, probabilities)
                if COPULA_FALLBACK_EMPIRICAL else nan_result)

    if len(target_features) < 2:
        return nan_result

    feat1, feat2 = target_features[0], target_features[1]
    x1    = event_data[feat1].values.astype(float)
    x2    = event_data[feat2].values.astype(float)
    revsa = event_data['revsa'].values.astype(float)

    # ── Step 2：Gamma 边际拟合（Bug 1 修正：参数为四元组含 offset）──
    params1 = fit_gamma_marginal(x1)
    params2 = fit_gamma_marginal(x2)

    if params1 is None or params2 is None:
        return (_empirical_threshold_fallback(event_data, target_features, probabilities)
                if COPULA_FALLBACK_EMPIRICAL else nan_result)

    # ── Step 3：均匀边际转换 ──
    eps = 1e-6
    u1    = np.clip(gamma_cdf(x1, params1), eps, 1 - eps)
    u2    = np.clip(gamma_cdf(x2, params2), eps, 1 - eps)
    # ★ Bug 2 修正：revsa 用经验CDF转换，直接纳入三变量 Copula
    u_loss = np.clip(_empirical_cdf_values(revsa), eps, 1 - eps)

    # 损失超标阈值：revsa < 中位数 → u_loss < 0.5（经验中位数）
    # 由于 _empirical_cdf_values 是连续近似，u_thresh ≈ 0.5
    revsa_median  = float(np.median(revsa))
    u_loss_thresh = float(np.clip(np.mean(revsa <= revsa_median), eps, 1 - eps))

    # ── Step 4：三变量 Copula 相关矩阵（feat1=0, feat2=1, loss=2）──
    rho_matrix = estimate_gaussian_copula_matrix([u1, u2, u_loss])
    rho_12     = float(rho_matrix[0, 1])

    results = {'copula_rho': rho_12, 'method': 'copula'}

    # 各特征的中位数（用于条件化另一特征）
    u_medians = {feat1: float(np.median(u1)), feat2: float(np.median(u2))}

    # ── Step 5：逐概率水平、逐特征求解阈值 ──
    feature_configs = [
        (feat1, x1, params1, u1, 0),   # (name, raw_x, params, u, col_idx_in_rho)
        (feat2, x2, params2, u2, 1),
    ]

    for prob in probabilities:
        prob_key = f'prob_{int(prob * 100)}pct'
        results[prob_key] = {}

        for (fname, x_arr, params, u_arr, col_idx) in feature_configs:

            # 另一特征的列索引与中位数
            other_col  = 1 - col_idx
            other_feat = feat2 if col_idx == 0 else feat1
            u_other_med = u_medians[other_feat]

            # 调整 rho_matrix 行列顺序为 [self, other, loss]
            order = [col_idx, other_col, 2]
            rho_sub = rho_matrix[np.ix_(order, order)]

            # 搜索网格（原始特征值空间）
            x_grid = np.linspace(
                np.percentile(x_arr, 5),
                np.percentile(x_arr, 95),
                COPULA_N_GRID
            )

            thresh = np.nan
            for x_star in x_grid:
                u_star = float(np.clip(gamma_cdf(x_star, params), eps, 1 - eps))

                # 事件子集：feat_self ≥ x*
                mask  = u_arr >= u_star
                n_sub = int(mask.sum())

                if n_sub >= COPULA_EMP_MIN_SUBSET:
                    # ── 经验条件概率（直接计数）──
                    loss_flag_sub = (revsa[mask] < revsa_median).astype(float)
                    cond_prob     = float(np.mean(loss_flag_sub))
                else:
                    # ── 三变量 Copula 解析条件概率（Bug 2 修正核心）──
                    # P(U_loss > u_thresh | U_self = u_star, U_other = u_other_med)
                    cond_prob = _copula_conditional_loss_prob(
                        u_self       = u_star,
                        u_other      = u_other_med,
                        u_loss_thresh= u_loss_thresh,
                        rho_matrix   = rho_sub
                    )

                if cond_prob >= prob:
                    thresh = float(x_star)
                    break

            results[prob_key][fname] = thresh

    return results


# ════════════════════════════════════════════════════════════════════════════
#  经验回退方法（样本不足时）
# ════════════════════════════════════════════════════════════════════════════

def _empirical_threshold_fallback(event_data, target_features, probabilities):
    """样本不足时的经验方法；标记 method='empirical'。"""
    results = {'copula_rho': np.nan, 'method': 'empirical'}
    revsa_median = event_data['revsa'].median()
    loss_flag    = (event_data['revsa'] < revsa_median).values

    for prob in probabilities:
        prob_key = f'prob_{int(prob * 100)}pct'
        results[prob_key] = {}
        for feat in target_features:
            x       = event_data[feat].values
            order   = np.argsort(x)
            x_s     = x[order]
            loss_s  = loss_flag[order]
            trigger = np.nan
            for i in range(len(x_s)):
                cond = loss_s[i:]
                if len(cond) < 3:
                    break
                if np.mean(cond) >= prob:
                    trigger = float(x_s[i])
                    break
            results[prob_key][feat] = trigger

    return results


# ════════════════════════════════════════════════════════════════════════════
#  核心函数3：单像元事件处理
# ════════════════════════════════════════════════════════════════════════════

def process_pixel_events(event_data, drought_features,
                          probabilities=None, pixel_id=None):
    """单像元：随机森林筛选特征 → 三变量 Copula 阈值计算。"""
    if probabilities is None:
        probabilities = TRIGGER_PROBABILITIES

    importance_df, top_features = identify_top_loss_drivers(
        event_data, drought_features, pixel_id=pixel_id
    )
    if not top_features or len(top_features) < 2:
        return None

    results = {
        'feature1':     top_features[0],
        'feature2':     top_features[1],
        'feature1_imp': float(importance_df.iloc[0]['importance']),
        'feature2_imp': float(importance_df.iloc[1]['importance']),
    }

    copula_results = calculate_trigger_threshold_copula(
        event_data, top_features, probabilities
    )

    results['copula_rho']    = copula_results.get('copula_rho', np.nan)
    results['copula_method'] = copula_results.get('method', 'failed')

    for prob in probabilities:
        prob_key  = f'prob_{int(prob * 100)}pct'
        feat_dict = copula_results.get(prob_key, {})
        for fk, fname in [('feature1', top_features[0]), ('feature2', top_features[1])]:
            results[f'{prob_key}_{fk}_trigger'] = feat_dict.get(fname, np.nan)

    results['avg_loss_prob'] = float(np.mean(
        (event_data['revsa'] < 0).astype(int).values
    ))
    return results


# ════════════════════════════════════════════════════════════════════════════
#  单像元完整处理流水线
# ════════════════════════════════════════════════════════════════════════════

def process_pixel(spei_series, ndvi_series, pixel_id):
    """
    ★ Bug 3 修正：NaN 比例超 50% 才跳过（不再要求零缺失）
    ★ Bug 4 修正：sum > SPEI_SUM_THRESHOLD_LOW 才跳过（信号不足），
                  sum < -130 的像元保留（严重干旱有研究价值）
    ★ Bug 6 修正：事件 'time' 特征改为 start_idx（发生月份绝对索引）
    """
    ndvi_series = ndvi_series.copy().astype(float)
    spei_series = spei_series.copy().astype(float)

    ndvi_series[ndvi_series < INVALID_VALUE_THRESHOLD] = np.nan
    spei_series[spei_series < INVALID_VALUE_THRESHOLD] = -3.0

    n_months = len(spei_series)

    # ── Bug 3 修正：允许最多 50% 的 SPEI 缺失 ──
    if np.sum(np.isnan(spei_series)) / n_months > MAX_NAN_RATIO:
        return None

    # ── Bug 4 修正：过滤干旱信号不足的像元（sum > -130），保留严重干旱像元 ──
    spei_sum = float(np.nansum(spei_series))
    if spei_sum > SPEI_SUM_THRESHOLD_LOW:          # 无显著累积干旱 → 跳过
        return None
    if float(np.nanmin(spei_series)) > -0.5:       # 从未发生任何干旱 → 跳过
        return None
    if np.sum(np.isnan(ndvi_series)) == n_months:  # NDVI 全缺失 → 跳过
        return None

    # ── 干旱事件识别 ──
    drought_events = identify_drought_events(spei_series, threshold=DROUGHT_THRESHOLD)
    if len(drought_events) == 0 or np.nansum(ndvi_series) == 0:
        return None

    valid_events = drought_events[
        drought_events['Duration'] >= MIN_DURATION
    ].copy().reset_index(drop=True)

    if len(valid_events) == 0:
        return None

    month_means = calculate_monthly_climatology(ndvi_series, start_month=START_MONTH)
    event_list  = []

    for event_idx, event in valid_events.iterrows():
        start_idx = int(event['Date_Ini_Ev'])
        end_idx   = int(event['Date_Fin_Ev'])

        if start_idx < 2 or end_idx >= n_months:
            continue

        event_ndvi = ndvi_series[start_idx: end_idx + 1]
        if np.any(event_ndvi[~np.isnan(event_ndvi)] < NDVI_MIN_THRESHOLD):
            continue

        months    = np.arange(start_idx, end_idx + 1) % 12
        vs_state  = month_means[months]
        vs        = ndvi_series[start_idx: end_idx + 1]
        anomalies = vs - vs_state
        revsa_val = calc_revsa(anomalies, method=REVSA_METHOD)
        dp        = float(abs(np.nanmin(spei_series[start_idx: end_idx + 1])))

        new_ev = {
            'revsa':    revsa_val,
            'duration': float(event['Duration']),
            'severity': float(event['Severity']),
            'intensity': float(event['Intesity']),
            'peak':     dp,
            'start_idx': start_idx,
            # ★ Bug 6 修正：用发生月份绝对索引代替事件序号
            'time':     float(start_idx),
        }

        if 'Interarrival' in event:
            new_ev['interarrival'] = float(event['Interarrival'])

        event_list.append(new_ev)

    if len(event_list) < 2:
        return None

    event_data = pd.DataFrame(event_list)

    # 计算到达间隔（若外部 run_theory 未提供）
    if 'interarrival' not in event_data.columns:
        start_times = event_data['start_idx'].values
        interarrival = np.concatenate([[np.nan], np.diff(start_times)])
        event_data['interarrival'] = interarrival

    event_data_clean = event_data.dropna(subset=DROUGHT_FEATURES + ['revsa'])

    if len(event_data_clean) < COPULA_MIN_SAMPLES:
        return None

    try:
        debug_id = (pixel_id[0] * 10000 + pixel_id[1]
                    if isinstance(pixel_id, tuple) else None)
        pixel_result = process_pixel_events(
            event_data_clean, DROUGHT_FEATURES,
            probabilities=TRIGGER_PROBABILITIES,
            pixel_id=debug_id
        )
        if pixel_result is None:
            return None
    except Exception as e:
        if DEBUG_MODE and isinstance(pixel_id, tuple):
            if pixel_id[0] % 50 == 0 and pixel_id[1] == 0:
                log_message(f"像元{pixel_id}处理失败: {e}", "WARNING")
        return None

    results = {
        'revsa_mean':       float(event_data_clean['revsa'].mean()),
        'duration_mean':    float(event_data_clean['duration'].mean()),
        'severity_mean':    float(event_data_clean['severity'].mean()),
        'intensity_mean':   float(event_data_clean['intensity'].mean()),
        'peak_mean':        float(event_data_clean['peak'].mean()),
        'interarrival_mean': float(event_data_clean['interarrival'].mean()),
        'n_events':         len(event_data_clean),
        'feature1_name':    pixel_result.get('feature1'),
        'feature1_imp':     pixel_result.get('feature1_imp', np.nan),
        'feature2_name':    pixel_result.get('feature2'),
        'feature2_imp':     pixel_result.get('feature2_imp', np.nan),
        'avg_loss_prob':    pixel_result.get('avg_loss_prob', np.nan),
        'copula_rho':       pixel_result.get('copula_rho', np.nan),
        'copula_method':    pixel_result.get('copula_method', 'failed'),
    }

    for prob in TRIGGER_PROBABILITIES:
        pi = int(prob * 100)
        for fk in ['feature1', 'feature2']:
            results[f'prob_{pi}pct_{fk}_threshold'] = \
                pixel_result.get(f'prob_{pi}pct_{fk}_trigger', np.nan)

    return pixel_id, results


# ════════════════════════════════════════════════════════════════════════════
#  检查点保存 / 加载
# ════════════════════════════════════════════════════════════════════════════

def save_checkpoint(result_dict, processed_pixels):
    try:
        with open(CHECKPOINT_FILE, 'wb') as f:
            pickle.dump({'result_dict': result_dict,
                         'processed_pixels': processed_pixels}, f)
        if DEBUG_MODE:
            log_message(f"已保存中间结果 ({len(processed_pixels):,} 像元)", "DEBUG")
    except Exception as e:
        log_message(f"保存中间结果失败: {e}", "WARNING")


def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, 'rb') as f:
                return pickle.load(f)
        except Exception:
            log_message("中间结果文件损坏，从头开始", "WARNING")
    return None


# ════════════════════════════════════════════════════════════════════════════
#  主函数
# ════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 80)
    print("  复合干热事件植被损失阈值分析系统 v7.0 — 三变量 Copula 全面修正版")
    print("  修正内容: Gamma偏移一致性 / 三变量Copula条件概率 / NaN筛选 /")
    print("            SPEI阈值方向 / run_theory默认值 / time特征 / 回退死代码")
    print("=" * 80)

    if RUN_THEORY_IMPORTED:
        log_message("已成功导入外部 Run Theory 模块", "SUCCESS")
    else:
        log_message(f"使用内置 Run Theory（DROUGHT_THRESHOLD={DROUGHT_THRESHOLD}）", "WARNING")

    log_message(f"干旱阈值: {DROUGHT_THRESHOLD}  最小持续: {MIN_DURATION}月  "
                f"SPEI累积筛选: sum > {SPEI_SUM_THRESHOLD_LOW}")
    log_message(f"Copula最小样本: {COPULA_MIN_SAMPLES}  "
                f"经验子集最小: {COPULA_EMP_MIN_SUBSET}  "
                f"NaN容忍率: {MAX_NAN_RATIO*100:.0f}%")

    # ── 读取数据 ──
    log_message("读取复合干热指数和NDVI数据...")
    with rasterio.open(inputpath_spei) as src:
        spei_data = src.read()
        profile   = src.profile
    with rasterio.open(inputpath_ndvi) as src:
        ndvi_data = src.read()

    errors, warns = validate_input_data(spei_data, ndvi_data)
    for e in errors:
        log_message(e, "ERROR")
    if errors:
        return
    for w in warns:
        log_message(w, "WARNING")

    n_bands, n_rows, n_cols = spei_data.shape
    total_pixels = n_rows * n_cols
    log_message(f"数据维度: {n_bands}步 × {n_rows}行 × {n_cols}列 = {total_pixels:,}像元")

    # ── 初始化结果字典 ──
    feature_codes = {
        'duration': 1, 'severity': 2, 'intensity': 3,
        'peak': 4, 'interarrival': 5, 'time': 6
    }

    result_dict = {
        'revsa_mean':        np.full((n_rows, n_cols), np.nan, dtype=np.float32),
        'duration_mean':     np.full((n_rows, n_cols), np.nan, dtype=np.float32),
        'severity_mean':     np.full((n_rows, n_cols), np.nan, dtype=np.float32),
        'intensity_mean':    np.full((n_rows, n_cols), np.nan, dtype=np.float32),
        'peak_mean':         np.full((n_rows, n_cols), np.nan, dtype=np.float32),
        'interarrival_mean': np.full((n_rows, n_cols), np.nan, dtype=np.float32),
        'n_events':          np.full((n_rows, n_cols), np.nan, dtype=np.float32),
        'feature1_code':     np.full((n_rows, n_cols), np.nan, dtype=np.float32),
        'feature1_imp':      np.full((n_rows, n_cols), np.nan, dtype=np.float32),
        'feature2_code':     np.full((n_rows, n_cols), np.nan, dtype=np.float32),
        'feature2_imp':      np.full((n_rows, n_cols), np.nan, dtype=np.float32),
        'avg_loss_prob':     np.full((n_rows, n_cols), np.nan, dtype=np.float32),
        'copula_rho':        np.full((n_rows, n_cols), np.nan, dtype=np.float32),
    }

    for prob in TRIGGER_PROBABILITIES:
        pi = int(prob * 100)
        result_dict[f'prob_{pi}pct_feature1_threshold'] = \
            np.full((n_rows, n_cols), np.nan, dtype=np.float32)
        result_dict[f'prob_{pi}pct_feature2_threshold'] = \
            np.full((n_rows, n_cols), np.nan, dtype=np.float32)

    # ── 检查点恢复 ──
    checkpoint = load_checkpoint()
    processed_pixels = set()
    if checkpoint:
        result_dict      = checkpoint['result_dict']
        processed_pixels = checkpoint['processed_pixels']
        log_message(f"从检查点恢复：已完成 {len(processed_pixels):,} 像元", "INFO")

    os.makedirs(outpath, exist_ok=True)

    processed_count = len(processed_pixels)
    valid_count     = 0
    copula_count    = 0
    empirical_count = 0

    with tqdm(total=total_pixels, initial=processed_count,
              desc="处理进度", unit="px", ncols=100) as pbar:
        for i in range(n_rows):
            for j in range(n_cols):
                if (i, j) in processed_pixels:
                    continue

                try:
                    res = process_pixel(spei_data[:, i, j],
                                        ndvi_data[:, i, j], (i, j))

                    if res is not None:
                        _, pr = res
                        valid_count += 1
                        if pr.get('copula_method') == 'copula':
                            copula_count += 1
                        else:
                            empirical_count += 1

                        # 写入统计字段
                        for key in ['revsa_mean', 'duration_mean', 'severity_mean',
                                    'intensity_mean', 'peak_mean', 'interarrival_mean',
                                    'n_events', 'avg_loss_prob', 'copula_rho']:
                            result_dict[key][i, j] = pr.get(key, np.nan)

                        result_dict['feature1_code'][i, j] = \
                            feature_codes.get(pr.get('feature1_name'), np.nan)
                        result_dict['feature1_imp'][i, j]  = pr.get('feature1_imp', np.nan)
                        result_dict['feature2_code'][i, j] = \
                            feature_codes.get(pr.get('feature2_name'), np.nan)
                        result_dict['feature2_imp'][i, j]  = pr.get('feature2_imp', np.nan)

                        # 写入阈值字段
                        for prob in TRIGGER_PROBABILITIES:
                            pi = int(prob * 100)
                            result_dict[f'prob_{pi}pct_feature1_threshold'][i, j] = \
                                pr.get(f'prob_{pi}pct_feature1_threshold', np.nan)
                            result_dict[f'prob_{pi}pct_feature2_threshold'][i, j] = \
                                pr.get(f'prob_{pi}pct_feature2_threshold', np.nan)

                    processed_pixels.add((i, j))
                    processed_count += 1

                    if processed_count % SAVE_INTERVAL == 0:
                        save_checkpoint(result_dict, processed_pixels)
                        pbar.set_postfix({
                            '有效': valid_count,
                            'Copula': copula_count,
                            '经验': empirical_count
                        })

                except Exception as e:
                    if DEBUG_MODE and i % 100 == 0 and j == 0:
                        log_message(f"像元({i},{j})异常: {e}", "WARNING")

                pbar.update(1)

    # ── 保存最终结果 ──
    log_message("正在保存最终结果栅格...")
    out_profile = profile.copy()
    out_profile.update(dtype='float32', count=len(result_dict), compress='lzw')
    output_file = os.path.join(outpath,
                               "CDHEs_vegetation_loss_copula_v7_thresholds.tif")

    with rasterio.open(output_file, 'w', **out_profile) as dst:
        for idx, (key, data) in enumerate(result_dict.items(), 1):
            dst.write(data.astype(np.float32), idx)
            dst.set_band_description(idx, key)

    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
        log_message("已清理检查点文件", "DEBUG")

    # ── 完成摘要 ──
    valid_pixels = int(np.sum(~np.isnan(result_dict['revsa_mean'])))
    print("\n" + "═" * 60)
    log_message("分析完成！", "SUCCESS")
    log_message(f"输出文件：{output_file}", "INFO")
    log_message(f"有效像元：{valid_pixels:,} / {total_pixels:,} "
                f"({valid_pixels / total_pixels * 100:.1f}%)", "INFO")
    log_message(f"Copula 方法：{copula_count:,}  经验方法：{empirical_count:,}", "INFO")

    log_message("输出波段说明：", "INFO")
    band_descriptions = {
        1: "revsa_mean           — 平均植被损失（REVSA）",
        2: "duration_mean        — 平均干旱持续时间（月）",
        3: "severity_mean        — 平均干旱严重度",
        4: "intensity_mean       — 平均干旱强度",
        5: "peak_mean            — 平均干旱峰值 |SPEI_min|",
        6: "interarrival_mean    — 平均到达间隔（月）",
        7: "n_events             — 有效事件总数",
        8: "feature1_code        — 主导特征编码（1-6）",
        9: "feature1_imp         — 主导特征重要性",
       10: "feature2_code        — 次级特征编码（1-6）",
       11: "feature2_imp         — 次级特征重要性",
       12: "avg_loss_prob        — 平均损失概率",
       13: "copula_rho           — Copula 相关系数 ρ₁₂",
    }
    start_band = 14
    for prob in TRIGGER_PROBABILITIES:
        pi = int(prob * 100)
        band_descriptions[start_band]     = f"prob_{pi}pct_feature1_threshold — {pi}%触发阈值（主导特征）"
        band_descriptions[start_band + 1] = f"prob_{pi}pct_feature2_threshold — {pi}%触发阈值（次级特征）"
        start_band += 2

    for band_num, desc in band_descriptions.items():
        log_message(f"  Band {band_num:02d}: {desc}", "INFO")

    print("═" * 60)


if __name__ == "__main__":
    main()
