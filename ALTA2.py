"""
ALT Web 应用 - 支持右删失数据
基于 reliability 库，支持单应力和双应力 ALT 数据拟合、模型自动选择、概率图展示
支持右删失（Right Censored）数据输入
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 导入 reliability 库的 ALT 相关模块
from reliability.ALT_fitters import (
    Fit_Weibull_Exponential,
    Fit_Weibull_Eyring,
    Fit_Weibull_Power,
    Fit_Lognormal_Exponential,
    Fit_Lognormal_Eyring,
    Fit_Lognormal_Power,
    Fit_Normal_Exponential,
    Fit_Normal_Eyring,
    Fit_Normal_Power,
    Fit_Exponential_Exponential,
    Fit_Exponential_Eyring,
    Fit_Exponential_Power,
    Fit_Weibull_Dual_Exponential,
    Fit_Weibull_Power_Exponential,
    Fit_Weibull_Dual_Power,
    Fit_Lognormal_Dual_Exponential,
    Fit_Lognormal_Power_Exponential,
    Fit_Lognormal_Dual_Power,
    Fit_Normal_Dual_Exponential,
    Fit_Normal_Power_Exponential,
    Fit_Normal_Dual_Power,
    Fit_Exponential_Dual_Exponential,
    Fit_Exponential_Power_Exponential,
    Fit_Exponential_Dual_Power,
    Fit_Everything_ALT,
)

# 单应力模型映射
SINGLE_STRESS_MODELS = {
    "Weibull_Exponential": Fit_Weibull_Exponential,
    "Weibull_Eyring": Fit_Weibull_Eyring,
    "Weibull_Power": Fit_Weibull_Power,
    "Lognormal_Exponential": Fit_Lognormal_Exponential,
    "Lognormal_Eyring": Fit_Lognormal_Eyring,
    "Lognormal_Power": Fit_Lognormal_Power,
    "Normal_Exponential": Fit_Normal_Exponential,
    "Normal_Eyring": Fit_Normal_Eyring,
    "Normal_Power": Fit_Normal_Power,
    "Exponential_Exponential": Fit_Exponential_Exponential,
    "Exponential_Eyring": Fit_Exponential_Eyring,
    "Exponential_Power": Fit_Exponential_Power,
}

# 双应力模型映射
DUAL_STRESS_MODELS = {
    "Weibull_Dual_Exponential": Fit_Weibull_Dual_Exponential,
    "Weibull_Power_Exponential": Fit_Weibull_Power_Exponential,
    "Weibull_Dual_Power": Fit_Weibull_Dual_Power,
    "Lognormal_Dual_Exponential": Fit_Lognormal_Dual_Exponential,
    "Lognormal_Power_Exponential": Fit_Lognormal_Power_Exponential,
    "Lognormal_Dual_Power": Fit_Lognormal_Dual_Power,
    "Normal_Dual_Exponential": Fit_Normal_Dual_Exponential,
    "Normal_Power_Exponential": Fit_Normal_Power_Exponential,
    "Normal_Dual_Power": Fit_Normal_Dual_Power,
    "Exponential_Dual_Exponential": Fit_Exponential_Dual_Exponential,
    "Exponential_Power_Exponential": Fit_Exponential_Power_Exponential,
    "Exponential_Dual_Power": Fit_Exponential_Dual_Power,
}


def parse_uploaded_data(uploaded_file, stress_type):
    df = pd.read_csv(uploaded_file)

    # 根据应力类型定义列名
    if stress_type == "单应力":
        stress_cols = ["stress"]
        rc_stress_cols = ["right_censored_stress"]
    else:
        stress_cols = ["stress_1", "stress_2"]
        rc_stress_cols = ["right_censored_stress_1", "right_censored_stress_2"]

    # 检查必须列是否存在
    required_cols = ["failure_time"] + stress_cols
    for col in required_cols:
        if col not in df.columns:
            st.error(f"CSV 文件必须包含列: {required_cols}")
            return None, None, None, None, None

    # 失效数据（只取 failure_time 非空的行）
    failures_mask = df["failure_time"].notna()
    failures = df.loc[failures_mask, "failure_time"].values.tolist()
    stress_lists = [df.loc[failures_mask, col].values.tolist() for col in stress_cols]

    # 右删失数据（如果存在）
    right_censored = None
    right_censored_stress_lists = None
    if all(col in df.columns for col in rc_stress_cols):
        # 检查是否所有右删失列都非空（至少有一个值）
        if not df[rc_stress_cols].isnull().all().all():
            # 去除缺失值（NaN）的行
            rc_df = df[["right_censored_time"] + rc_stress_cols].dropna()
            if not rc_df.empty:
                right_censored = rc_df["right_censored_time"].values.tolist()
                right_censored_stress_lists = [rc_df[col].values.tolist() for col in rc_stress_cols]

    # 返回原始 DataFrame 及其他数据
    return df, failures, stress_lists, right_censored, right_censored_stress_lists


def run_alt_analysis(failures, stress_lists, right_censored, right_censored_stress_lists,
                     model_func, use_level_stress, **kwargs):
    """
    执行 ALT 模型拟合，支持右删失
    """
    if len(stress_lists) == 1:
        # 单应力
        kwargs_single = {
            'failures': failures,
            'failure_stress': stress_lists[0],
            'use_level_stress': use_level_stress,
        }
        if right_censored is not None:
            kwargs_single['right_censored'] = right_censored
            kwargs_single['right_censored_stress'] = right_censored_stress_lists[0]
        return model_func(**kwargs_single, **kwargs)
    else:
        # 双应力
        kwargs_dual = {
            'failures': failures,
            'failure_stress_1': stress_lists[0],
            'failure_stress_2': stress_lists[1],
            'use_level_stress': use_level_stress,
        }
        if right_censored is not None:
            kwargs_dual['right_censored'] = right_censored
            kwargs_dual['right_censored_stress_1'] = right_censored_stress_lists[0]
            kwargs_dual['right_censored_stress_2'] = right_censored_stress_lists[1]
        return model_func(**kwargs_dual, **kwargs)


# ==================== Streamlit UI ====================
st.set_page_config(page_title="ALT 分析工具（支持右删失）", layout="wide")
st.title("🚀 加速寿命测试 (ALT) 分析工具")
st.markdown("### ✨ 支持右删失 (Right Censored) 数据，更贴近实际测试场景")
st.markdown("基于 `reliability` 库，支持单/双应力 ALT 数据拟合、模型自动选择与可视化。")

# 侧边栏：数据上传
with st.sidebar:
    st.header("📂 数据上传")
    st.markdown(
        """
        **CSV 格式要求**：
        - **必须包含**：`failure_time` 列
        - **单应力**：必须包含 `stress` 列
        - **双应力**：必须包含 `stress_1` 和 `stress_2` 列
        - **右删失（可选）**：
            - 单应力：`right_censored_time` 和 `right_censored_stress`
            - 双应力：`right_censored_time`、`right_censored_stress_1`、`right_censored_stress_2`
        - 缺失值用空单元格表示即可。
        """
    )
    uploaded_file = st.file_uploader("上传 CSV 文件", type=["csv"])

    st.header("⚙️ 分析设置")

    stress_type = st.radio(
        "应力类型",
        ["单应力", "双应力"],
        help="单应力：只有一个应力变量（如温度）；双应力：两个应力变量（如温度和电压）"
    )

    if stress_type == "单应力":
        model_dict = SINGLE_STRESS_MODELS
    else:
        model_dict = DUAL_STRESS_MODELS

    # 模型选择
    use_auto_select = st.checkbox("自动选择最佳模型 (Fit_Everything_ALT)", value=True)
    if not use_auto_select:
        selected_model = st.selectbox(
            "选择 ALT 模型",
            list(model_dict.keys()),
            help="共 12 种单应力模型 / 12 种双应力模型"
        )
        model_func = model_dict[selected_model]

    # 使用应力
    st.subheader("📌 使用应力 (Use Level Stress)")
    if stress_type == "单应力":
        use_stress = st.number_input("使用应力值", value=60.0, step=1.0)
    else:
        col1, col2 = st.columns(2)
        with col1:
            use_stress_1 = st.number_input("应力 1 使用值", value=330.0, step=1.0)
        with col2:
            use_stress_2 = st.number_input("应力 2 使用值", value=2.5, step=0.1)
        use_stress = [use_stress_1, use_stress_2]

    # 高级选项
    with st.expander("高级选项"):
        show_life_stress_plot = st.checkbox("显示寿命-应力图", value=True)
        show_probability_plot = st.checkbox("显示概率图", value=True)
        print_results = st.checkbox("打印详细结果", value=True)

    run_btn = st.button("▶ 运行分析", type="primary")


# ==================== 主区域 ====================
if run_btn and uploaded_file is not None:
    with st.spinner("正在分析数据..."):
        # 解析数据
        df_raw, failures, stress_lists, right_censored, right_censored_stress_lists = parse_uploaded_data(
            uploaded_file, stress_type
        )
        if failures is None:
            st.stop()

        st.success(f"✅ 数据加载成功：{len(failures)} 个失效数据")
        if right_censored is not None:
            st.info(f"📌 检测到右删失数据：{len(right_censored)} 个")

        # 显示数据预览
        with st.expander("📊 数据预览"):
            st.dataframe(df_raw)

        try:
            if use_auto_select:
                # 自动选择最佳模型
                st.info("正在尝试拟合所有 12 个模型，自动选择最佳...")
                if len(stress_lists) == 1:
                    result = Fit_Everything_ALT(
                        failures=failures,
                        failure_stress_1=stress_lists[0],
                        use_level_stress=use_stress,
                        right_censored=right_censored,
                        right_censored_stress_1=right_censored_stress_lists[0] if right_censored_stress_lists else None,
                        show_life_stress_plot=show_life_stress_plot,
                        show_probability_plot=show_probability_plot,
                        print_results=print_results,
                    )
                else:
                    result = Fit_Everything_ALT(
                        failures=failures,
                        failure_stress_1=stress_lists[0],
                        failure_stress_2=stress_lists[1],
                        use_level_stress=use_stress,
                        right_censored=right_censored,
                        right_censored_stress_1=right_censored_stress_lists[0] if right_censored_stress_lists else None,
                        right_censored_stress_2=right_censored_stress_lists[1] if right_censored_stress_lists else None,
                        show_life_stress_plot=show_life_stress_plot,
                        show_probability_plot=show_probability_plot,
                        print_results=print_results,
                    )
                st.success("✅ 自动选择完成！最佳模型已在图中展示。")
            else:
                # 手动选择模型
                st.info(f"正在拟合模型: {selected_model}")
                result = run_alt_analysis(
                    failures=failures,
                    stress_lists=stress_lists,
                    right_censored=right_censored,
                    right_censored_stress_lists=right_censored_stress_lists,
                    model_func=model_func,
                    use_level_stress=use_stress,
                    show_life_stress_plot=show_life_stress_plot,
                    show_probability_plot=show_probability_plot,
                    print_results=print_results,
                )
                st.success(f"✅ 模型 {selected_model} 拟合完成！")

            # 显示结果摘要
            st.subheader("📋 结果摘要")
            if hasattr(result, "mean_life"):
                st.metric("预测平均寿命 (使用应力下)", f"{result.mean_life:.2f}")
            if hasattr(result, "loglik"):
                st.metric("Log-Likelihood", f"{result.loglik:.4f}")
            if hasattr(result, "AICc"):
                st.metric("AICc", f"{result.AICc:.4f}")

            # 显示拟合参数
            if hasattr(result, "results"):
                st.subheader("📐 拟合参数")
                st.dataframe(result.results)

            # 显示图形（Streamlit 会自动捕获 matplotlib 输出）
            st.subheader("📈 分析图表")
            st.pyplot(plt)

            # 显示额外信息
            with st.expander("📖 模型解读与右删失说明"):
                st.markdown(
                    """
                    **右删失数据处理**：
                    - 当测试过程中部分样品未失效时，将其记录为右删失数据。
                    - 这些数据提供了“该样品至少存活到此时”的信息，有助于更准确估计寿命分布参数。
                    - 本工具自动检测 CSV 中是否包含 `right_censored_time` 及对应应力列，若存在则自动纳入分析。
                    
                    **ALT 概率图解读要点**：
                    1. **拟合优度**：虚线是否合理穿过所有数据点（含删失点用特殊标记）
                    2. **AICc / BIC**：值越小，模型拟合越好
                    3. **形状参数变化**：变化应 < 50%，否则可能表明不同应力下失效模式不同
                    
                    **单应力模型**：Exponential、Eyring、Power 三种寿命-应力关系 × 4种分布
                    **双应力模型**：Dual_Exponential、Power_Exponential、Dual_Power 三种组合 × 4种分布
                    """
                )

        except Exception as e:
            st.error(f"❌ 分析出错：{e}")
            st.markdown("请检查数据格式是否正确，或尝试更换模型。")

elif run_btn and uploaded_file is None:
    st.warning("⚠️ 请先上传 CSV 数据文件")


# ==================== 使用示例 ====================
with st.expander("📝 使用示例：包含右删失数据的 CSV 格式"):
    st.markdown("**单应力（含右删失）示例 CSV 内容：**")
    example_csv = """failure_time,stress,right_censored_time,right_censored_stress
1200,80,,
1350,80,,
1450,80,,
1500,80,,
,80,1650,80
,80,1700,80
800,100,,
850,100,,
900,100,,
1000,100,,
,100,1100,100
,100,1150,100
500,120,,
550,120,,
600,120,,
,120,700,120
,120,720,120"""
    st.code(example_csv, language="csv")
    st.markdown("**注意**：所有数值均为纯数字，不含任何注释或文字。缺失值用空单元格表示。")