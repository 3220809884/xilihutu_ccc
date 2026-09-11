"""Generate problem-3 paper tables and a concise model/results chapter."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "问题3" / "结果"
DATES = ("2025-03-20", "2025-06-21", "2025-09-23", "2025-12-21")


def fmt(value):
    return f"{float(value):.4f}"


def main() -> None:
    summary = json.loads((OUT / "problem3_summary.json").read_text(encoding="utf-8"))
    frame = pd.read_csv(OUT / "problem3_dispatch_all_2025.csv")
    frame = frame[frame.date >= "2025-02-01"]
    daily = pd.read_csv(OUT / "problem3_daily_summary.csv").set_index("date")
    events = pd.read_csv(OUT / "problem3_emergency_events.csv")
    comparison = pd.read_csv(OUT / "problem3_update_policy_comparison.csv")
    stage = pd.read_csv(OUT / "problem3_stage_metrics.csv")
    stage = stage.rename(columns={"mean_absolute_adjustment_kwh": "absolute_adjustment_kwh"})
    md = [
        "# 问题3：基于多时刻光伏预报的滚动MPC", "",
        "## 1. 模型与信息边界", "",
        "每天0:00使用附件3当时发布的未来24小时光伏预报形成144个10分钟时段的原计划。6:00、12:00、18:00只重算尚未开始的时段；已执行购电、储能动作与SOC全部锁定。负载预测仅使用目标日之前已完成日期的同期负载，不能读取当天未来实际负载。实际负载和光伏只进入当前时段的反馈执行与历史回测。", "",
        "功率变量单位为kW，时间步长Δt=1/6小时，电量与SOC单位为kWh。充、放电效率均为0.9，接口功率上限均为5000 kW，SOC保持在1200—10800 kWh，单时段由二元变量保证充放电互斥。实际日末SOC传递到下一天，不能每日重置。", "",
        "附件3的‘预报h小时’按发布时间+h解释。整点预报通过分段线性插值转换到10分钟区间结束时刻；发布时间处以当时已经获得的光伏实测值作为插值锚点。0:00、6:00、12:00、18:00预报分别用于0:00—6:00、6:00—12:00、12:00—18:00、18:00—24:00最终执行区间。", "",
        "负载预测采用最近W个完整日同期负载的α经验分位数。用1月15—31日的严格历史回测选择参数，选定W=%d、α=%.2f，并于2月1日冻结。" % (summary["selected_window_days"], summary["selected_load_quantile"]), "",
        "## 2. 计划与调整优化", "",
        "0:00计划MILP最小化Σπ_t q_t^plan Δt。每次更新从当前实际SOC出发，对剩余时段满足预测供需平衡、SOC动态、上下界、功率上限、互斥与计划末端SOC=6000 kWh。6000 kWh是有限时域储备假设，只约束预测轨迹；实际SOC不强制回到6000。", "",
        "调整结算采用：C_adjust=Σ[1.5π_t(q_t^adj−q_t^plan)_+−0.5π_t(q_t^plan−q_t^adj)_+]Δt。它表示原计划费先全额计入，减少计划量时退回原价50%，增加量按原价1.5倍补计。调整MILP以该凸分段线性费用最小，比较基准始终是当天0:00原计划。", "",
        "实际执行固定当前有效的调整购电量。富余时优先在功率与SOC上限内充电，缺口时优先放电，剩余缺口按5π_t紧急购电；下一预报时刻以实际SOC重新滚动求解。", "",
        "## 3. 全期结果与预报时刻价值", "",
        "| 指标 | 数值 |", "|---|---:|",
        f"| 计划购电量（kWh） | {fmt(summary['plan_kwh'])} |",
        f"| 最终调整购电量（kWh） | {fmt(summary['adjusted_kwh'])} |",
        f"| 增加/减少调整量（kWh） | {fmt(summary['increase_kwh'])} / {fmt(summary['decrease_kwh'])} |",
        f"| 紧急购电量（kWh） | {fmt(summary['emergency_kwh'])} |",
        f"| 计划购电费（元） | {fmt(summary['plan_cost_yuan'])} |",
        f"| 调整相关费用（元） | {fmt(summary['adjustment_cost_yuan'])} |",
        f"| 紧急购电费（元） | {fmt(summary['emergency_cost_yuan'])} |",
        f"| 总购电费（元） | {fmt(summary['total_cost_yuan'])} |",
        f"| 2月1日初/12月31日末SOC（kWh） | {fmt(summary['feb1_initial_soc_kwh'])} / {fmt(summary['dec31_terminal_soc_kwh'])} |", "",
        "逐步加入预报时刻的全年对比：", "",
        "| 使用预报时刻 | 总费用（元） | 相对仅0:00节省（元） | 新增时刻边际节省（元） | 紧急购电量（kWh） |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in comparison.itertuples(index=False):
        md.append(f"| {row.forecast_times} | {fmt(row.total_cost_yuan)} | {fmt(row.saving_vs_0_yuan)} | {fmt(row.marginal_saving_yuan)} | {fmt(row.emergency_kwh)} |")
    md += [
        "", f"完整滚动策略比只采用0:00预报少支出{fmt(summary['saving_from_updates_yuan'])}元，降幅为{100*summary['saving_from_updates_ratio']:.4f}%。各策略末SOC相同，因此该比较不受期末库存差异驱动。6:00、12:00、18:00三个新增时刻的边际节省均为正，故在当前数据、结算规则和反馈假设下均应引入。18:00以后光伏接近零，其价值主要来自用最新实际SOC重算晚间购电，而非夜间光伏预测本身。", "",
        "最终执行区间的光伏预报误差与调整规模：", "",
        "| 发布时间 | 光伏MAE（kW） | 绝对调整量（kWh） | 净调整量（kWh） |", "|---:|---:|---:|---:|",
    ]
    for row in stage.itertuples(index=False):
        md.append(f"| {int(row.forecast_issue_hour)}:00 | {fmt(row.pv_forecast_mae_kw)} | {fmt(row.absolute_adjustment_kwh)} | {fmt(row.net_adjustment_kwh)} |")
    md += ["", "## 4. 指定日期结果", "", "表1采用最终有效的调整购电量；全天购电费为计划费与调整相关费用之和，不含表3紧急费用。模板的时段文字整体较内部物理区间晚10分钟，按slot位置写入，论文时段按真实物理区间取值。", ""]
    tex = ["% UTF-8 problem-3 table fragment; compile in a ctex/XeLaTeX document."]
    for date in DATES:
        group = frame[frame.date == date].reset_index(drop=True)
        row = daily.loc[date]
        md += [f"### {date}", "", "表1：最终购电量及普通购电结算费", "", "| 时间段 | 购电量 | 时间段 | 购电量 | 时间段 | 购电量 |", "|---|---:|---|---:|---|---:|"]
        tex += [r"\begin{table}[htbp]\centering", f"\\caption{{{date}最终购电量及普通购电结算费}}", r"\begin{tabular}{|c|r|c|r|c|r|}\hline", r"时间段 & 购电量 & 时间段 & 购电量 & 时间段 & 购电量 \\ \hline"]
        for slots in ((60, 72, 84), (96, 108, 120)):
            cells = []
            for slot in slots:
                cells += [f"{slot//6:02d}:00-{slot//6:02d}:10", fmt(group.adjusted_kwh.iloc[slot])]
            md.append("| " + " | ".join(cells) + " |")
            tex.append(" & ".join(cells) + r" \\ \hline")
        md += [f"| 全天购电量 | {fmt(row.adjusted_kwh)} | 全天购电费 | {fmt(row.ordinary_settlement_cost_yuan)} | | |", "", "表2：实际充放电量与首末SOC", "", "| 时间段 | 充电量 | 放电量 | 时间段 | 充电量 | 放电量 |", "|---|---:|---:|---|---:|---:|"]
        charges = group.charge_kwh.to_numpy().reshape(6, 24).sum(axis=1)
        discharges = group.discharge_kwh.to_numpy().reshape(6, 24).sum(axis=1)
        storage_tex = [r"\begin{table}[htbp]\centering", f"\\caption{{{date}实际储能调度}}", r"\begin{tabular}{|c|r|r|c|r|r|}\hline", r"时间段 & 充电量 & 放电量 & 时间段 & 充电量 & 放电量 \\ \hline"]
        for start in (0, 2, 4):
            cells = []
            for block in (start, start + 1):
                cells += [f"{4*block}:00-{4*block+4}:00", fmt(charges[block]), fmt(discharges[block])]
            md.append("| " + " | ".join(cells) + " |")
            storage_tex.append(" & ".join(cells) + r" \\ \hline")
        md += [f"| 0:00储电量 | {fmt(row.soc_start_kwh)} | | 24:00储电量 | {fmt(row.soc_end_kwh)} | |", "", f"该日计划费{fmt(row.plan_cost_yuan)}元，调整相关费用{fmt(row.adjustment_cost_yuan)}元，紧急费用{fmt(row.emergency_cost_yuan)}元，总费用{fmt(row.total_cost_yuan)}元。", ""]
        tex += [rf"\multicolumn{{2}}{{|c|}}{{全天购电量}} & {fmt(row.adjusted_kwh)} & \multicolumn{{2}}{{c|}}{{全天购电费}} & {fmt(row.ordinary_settlement_cost_yuan)} \\ \hline", r"\end{tabular}\end{table}"]
        storage_tex += [rf"\multicolumn{{2}}{{|c|}}{{0:00储电量}} & {fmt(row.soc_start_kwh)} & \multicolumn{{2}}{{c|}}{{24:00储电量}} & {fmt(row.soc_end_kwh)} \\ \hline", r"\end{tabular}\end{table}"]
        tex += storage_tex
    md += ["## 5. 表3：指定日期紧急购电", "", "同日连续正紧急购电时段合并，购电量为其中各10分钟电量之和。", "", "| 2025.3.20 | | 2025.6.21 | | 2025.9.23 | | 2025.12.21 | |", "|---|---:|---|---:|---|---:|---|---:|", "| 时间段 | 购电量 | 时间段 | 购电量 | 时间段 | 购电量 | 时间段 | 购电量 |"]
    groups = [events[events.date == date].to_dict(orient="records") for date in DATES]
    for group in groups:
        if not group:
            group.append({"interval": "无", "emergency_kwh": 0})
    for index in range(max(map(len, groups))):
        cells = []
        for group in groups:
            cells += [group[index]["interval"], fmt(group[index]["emergency_kwh"])] if index < len(group) else ["", ""]
        md.append("| " + " | ".join(cells) + " |")
    totals = []
    for date in DATES:
        totals += ["合计", fmt(daily.loc[date].emergency_kwh)]
    md += ["| " + " | ".join(totals) + " |", "", "## 6. 校验与局限", "", f"交付包含334×144=48096个计划值和同样数量的最终调整值。供需平衡最大残差{summary['checks']['max_balance_residual_kw']:.3g} kW，SOC转移最大残差{summary['checks']['max_soc_transition_residual_kwh']:.3g} kWh，跨时段及跨日SOC残差为0，无同时充放电。", "", "历史负载分位数、计划末端6000 kWh和实时优先补缺反馈均为建模假设。结果证明所实现策略在历史数据上有效，不代表含全部预测不确定性的随机动态控制全局最优。", ""]
    tex += [r"\begin{table}[htbp]\centering", r"\caption{指定日期紧急购电量}", r"\small\begin{tabular}{|c|r|c|r|c|r|c|r|}\hline", r"\multicolumn{2}{|c|}{2025.3.20} & \multicolumn{2}{c|}{2025.6.21} & \multicolumn{2}{c|}{2025.9.23} & \multicolumn{2}{c|}{2025.12.21} \\ \hline", r"时间段 & 购电量 & 时间段 & 购电量 & 时间段 & 购电量 & 时间段 & 购电量 \\ \hline"]
    for index in range(max(map(len, groups))):
        cells = []
        for group in groups:
            cells += [group[index]["interval"], fmt(group[index]["emergency_kwh"])] if index < len(group) else ["", ""]
        tex.append(" & ".join(cells) + r" \\ \hline")
    tex += [" & ".join(totals) + r" \\ \hline", r"\end{tabular}\end{table}"]
    (ROOT / "论文/问题3_模型与结果.md").write_text("\n".join(md), encoding="utf-8")
    (ROOT / "论文/问题3_结果表.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    print(daily.loc[list(DATES), ["plan_kwh", "adjusted_kwh", "ordinary_settlement_cost_yuan", "emergency_kwh", "total_cost_yuan"]].to_string())


if __name__ == "__main__":
    main()
