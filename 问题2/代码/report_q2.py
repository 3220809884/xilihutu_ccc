"""Generate the problem-2 paper chapter and exact-layout LaTeX tables from saved results."""
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "问题2/结果"
DATES = ["2025-03-20", "2025-06-21", "2025-09-23", "2025-12-21"]


def main():
    summary = json.loads((OUT/"problem2_summary.json").read_text(encoding="utf-8"))
    df = pd.read_csv(OUT/"problem2_dispatch_all_2025.csv")
    daily = pd.read_csv(OUT/"problem2_daily_summary.csv").set_index("date")
    events = pd.read_csv(OUT/"problem2_emergency_events.csv")
    fmt = lambda x: f"{float(x):.4f}"
    md = ["# 问题2：基于历史净负荷分位数的逐日计划购电与实时执行", "",
          "## 1. 问题分析与信息边界", "",
          "每日0:00提交当天144个10分钟时段的计划购电功率。计划提交后按计划量全额结算，实际缺口按同时段电价的5倍补购。普通计划在日内保持固定，储能根据当前供需差执行。负载、光伏和各功率变量采用kW，储电量采用kWh，时间步长Δt=1/6小时。", "",
          "仅使用附件1电价与附件2历史负载、光伏。问题代码从统一主表读取这些字段，不读取附件3预测、附件4电价或附件1负载。设N=L−PV，直接预测净负荷，保留历史负载和光伏的同期组合关系。目标日d的预测只使用日期小于d的完整日记录。", "",
          "## 2. 历史预测与参数选择", "",
          "对时段t，取最近W个已完成日期的净负荷样本，其α经验分位数为风险调整预测：", "",
          "    N_hat[d,t] = Quantile_alpha({N[j,t] : max(0,d-W) <= j < d})", "",
          "分位数采用线性插值。α控制提前购电与五倍价缺电风险之间的权衡；储能使其不同于独立时段的分位数采购问题，因此通过历史闭环费用选择参数。候选W∈{7,14,28}、α∈{0.50,0.65,0.80,0.90,0.95}。验证期为2025年1月15日至31日，每个候选均按时间顺序预测、优化、执行，并从同一验证初始SOC出发。评分为计划费加紧急费，再加(6000−验证末SOC)×最低日内电价/0.9的库存折算值；该折算仅用于候选比较，不计入正式购电费。", "",
          f"1月选出W={summary['selected_window_days']}、α={summary['selected_quantile']:.2f}，2月1日0:00冻结参数，2—12月仅更新滚动历史样本。选参完全不使用2—12月的费用或实际值。", "",
          "冷启动：1月1日无任何历史数据，采用零计划、储能静置，负载净缺口紧急购电；1月2—31日使用预先固定的W=14、α=0.80运行，初始SOC为题定6000 kWh。选参的虚拟验证轨迹不替换1月实际运行轨迹，也不倒放重算1月。1月成本不包含在题定2—12月交付统计中。", "",
          f"由1月真实递推得到2月1日初SOC={fmt(summary['feb1_initial_soc_kwh'])} kWh。1月的全部轨迹也保存，以便追溯该初始条件。", "",
          "## 3. 日前MILP", "",
          "决策变量为q_t、c_t、d_t≥0（kW）、S_t（kWh）和z_t∈{0,1}。以真实日初SOC为初值，求解：", "",
          "    min Σ_t π_t q_t Δt", "",
          "    q_t + d_t − c_t >= N_hat[d,t]",
          "    S[t+1] = S[t] + 0.9 c_t Δt − d_t Δt/0.9",
          "    1200 <= S[t] <= 10800",
          "    0 <= c_t <= 5000 z_t",
          "    0 <= d_t <= 5000 (1−z_t)",
          "    S[0] = 当天实际初始储电量，S[144] = 6000", "",
          "日末6000 kWh是计划模型的工程储备假设，用于抑制有限日时域末端过度放电；问题二没有要求日初日末相等。该目标仅约束预测计划，实际执行不强制恢复到6000 kWh。每个计划MILP在自身预测及该目标条件下求得最优解，但完整含预测误差的反馈策略不声称是随机动态控制全局最优。", "",
          f"使用SciPy {summary['scipy_version']}的HiGHS MILP求解，相对间隙阈值10⁻⁹，单次时限30秒，非成功状态直接终止。本次正式逐日计划最大报告间隙为{summary['max_daily_milp_gap']:.3g}。", "",
          "## 4. 实时执行、紧急购电与跨日状态", "",
          "设当前时段已知的富余功率b_t=q_t+PV_t−L_t。假定当前10分钟时段的负载/光伏可被实时测得并以该时段代表功率执行，无未来时段信息。", "",
          "当b_t≥0时，令c_t=min(b_t,5000,(10800−S_t)/(0.9Δt))、d_t=e_t=0，剩余记为未利用富余功率。", "",
          "当b_t<0时，令d_t=min(−b_t,5000,(S_t−1200)×0.9/Δt)、c_t=0、e_t=−b_t−d_t。紧急电只用于剩余负载缺口。", "",
          "    S[t+1] = S[t] + 0.9 c_t Δt − d_t Δt/0.9",
          "    q_t + e_t + PV_t + d_t = L_t + c_t + w_t，w_t >= 0",
          "    C_plan[d] = Σ_t π_t q_t Δt",
          "    C_emergency[d] = Σ_t 5π_t e_t Δt",
          "    C_total[d] = C_plan[d] + C_emergency[d]", "",
          "下一天的初始SOC等于当日实际末SOC。上述反馈是避免当前紧急缺口的局部规则，可能牺牲后续高价时段的储能价值，是本方案的限制。日内没有调整普通购电计划。", "",
          "## 5. 2025年2—12月总体结果", "",
          "| 指标 | 数值 |", "|---|---:|",
          f"| 交付天数 / 时段数 | {summary['days']} / {summary['slots']} |",
          *[f"| {label} | {fmt(summary[key])} |" for label,key in [
              ("计划购电量（kWh）","plan_kwh"),("紧急购电量（kWh）","emergency_kwh"),
              ("计划购电费（元）","plan_cost_yuan"),("紧急购电费（元）","emergency_cost_yuan"),
              ("总购电费（元）","total_cost_yuan"),("实际充电量（kWh）","charge_kwh"),
              ("实际放电量（kWh）","discharge_kwh"),("12月31日末SOC（kWh）","dec31_terminal_soc_kwh")]],
          "",
          f"采用相同净负荷风险预测、无储能的计划q=max(N_hat,0)作为对照，费用为{fmt(summary['baseline_without_storage_cost_yuan'])}元。本方案低{fmt(summary['cost_saving_vs_same_forecast_without_storage_yuan'])}元，即{100*summary['cost_saving_vs_same_forecast_without_storage_yuan']/summary['baseline_without_storage_cost_yuan']:.4f}%。该对照共享预测和选定参数，未单独为无储能策略重新调参，不代表所有无储能策略的最优费用。", "",
          f"附带报告历史中位数净负荷预测的MAE={fmt(summary['median_net_forecast_mae_kw'])} kW、RMSE={fmt(summary['median_net_forecast_rmse_kw'])} kW。风险分位数预测的实测覆盖率为{100*summary['risk_forecast_empirical_coverage']:.4f}%，不是供电可靠率；经紧急补购后全部时段满足负载。", "",
          "## 6. 指定日期结果：按题面表1、表2布局", "",
          "表1中的时段购电量与全天购电量均指0:00计划量，全天购电费为计划费；紧急量在表3单列，各日含紧急购电的总费用另行给出。表2全部采用实际执行的接口充放电电量。单位：电量kWh、费用元。", "",
          "所有物理时段按统一主表解释：slot=0为00:00—00:10，slot=143为23:50—24:00。原模板的计划表时段标签整体偏移10分钟，按项目约定保留文字、依slot位置写入B:EO。论文的10:00—10:10对应slot=60，即模板BJ列；12:00、14:00、16:00、18:00、20:00对应BV、CH、CT、DF、DR列。不能按原模板偏移文字重新取值。", ""]
    tex = ["% UTF-8. Include in an existing ctex/XeLaTeX document; this is a table fragment.",
           "% All amounts in kWh, all costs in yuan. Plan costs exclude emergency costs."]
    for date in DATES:
        g = df[df.date==date].reset_index(drop=True)
        row = daily.loc[date]
        md += [f"### {date}", "", "表1：计划购电量及计划费用", "",
               "| 时间段 | 购电量 | 时间段 | 购电量 | 时间段 | 购电量 |",
               "|---|---:|---|---:|---|---:|"]
        tex += [r"\begin{table}[htbp]\centering",f"\\caption{{{date}计划购电量及计划费用（题面表1格式）}}",
                r"\begin{tabular}{|c|r|c|r|c|r|}\hline",r"时间段 & 购电量 & 时间段 & 购电量 & 时间段 & 购电量 \\\hline"]
        for slots in [[60,72,84],[96,108,120]]:
            cells=[]
            for t in slots:
                cells += [f"{t//6:02d}:00-{t//6:02d}:10", fmt(g.plan_kwh.iloc[t])]
            md.append("| "+" | ".join(cells)+" |")
            tex.append(" & ".join(cells)+r" \\\hline")
        md += [f"| 全天购电量 | {fmt(row.plan_kwh)} | 全天购电费 | {fmt(row.plan_cost_yuan)} | | |", "",
               "表2：实际储能充放电量与首末储电量", "",
               "| 时间段 | 充电量 | 放电量 | 时间段 | 充电量 | 放电量 |",
               "|---|---:|---:|---|---:|---:|"]
        tex += [f"\\multicolumn{{2}}{{|c|}}{{全天购电量}} & {fmt(row.plan_kwh)} & \\multicolumn{{2}}{{c|}}{{全天购电费}} & {fmt(row.plan_cost_yuan)} \\\\\\hline",
                r"\end{tabular}\end{table}",r"\begin{table}[htbp]\centering",f"\\caption{{{date}实际储能调度（题面表2格式）}}",
                r"\begin{tabular}{|c|r|r|c|r|r|}\hline",r"时间段 & 充电量 & 放电量 & 时间段 & 充电量 & 放电量 \\\hline"]
        charges=g.charge_kwh.to_numpy().reshape(6,24).sum(axis=1)
        discharges=g.discharge_kwh.to_numpy().reshape(6,24).sum(axis=1)
        for b in [0,2,4]:
            cells=[]
            for j in [b,b+1]:
                cells += [f"{4*j}:00-{4*j+4}:00",fmt(charges[j]),fmt(discharges[j])]
            md.append("| "+" | ".join(cells)+" |")
            tex.append(" & ".join(cells)+r" \\\hline")
        md += [f"| 0:00储电量 | {fmt(row.soc_start_kwh)} | | 24:00储电量 | {fmt(row.soc_end_kwh)} | |", "",
               f"该日紧急购电{fmt(row.emergency_kwh)} kWh，紧急费用{fmt(row.emergency_cost_yuan)}元，计划及紧急总购电量{fmt(row.plan_kwh+row.emergency_kwh)} kWh，总购电费{fmt(row.total_cost_yuan)}元。", ""]
        tex += [f"\\multicolumn{{2}}{{|c|}}{{0:00储电量}} & {fmt(row.soc_start_kwh)} & \\multicolumn{{2}}{{c|}}{{24:00储电量}} & {fmt(row.soc_end_kwh)} \\\\\\hline",
                r"\end{tabular}\end{table}"]
    md += ["## 7. 表3：四个指定日期的紧急购电", "",
           "同日连续发生紧急购电的10分钟时段合并为一个区间，数量为各时段电量之和。合并区间费用仍逐时采用5π_t计价，不能取一个区间单价代替。", "",
           "| 2025.3.20 | | 2025.6.21 | | 2025.9.23 | | 2025.12.21 | |",
           "|---|---:|---|---:|---|---:|---|---:|",
           "| 时间段 | 购电量 | 时间段 | 购电量 | 时间段 | 购电量 | 时间段 | 购电量 |"]
    tex += [r"\begin{table}[htbp]\centering",r"\caption{指定日期紧急购电量（题面表3格式）}",r"\small\setlength{\tabcolsep}{3pt}",
            r"\begin{tabular}{|c|r|c|r|c|r|c|r|}\hline",
            r"\multicolumn{2}{|c|}{2025.3.20} & \multicolumn{2}{c|}{2025.6.21} & \multicolumn{2}{c|}{2025.9.23} & \multicolumn{2}{c|}{2025.12.21} \\\hline",
            r"时间段 & 购电量 & 时间段 & 购电量 & 时间段 & 购电量 & 时间段 & 购电量 \\\hline"]
    groups=[events[events.date==date].to_dict(orient="records") for date in DATES]
    for group in groups:
        if not group:
            group.append(dict(interval="无",emergency_kwh=0))
    for i in range(max(map(len,groups))):
        cells=[]
        for group in groups:
            cells += [group[i]["interval"],fmt(group[i]["emergency_kwh"])] if i<len(group) else ["",""]
        md.append("| "+" | ".join(cells)+" |")
        tex.append(" & ".join(cells)+r" \\\hline")
    totals=[]
    for date in DATES:
        totals += ["合计",fmt(daily.loc[date].emergency_kwh)]
    md += ["| "+" | ".join(totals)+" |", "", "## 8. 校验、局限与复现", "",
           f"交付数据包含334×144=48096个计划值，实际储能执行轨迹覆盖完整365天。供电平衡最大残差为{summary['checks']['max_balance_residual_kw']:.3g} kW，SOC转移最大残差为{summary['checks']['max_soc_transition_residual_kwh']:.3g} kWh，相邻时段及跨日SOC残差为0，无同时充放电时段。细微浮点残差采用10⁻⁶容差判断。", "",
           "统计结果为上述明确策略的历史回测，不能解释为已知全年真实负载、光伏时的事后最优值。经验分位数不保证季节变化后的精确覆盖率，当前实时贪心反馈未显式优化未来紧急风险，日末计划目标和1月冷启动也是建模假设。未来可在严格信息边界下引入天气、周内类型、场景随机规划及储能末端价值敏感性分析。", "",
           "代码：问题2/代码/solve_q2.py（求解）；export_q2.mjs（填充原模板）；report_q2.py（本章及LaTeX表格）；verify_q2.py（独立检查）。结果：问题2/结果/result2.xlsx；problem2_dispatch_all_2025.csv（全时段）；problem2_daily_summary.csv（日汇总）；problem2_emergency_events.csv（合并紧急事件）；problem2_january_validation.csv（全部15组候选）；problem2_information_audit.csv（逐日信息集）。数值未手工改写，显示保留四位小数，精确复算以CSV/JSON全精度值为准。", ""]
    tex += [" & ".join(totals)+r" \\\hline",r"\end{tabular}\end{table}"]
    (ROOT/"论文/问题2_模型与结果.md").write_text("\n".join(md),encoding="utf-8")
    (ROOT/"论文/问题2_结果表.tex").write_text("\n".join(tex)+"\n",encoding="utf-8")
    print(daily.loc[DATES].to_string())


if __name__ == "__main__":
    main()
