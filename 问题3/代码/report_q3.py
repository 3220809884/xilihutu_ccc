"""Generate scientific figures and all four requested dates' tables from solved data."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

OUT=Path(__file__).resolve().parents[1]/"结果"
DATES=["2025-03-20","2025-06-21","2025-09-23","2025-12-21"]


def table(frame, digits=2):
    cols=frame.columns.tolist()
    lines=["| "+" | ".join(cols)+" |","|"+"|".join(["---"]*len(cols))+"|"]
    for row in frame.itertuples(index=False,name=None):
        cells=[f"{x:,.{digits}f}" if isinstance(x,(float,np.floating)) else str(x) for x in row]
        lines.append("| "+" | ".join(cells)+" |")
    return "\n".join(lines)


def main():
    f=pd.read_csv(OUT/"problem3_dispatch_all_2025.csv")
    g=f[f.date>="2025-02-01"]
    daily=pd.read_csv(OUT/"problem3_daily_summary.csv")
    s=json.loads((OUT/"problem3_summary.json").read_text())
    comp=pd.read_csv(OUT/"problem3_strategy_comparison.csv")
    common=pd.read_csv(OUT/"problem3_forecast_common_targets.csv")
    events=pd.read_csv(OUT/"problem3_emergency_events.csv")
    windows=pd.read_csv(OUT/"problem3_rolling_windows.csv.gz")
    font=Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf")
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams["font.family"]=font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({"axes.unicode_minus":False,"font.size":10,"axes.spines.top":False,"axes.spines.right":False,"figure.dpi":150})

    # Forecast comparison at the four dates explicitly requested in the question.
    fig,axes=plt.subplots(2,2,figsize=(11,6.5),sharex=True,layout="constrained")
    for ax,date in zip(axes.ravel(),DATES):
        day=f[f.date==date]; x=(day.slot.to_numpy()+1)/6
        old=windows[windows.issue_datetime==date+" 00:00:00"]
        ax.plot(x,day.pv_actual_kw,color="#242424",lw=1.5,label="光伏实测")
        ax.plot(x,old.pv_forecast_kw,color="#D97706",ls="--",lw=1.2,label="0:00预报")
        ax.plot(x,day.pv_forecast_kw,color="#2563EB",lw=1.2,label="当前可用预报")
        for h in [6,12,18]: ax.axvline(h,color="#BBBBBB",lw=.6,ls=":")
        ax.set_title(date); ax.set_xlabel("时刻（h）"); ax.set_ylabel("光伏功率（kW）"); ax.set_xticks([0,6,12,18,24]); ax.grid(axis="y",alpha=.15)
    axes[0,0].legend(frameon=False,fontsize=9)
    fig.savefig(OUT/"q3_forecast_four_days.png",dpi=300); plt.close(fig)

    day=f[f.date=="2025-06-21"]; x=(day.slot.to_numpy()+.5)/6
    fig,axes=plt.subplots(3,1,figsize=(11,8),sharex=True,layout="constrained")
    axes[0].step(x,day.plan_kwh*6,where="mid",color="#D97706",label="原计划购电功率")
    axes[0].step(x,day.adjusted_purchase_kwh*6,where="mid",color="#2563EB",label="调整后购电功率")
    axes[0].fill_between(x,0,day.emergency_kwh*6,step="mid",color="#DC2626",alpha=.6,label="紧急购电功率")
    axes[0].set_ylabel("功率（kW）"); axes[0].legend(frameon=False,ncol=3,fontsize=9)
    axes[1].fill_between(x,0,day.charge_kwh*6,step="mid",color="#0D9488",alpha=.7,label="充电")
    axes[1].fill_between(x,0,-day.discharge_kwh*6,step="mid",color="#8B5CF6",alpha=.7,label="放电（负向显示）")
    axes[1].set_ylabel("储能功率（kW）"); axes[1].legend(frameon=False,ncol=2,fontsize=9)
    axes[2].plot(np.arange(145)/6,np.r_[day.soc_start_kwh.iloc[0],day.soc_end_kwh],color="#242424",label="实际储电量")
    axes[2].axhline(1200,color="#DC2626",ls="--",lw=.8); axes[2].axhline(10800,color="#DC2626",ls="--",lw=.8)
    axes[2].set_ylabel("储电量（kWh）"); axes[2].set_xlabel("2025-06-21 时刻（h）"); axes[2].set_ylim(500,11500)
    for ax in axes:
        for h in [6,12,18]: ax.axvline(h,color="#BBBBBB",lw=.7,ls=":")
        ax.set_xlim(0,24); ax.set_xticks([0,4,6,8,12,16,18,20,24]); ax.grid(axis="y",alpha=.15)
    fig.savefig(OUT/"q3_dispatch_june21.png",dpi=300); plt.close(fig)

    strategies=comp[(comp.alpha==.8)&(comp.terminal_target_kwh==6000)].copy()
    strategies=strategies.sort_values("total_cost_yuan",ascending=False)
    fig,axes=plt.subplots(1,2,figsize=(11,5),layout="constrained")
    axes[0].barh(strategies.strategy,strategies.total_cost_yuan/1e4,color="#2563EB")
    axes[0].set_xlabel("总购电费（万元）"); axes[0].set_ylabel("使用的预报时刻（h）")
    axes[1].barh(strategies.strategy,strategies.emergency_kwh/1e3,color="#D97706")
    axes[1].set_xlabel("紧急购电量（千kWh）")
    for ax in axes: ax.grid(axis="x",alpha=.15)
    fig.savefig(OUT/"q3_update_strategy_comparison.png",dpi=300); plt.close(fig)

    purchased,storage=[] ,[]
    lines=["# 问题3结果报告", "", "本次新增求解：附件3因果预报驱动的24小时滚动MILP及实际运行回放。正式统计区间为2025年2月1日至12月31日，共334天。模型公式、假设与复现步骤见上一级《建模笔记.md》。", "", "## 1. 主方案结果", "", table(pd.DataFrame({"指标":["计划购电费（元）","调整相关费用（元）","紧急购电费（元）","总购电费（元）","紧急购电量（kWh）","上调购电量（kWh）","2月1日初始储电量（kWh）","12月31日结束储电量（kWh）"],"数值":[s[k] for k in ["plan_cost_yuan","adjustment_cost_yuan","emergency_cost_yuan","total_cost_yuan","emergency_kwh","adjustment_up_kwh","initial_soc_kwh","final_soc_kwh"]]})), "", "费用按原计划全额结算，再加上调1.5倍电价、下调0.5倍违约费和实际紧急购电5倍电价；调整量以最终生效值与0:00原计划之差计算，每个执行时段只结算一次。", "", "## 2. 是否需要日内预报", "", "下列8种固定时刻组合使用相同负载预测、相同风险参数、相同2月1日实际初始储电量。结论仅适用于该模型、执行规则和数据，不是对所有策略的最优性证明。", "", table(strategies[["strategy","total_cost_yuan","emergency_kwh","adjustment_cost_yuan","final_soc_kwh"]].rename(columns={"strategy":"预报时刻","total_cost_yuan":"总费用（元）","emergency_kwh":"紧急购电（kWh）","adjustment_cost_yuan":"调整费用（元）","final_soc_kwh":"年末储电量（kWh）"})), "", f"四次预报方案相对仅0:00方案，费用减少{s.get('saving_vs_midnight_yuan',0):,.2f}元（{100*s.get('saving_vs_midnight_fraction',0):.2f}%）。", "", "![预报时刻对照](q3_update_strategy_comparison.png)", "", "不能预先断言预报越新，所有时段误差就必然更小。以下在完全相同的目标时段比较0:00预报与新预报：", "", table(common.rename(columns={"update_hour":"更新时刻","forecast":"预报版本","n_common_targets":"相同目标点数","MAE_kw":"MAE（kW）","RMSE_kw":"RMSE（kW）"})), "", "![指定四天的光伏预报](q3_forecast_four_days.png)", "", "## 3. 指定日期的表1、表2、表3", "", "表1列出原计划、调整后及紧急购电量，防止把调整后购电量误写为调整增量。所有电量单位为kWh，费用单位为元。表2使用实际执行的储能轨迹。"]
    for date in DATES:
        day=g[g.date==date]; ds=daily[daily.date==date].iloc[0]
        selected=day[day.slot.isin([60,72,84,96,108,120])][["time_interval","plan_kwh","adjusted_purchase_kwh","emergency_kwh"]].copy()
        selected.columns=["时间段","原计划购电量","调整后购电量","紧急购电量"]
        for r in selected.to_dict("records"): purchased.append({"日期":date,**r})
        blocks=[]
        for b in range(6):
            part=day.iloc[24*b:24*(b+1)]
            row={"时间段":f"{b*4:02d}:00-{b*4+4:02d}:00","充电量":float(part.charge_kwh.sum()),"放电量":float(part.discharge_kwh.sum())}
            blocks.append(row); storage.append({"日期":date,**row})
        ev=events[events.date==date][["interval","emergency_kwh"]].rename(columns={"interval":"紧急购电时间段","emergency_kwh":"紧急购电量"})
        lines += ["", f"### {date}", "", "表1 指定时间段购电量", "",table(selected), "", f"全天原计划购电量 {ds.plan_kwh:,.2f}；调整后购电量 {ds.adjusted_purchase_kwh:,.2f}；紧急购电量 {ds.emergency_kwh:,.2f}。计划费 {ds.plan_cost_yuan:,.2f}；调整费 {ds.adjustment_cost_yuan:,.2f}；紧急费 {ds.emergency_cost_yuan:,.2f}；总费 {ds.total_cost_yuan:,.2f}。", "", "表2 储能充放电量", "", table(pd.DataFrame(blocks)), "",f"0:00储电量 {ds.initial_soc_kwh:,.2f}；24:00储电量 {ds.terminal_soc_kwh:,.2f}。", "", "表3 紧急购电结果", "", "当日无紧急购电。" if ev.empty else table(ev)]
    lines += ["", "![6月21日调度](q3_dispatch_june21.png)", "", "## 4. 参数敏感性与局限", "", "风险分位数0.8、终端参考储电量6000kWh是在对照实验前固定的主方案参数。下列为敏感性分析，未根据全年的测试结果回选参数。0表示不加误差缓冲，并非使用0分位数。", "", table(comp[comp.strategy=="0+6+12+18"][["alpha","terminal_target_kwh","total_cost_yuan","emergency_kwh","final_soc_kwh"]].rename(columns={"alpha":"风险分位数","terminal_target_kwh":"规划终端参考（kWh）","total_cost_yuan":"总费用（元）","emergency_kwh":"紧急购电（kWh）","final_soc_kwh":"实际年末储电（kWh）"})), "", "1. 滚动MILP求解的是当前信息下的确定性风险缓冲近似；并非带完整场景树的多阶段随机规划。求解器的最优间隙不能解释成全年随机运行成本的最优间隙。", "2. 实际储能执行采用当前余电优先充电、缺口优先放电、剩余缺口紧急购电的因果规则。规划储能轨迹与实际轨迹可不同，后续滚动用真实SOC纠正。该规则未对每10分钟的未来应急价值再优化，是进一步改进方向。", "3. 日内滚动窗口跨到次日时，次日负载暂用当前日已发布的24小时负载曲线同钟点延拓；次日购电量只是内部展望，实际到次日0:00重新制定和计费。", "4. 风险缓冲用过去28天同发布时间、同提前时距的净负荷误差，0.8不等于全年80%的联合供电保证；没有假设误差正态。", "5. 终端储电量参考值是建模参数，题面仅第一问要求首末储电量相等。第三问不重置日初SOC。1月用于连续运行热启动，已将2月1日初始状态列出。", "6. 结算采用题面字面口径：计划费不退、下调另付违约费。这使下调被保留原量并弃余电支配，因此本方案下调量为零。若比赛方另有退款解释，必须统一更改目标函数及费用复核后重算。", "", "## 5. 输出口径和复核", "", "result3.xlsx保留官方四张工作表。计划表保存0:00原计划；调整表保存每个时段最终生效的购电量（包括未调整时段的原值），最后一列明确为调整相关费用；充放电表为实际执行量；紧急表合并同一天内连续的紧急时段。", "", "原模板的144个时段表头从0:10-0:20开始，末尾为次日0:00-0:10，与题目整日及主表不一致。仅输出副本将表头改为00:00-00:10至23:50-24:00，数值按日期+slot写入，原附件不改。", "", "数值平衡、效率、SOC边界与跨日连续、信息时点、计划量结算、调整量结算、Excel逐格映射均由verify_q3.py独立检查，检查结果见problem3_verification.json。", "", "完整预报误差按发布时间和提前时距分组保存在problem3_forecast_metrics.csv；所有滚动窗口（含次日未执行展望）保存在problem3_rolling_windows.csv.gz。", ""]
    (OUT/"问题3结果报告.md").write_text("\n".join(lines),encoding="utf-8")
    pd.DataFrame(purchased).to_csv(OUT/"table1_four_dates.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(storage).to_csv(OUT/"table2_four_dates.csv",index=False,encoding="utf-8-sig")
    events[events.date.isin(DATES)].to_csv(OUT/"table3_four_dates.csv",index=False,encoding="utf-8-sig")
    print("Generated result report, three CSV tables, and three 300-DPI figures.")


if __name__=="__main__": main()
