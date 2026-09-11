"""Generate Q4 scientific figures, requested-date tables, and measured conclusions."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

OUT=Path(__file__).resolve().parents[1]/'结果'
DATES=['2025-03-20','2025-06-21','2025-09-23','2025-12-21']
LABELS={'interval_robust':'区间鲁棒（主方案）','scenario_expected':'场景期望','half_radius':'半径×0.5','wider_radius':'半径×1.5','fixed_price_planning':'固定电价制定计划','perfect_price_benchmark':'提前知道真实价格（不可部署）'}

def table(f,digits=2):
    rows=['| '+' | '.join(f.columns)+' |','|'+'|'.join(['---']*len(f.columns))+'|']
    for r in f.itertuples(index=False,name=None):
        rows.append('| '+' | '.join(f'{v:,.{digits}f}' if isinstance(v,(float,np.floating)) else str(v) for v in r)+' |')
    return '\n'.join(rows)

def main():
    font=Path('/System/Library/Fonts/Supplemental/Arial Unicode.ttf')
    if font.exists():
        font_manager.fontManager.addfont(str(font)); plt.rcParams['font.family']=font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({'axes.unicode_minus':False,'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    summaries=json.loads((OUT/'problem4_summary.json').read_text())
    comp=pd.read_csv(OUT/'problem4_strategy_comparison.csv')
    forecasts=pd.read_csv(OUT/'price_forecasts_all_issues.csv.gz')
    metrics=pd.read_csv(OUT/'price_forecast_metrics.csv')
    models=pd.read_csv(OUT/'price_model_january_validation.csv')
    f2=pd.read_csv(OUT/'problem4-2_dispatch_all_2025.csv'); f3=pd.read_csv(OUT/'problem4-3_dispatch_all_2025.csv')
    frames={'4-2':f2,'4-3':f3}
    resultrows=[]
    for name,label in [('plan_cost_yuan','原计划费用（元）'),('adjustment_cost_yuan','调整费用（元）'),('emergency_cost_yuan','紧急购电费（元）'),('total_cost_yuan','总费用（元）'),('emergency_kwh','紧急购电量（kWh）'),('adjustment_up_kwh','上调购电量（kWh）'),('initial_soc_kwh','2月1日初始储电（kWh）'),('final_soc_kwh','年末实际储电（kWh）')]:
        resultrows.append({'指标':label,**{v:summaries[v][name] for v in frames}})
    lines=['# 第四问结果与论文素材','','本次新增求解：因果电价预测、历史误差场景、区间鲁棒MILP、全年执行回放和官方结果表。统计期为2025年2月1日至12月31日（334天）；1月只做选择/热启动。公式、信息假设和复现入口见上一级《建模笔记.md》。','','## 1. 主方案结果','',table(pd.DataFrame(resultrows)),'','4-2保持第二问日前计划和历史光伏预测；4-3保持第三问附件3预报与0、6、12、18时滚动更新。二者的预测信息、风险缓冲和热启动结果均不同，不能把费用差额全解释为MPC本身的作用。计划费按计划量结算，不按实际使用量减免。所有方案实际费用均逐日逐时使用附件4电价。','','## 2. 电价预测与区间检验','','仅在1月15—31日滚动验证中选择预测模型，2月起冻结为前7天/前14天同钟点平均；未用全年测试集选模型。无未来历史时使用已观测同钟点，首日无历史时使用预设0.6元/kWh先验。日内仅用已发生的近6小时误差做衰减修正。','',table(models[['model','MAE_yuan_per_kwh','RMSE_yuan_per_kwh']].rename(columns={'model':'候选模型','MAE_yuan_per_kwh':'1月验证MAE','RMSE_yuan_per_kwh':'1月验证RMSE'}),5),'','按发布时间和提前时距统计如下。单位为元/kWh；覆盖率是测试期逐点经验覆盖，不是24小时联合保证。','',table(metrics[['issue_hour','lead_band','MAE_yuan_per_kwh','RMSE_yuan_per_kwh','interval_coverage']].rename(columns={'issue_hour':'发布时间h','lead_band':'提前时距','MAE_yuan_per_kwh':'MAE','RMSE_yuan_per_kwh':'RMSE','interval_coverage':'经验覆盖率'}),5)]
    fig,axes=plt.subplots(2,2,figsize=(11,6.6),sharex=True,layout='constrained')
    for ax,date in zip(axes.ravel(),DATES):
        g=forecasts[forecasts.issue_datetime==date+' 00:00:00']; x=g.lead_slot/6
        ax.fill_between(x,g.lower,g.upper,color='#2563EB',alpha=.14,label='历史误差区间')
        ax.plot(x,g.actual_price_for_evaluation_only,color='#202020',lw=1.25,label='实际价格（事后）')
        ax.plot(x,g.scenario_mean_prediction,color='#2563EB',lw=1.15,label='日前预测均值')
        ax.set_xlabel(date+' 时刻（h）'); ax.set_ylabel('电价（元/kWh）'); ax.set_xticks([0,6,12,18,24]); ax.grid(axis='y',alpha=.15)
    axes[0,0].legend(frameon=False,fontsize=8)
    fig.savefig(OUT/'q4_price_forecasts_four_days.png',dpi=300); plt.close(fig)
    coverage=float(np.average(metrics.interval_coverage,weights=metrics.n))
    lines+=['',f'整体逐点区间覆盖率为 {coverage:.2%}。半径使用历史绝对偏差90%分位数，但测试覆盖率未达到90%，不能写成“90%可靠保证”。采用盒式集合还忽略了极端电价是否会同时发生，可能偏保守。','','![指定四天电价预测](q4_price_forecasts_four_days.png)','','图1 指定四天的日前电价与实际电价。黑线仅用于事后评价，优化时不可见。区间可能漏掉尖峰，风险半径不代表确定的统计置信区间。','','## 3. 对照实验与敏感性','','每种问题内部的对照使用相同2月1日初始SOC、负载/光伏预测与执行规则。主参数半径倍率1在测试前固定，未看测试结果后挑选。场景期望模型共享同一个决策向量，其线性目标等价于用场景均价求解；不是完整多阶段随机场景树。']
    for variant in frames:
        c=comp[comp.variant==variant].copy()
        c['strategy']=c.strategy.map(LABELS)
        lines+=['',f'### {variant}','',table(c[['strategy','total_cost_yuan','emergency_kwh','daily_cost_upper5pct_mean_yuan','final_soc_kwh']].rename(columns={'strategy':'策略','total_cost_yuan':'总费用（元）','emergency_kwh':'紧急购电（kWh）','daily_cost_upper5pct_mean_yuan':'最贵5%日期均费（元）','final_soc_kwh':'年末SOC（kWh）'}))]
        group=comp[comp.variant==variant].set_index('strategy')
        difference=group.loc['interval_robust','total_cost_yuan']-group.loc['scenario_expected','total_cost_yuan']
        lines+=['',f'主方案相对场景期望的实际总费用变化：{difference:+,.2f}元（{difference/group.loc["scenario_expected","total_cost_yuan"]:+.2%}）。此为回测结果，不预设鲁棒一定更省钱。']
    lines+=['','“固定电价制定计划”仍按附件4真实电价结算，隔离规划价格信号的影响；它不是原第二/三问固定电价总账。“提前知道真实价格”仅为信息价值参考，不能作为可部署方案，也不是全年全策略的严格下界。不同方案年末SOC不完全相等，费用比较应结合上表末态，不能假装末态一致。最贵5%日期均费是描述性统计，不是本次优化目标中的CVaR。']
    fig,axes=plt.subplots(1,2,figsize=(11,4.8),layout='constrained')
    for ax,variant in zip(axes,frames):
        c=comp[(comp.variant==variant)&comp.strategy.isin(['interval_robust','scenario_expected','half_radius','wider_radius'])]
        ax.bar(np.arange(len(c)),c.total_cost_yuan/1e4,color=['#2563EB','#8B98AA','#69A89A','#D69B51'])
        ax.set_xticks(np.arange(len(c)),[LABELS[x] for x in c.strategy],rotation=18,ha='right',fontsize=9)
        ax.set_ylabel(variant+' 总费用（万元）'); ax.grid(axis='y',alpha=.15)
    fig.savefig(OUT/'q4_price_risk_comparison.png',dpi=300); plt.close(fig)
    lines+=['','![价格风险敏感性](q4_price_risk_comparison.png)','','图2 同一预测与执行框架内的价格风险半径对照。柱图从零开始，展示总体费用；小额差异以精确表格为准。','','## 4. 指定四天的表1—表3','','表1时间段使用00:00起的0基slot60、72、84、96、108、120；电量单位kWh，费用单位元。调整后购电量不是调整增量。表2是实际执行值，不是优化器尚未执行的规划轨迹。']
    for variant,f in frames.items():
        e=pd.read_csv(OUT/f'problem{variant}_emergency_events.csv'); purchased=[]; blocks=[]
        lines+=['',f'### {variant} 指定日期']
        for date in DATES:
            day=f[f.date==date]
            selected=day[day.slot.isin([60,72,84,96,108,120])][['time_interval','plan_kwh','adjusted_purchase_kwh','emergency_kwh']].rename(columns={'time_interval':'时间段','plan_kwh':'原计划','adjusted_purchase_kwh':'最终生效购电','emergency_kwh':'紧急购电'})
            purchased.extend({'日期':date,**r} for r in selected.to_dict('records'))
            bs=[]
            for b in range(6):
                part=day.iloc[b*24:(b+1)*24]; row={'日期':date,'时间段':f'{b*4:02d}:00-{(b+1)*4:02d}:00','充电量':part.charge_kwh.sum(),'放电量':part.discharge_kwh.sum()}; bs.append(row); blocks.append(row)
            ev=e[e.date==date][['interval','emergency_kwh']].rename(columns={'interval':'连续紧急时间段','emergency_kwh':'紧急购电量'})
            lines+=['',f'#### {date}','','表1 购电量','',table(selected),'',f'全天计划量 {day.plan_kwh.sum():,.2f}；最终生效量 {day.adjusted_purchase_kwh.sum():,.2f}；紧急量 {day.emergency_kwh.sum():,.2f}。计划费 {day.plan_cost_yuan.sum():,.2f}；调整费 {day.adjustment_cost_yuan.sum():,.2f}；紧急费 {day.emergency_cost_yuan.sum():,.2f}；总费 {day.total_cost_yuan.sum():,.2f}。','','表2 储能电量','',table(pd.DataFrame(bs).drop(columns='日期')),'',f'0:00储电 {day.soc_start_kwh.iloc[0]:,.2f}；24:00储电 {day.soc_end_kwh.iloc[-1]:,.2f}。','','表3 紧急购电','','无。' if ev.empty else table(ev)]
        pd.DataFrame(purchased).to_csv(OUT/f'problem{variant}_table1_four_dates.csv',index=False,encoding='utf-8-sig')
        pd.DataFrame(blocks).to_csv(OUT/f'problem{variant}_table2_four_dates.csv',index=False,encoding='utf-8-sig')
        e[e.date.isin(DATES)].to_csv(OUT/f'problem{variant}_table3_four_dates.csv',index=False,encoding='utf-8-sig')
    fig,axes=plt.subplots(3,1,figsize=(11,8),sharex=True,layout='constrained')
    day=f3[f3.date=='2025-06-21']; x=(day.slot+.5)/6
    axes[0].plot(x,day.price_yuan_per_kwh,color='#202020',label='实际价格（事后）')
    axes[0].plot(x,day.price_optimization_yuan_per_kwh,color='#2563EB',label='当前优化使用的价格上界')
    axes[0].set_ylabel('元/kWh'); axes[0].legend(frameon=False,ncol=2,fontsize=9)
    axes[1].step(x,day.plan_kwh*6,where='mid',color='#D97706',label='原计划')
    axes[1].step(x,day.adjusted_purchase_kwh*6,where='mid',color='#2563EB',label='最终生效购电')
    axes[1].fill_between(x,0,day.emergency_kwh*6,step='mid',color='#DC2626',alpha=.6,label='紧急购电')
    axes[1].set_ylabel('等效功率（kW）'); axes[1].legend(frameon=False,ncol=3,fontsize=9)
    axes[2].plot(np.arange(145)/6,np.r_[day.soc_start_kwh.iloc[0],day.soc_end_kwh],color='#0D9488')
    axes[2].axhline(1200,ls='--',lw=.8,color='#888888'); axes[2].axhline(10800,ls='--',lw=.8,color='#888888')
    axes[2].set_ylabel('实际储电量（kWh）'); axes[2].set_xlabel('2025-06-21 时刻（h）'); axes[2].set_ylim(500,11500)
    for ax in axes:
        for h in (6,12,18): ax.axvline(h,ls=':',lw=.7,color='#AAAAAA')
        ax.set_xlim(0,24); ax.set_xticks([0,6,12,18,24]); ax.grid(axis='y',alpha=.15)
    fig.savefig(OUT/'q4_rolling_dispatch_june21.png',dpi=300); plt.close(fig)
    lines+=['','![4-3滚动执行](q4_rolling_dispatch_june21.png)','','图3 6月21日价格信号、购电计划和真实储能状态。竖线是日内更新点；计划变量kWh乘6仅为图中等效kW显示，不改变结算量。','','## 5. 已完成校验与使用边界','','独立校验见problem4_verification.json：改变未来电价不改变已发布预测；1月模型选择不受2—12月价格扰动；逐时费用使用对应日期实际价格；表格逐格一致；SOC连续、效率、功率上限、互斥和≥平衡满足。Excel公式经重算和两天的真实输入扰动测试，导出副本修正官方模板晚10分钟的表头，原附件未变。','','本方案为“价格风险缓冲＋滚动确定性MILP＋因果执行”的可复现实验，不是全年随机控制全局最优解。未来电价不可知是参考截图采用的建模假设，而不是题面明示的发布机制。价格场景未同时优化光伏/负载联合分布；实时储能采用余电先充、缺口先放规则，规划与实际轨迹可不同。4-2每日规划终态6000、4-3窗口规划终态至少6000是继承的工程约束，不重置实际SOC。','','论文可用的结论：在上述信息与结算假设下，滚动方案减少了本实验的紧急缺口，但盒式电价鲁棒不保证优于期望价格策略。保留实际差异、区间欠覆盖、末态差异等限制，比只宣称“更鲁棒所以更便宜”更符合数据。','']
    (OUT/'问题4结果报告.md').write_text('\n'.join(lines),encoding='utf-8')
    print('Generated Q4 report, six CSV tables, three figures.')

if __name__=='__main__': main()
