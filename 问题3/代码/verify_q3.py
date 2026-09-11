"""Independent reconciliation of raw master, rolling plans, accounts and result3.xlsx."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from openpyxl import load_workbook
from solve_q3_mpc import read_inputs, risk_buffers, optimize, execute

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "问题3/结果"
TOL = 3e-5


def close(a,b,tol=TOL):
    assert np.allclose(np.asarray(a,dtype=float),np.asarray(b,dtype=float),atol=tol,rtol=0), f"Max error: {np.max(np.abs(np.asarray(a)-np.asarray(b)))}"


def verify_logic():
    # Units and physical conversion checked with known examples.
    c,d,e,spill,s=execute(100.,600.,0.,6000.)
    close([c,d,e,spill,s],[0,0,0,0,6000])
    c,d,e,spill,s=execute(100.,0.,0.,6000.)
    close([c,d,e,spill,s],[100,0,0,0,6090])
    c,d,e,spill,s=execute(0.,600.,0.,6000.)
    close([c,d,e,spill,s],[0,100,0,0,6000-100/.9])
    # Full-plan payment plus down/up fee. Scalar independent formula.
    for plan,adj,expected in [(100,80,110),(100,120,130),(100,100,100)]:
        close(plan + 1.5*max(adj-plan,0) + .5*max(plan-adj,0),expected)
    # At minimum SOC, unchanged terminal, 600 kW * 1/6 h needs exactly 100 kWh.
    p=optimize(np.array([600.]),np.zeros(1),np.ones(1),1200.,terminal=1200.)
    close(p["q"],[100]); close(p["emergency_proxy"],[0])
    # Downward revision is dominated with >= balance and full sunk plan payment.
    p=optimize(np.zeros(1),np.zeros(1),np.ones(1),1200.,baseline=np.array([100.]),terminal=1200.)
    close(p["q"],[100])
    data=read_inputs()
    before,_=risk_buffers(data,.8)
    # Future residuals cannot alter the current issue's reserve calibration.
    data.residual[100:]=1e8
    after,_=risk_buffers(data,.8)
    close(before[:101],after[:101])


def main():
    verify_logic()
    f=pd.read_csv(OUT/"problem3_dispatch_all_2025.csv")
    g=f[f.date>="2025-02-01"].copy()
    w=pd.read_csv(OUT/"problem3_rolling_windows.csv.gz")
    audit=pd.read_csv(OUT/"problem3_information_audit.csv",keep_default_na=False)
    risk=pd.read_csv(OUT/"problem3_risk_calibration_audit.csv",keep_default_na=False)
    summary=json.loads((OUT/"problem3_summary.json").read_text())
    daily=pd.read_csv(OUT/"problem3_daily_summary.csv")
    source=pd.read_csv(ROOT/"公共代码/数据/master_10min.csv")
    assert len(f)==365*144 and len(g)==334*144 and len(audit)==365*4
    assert np.array_equal(f.slot,np.tile(np.arange(144),365))
    close(f.load_actual_kw,source.load_actual_kw)
    close(f.pv_actual_kw,source.pv_actual_kw)
    close(f.price_yuan_per_kwh,source.price_fixed_yuan_per_kwh)
    assert f.time_interval.iloc[0]=="00:00-00:10" and f.time_interval.iloc[-1]=="23:50-24:00"
    target=pd.to_datetime(f.date)+pd.to_timedelta((f.slot+1)*10,unit="min")
    assert (pd.to_datetime(f.target_end)==target).all()
    assert (pd.to_datetime(f.issue_datetime)<target).all()
    assert (f.issue_hour==(f.slot//36)*6).all()
    close(f.soc_start_kwh.iloc[0],6000)
    close(f.soc_start_kwh.iloc[1:],f.soc_end_kwh.iloc[:-1])
    close(f.soc_end_kwh-f.soc_start_kwh,.9*f.charge_kwh-f.discharge_kwh/.9)
    close(f.adjusted_purchase_kwh+f.emergency_kwh+f.pv_actual_kw/6+f.discharge_kwh-f.load_actual_kw/6-f.charge_kwh,f.spill_kwh)
    assert (f.spill_kwh>=-TOL).all()
    assert min(f.soc_start_kwh.min(),f.soc_end_kwh.min())>=1200-TOL
    assert max(f.soc_start_kwh.max(),f.soc_end_kwh.max())<=10800+TOL
    assert max(f.charge_kwh.max(),f.discharge_kwh.max())<=5000/6+TOL
    assert not ((f.charge_kwh>TOL)&(f.discharge_kwh>TOL)).any()
    close(f.plan_cost_yuan,f.plan_kwh*f.price_yuan_per_kwh)
    close(f.adjustment_up_kwh,np.maximum(f.adjusted_purchase_kwh-f.plan_kwh,0))
    close(f.adjustment_down_kwh,np.maximum(f.plan_kwh-f.adjusted_purchase_kwh,0))
    close(f.adjustment_cost_yuan,f.price_yuan_per_kwh*(1.5*f.adjustment_up_kwh+.5*f.adjustment_down_kwh))
    close(f.emergency_cost_yuan,5*f.price_yuan_per_kwh*f.emergency_kwh)
    close(f.total_cost_yuan,f.plan_cost_yuan+f.adjustment_cost_yuan+f.emergency_cost_yuan)
    for col in daily.columns:
        if col in g and col!="date":
            close(daily[col],g.groupby("date")[col].sum())
            close(g[col].sum(),summary[col],tol=.001)
    close(w.soc_end_kwh-w.soc_start_kwh,.9*w.charge_kwh-w.discharge_kwh/.9)
    slack=w.purchase_kwh+w.emergency_proxy_kwh+w.discharge_kwh-w.charge_kwh-(w.load_forecast_kw-w.pv_forecast_kw+w.buffer_kw)/6
    assert slack.min()>-TOL
    assert not ((w.charge_kwh>TOL)&(w.discharge_kwh>TOL)).any()
    assert w.soc_start_kwh.min()>=1200-TOL and w.soc_end_kwh.max()<=10800+TOL
    issue=pd.to_datetime(w.issue_datetime)
    assert (pd.to_datetime(w.target_end)==issue+pd.to_timedelta(w.lead_slot*10,unit="min")).all()
    committed=w[w.committed].copy()
    assert len(committed)==len(f) and not committed.duplicated("target_end").any()
    committed=committed.set_index("target_end").loc[f.target_end]
    close(committed.purchase_kwh,f.adjusted_purchase_kwh)
    midnight=w[pd.to_datetime(w.issue_datetime).dt.hour==0].set_index("target_end").loc[f.target_end]
    close(midnight.purchase_kwh,f.plan_kwh)
    for row in risk.itertuples():
        assert not row.latest_history_target or row.latest_history_target<=row.issue_datetime
        assert not row.history_last_issue_date or row.history_last_issue_date<row.issue_datetime[:10]
    for row in audit.itertuples():
        assert not row.load_history_through_date or row.load_history_through_date<row.date
        assert not row.load_model_train_end_date or row.load_model_train_end_date<row.date
    # Exact integer-lead interpolation and hour-zero anchor checks.
    hourly=pd.read_csv(ROOT/"公共代码/数据/forecast_attachment3_hourly.csv")
    hourly=hourly.set_index(["issue_datetime","target_datetime"])
    whole=w[w.lead_slot%6==0]
    keys=pd.MultiIndex.from_frame(whole[["issue_datetime","target_end"]])
    close(whole.pv_forecast_kw,hourly.loc[keys,"pv_forecast_kw"])
    assert not w[w.provisional_next_day].committed.any()
    # Official workbook: all cells, not only totals, must match the time series.
    book=load_workbook(OUT/"result3.xlsx",data_only=True,read_only=False)
    assert book.sheetnames==["计划购电量","调整购电量","充放电量","紧急购电量"]
    intervals=f.time_interval.iloc[:144].tolist()
    for name,col,cost_col in [("计划购电量","plan_kwh","plan_cost_yuan"),("调整购电量","adjusted_purchase_kwh","adjustment_cost_yuan")]:
        sheet=book[name]
        assert [sheet.cell(1,j).value for j in range(2,146)]==intervals
        for i,(date,day) in enumerate(g.groupby("date"),start=2):
            assert pd.Timestamp(sheet.cell(i,1).value).strftime("%Y-%m-%d")==date
            close([sheet.cell(i,j).value for j in range(2,146)],day[col])
            close([sheet.cell(i,146).value,sheet.cell(i,147).value],[day[col].sum(),day[cost_col].sum()])
    for i,(date,day) in enumerate(g.groupby("date")):
        r=2+i*6; sheet=book["充放电量"]
        close([sheet.cell(r+j,3).value for j in range(6)],day.charge_kwh.to_numpy().reshape(6,24).sum(axis=1))
        close([sheet.cell(r+j,4).value for j in range(6)],day.discharge_kwh.to_numpy().reshape(6,24).sum(axis=1))
        close([sheet.cell(r,6).value,sheet.cell(r+1,6).value],[day.soc_start_kwh.iloc[0],day.soc_end_kwh.iloc[-1]])
    ev=pd.read_csv(OUT/"problem3_emergency_events.csv")
    found=[]; current_date=None
    for date,span,energy in book["紧急购电量"].iter_rows(min_row=2,max_col=3,values_only=True):
        if date is not None: current_date=pd.Timestamp(date).strftime("%Y-%m-%d")
        if span and span!="无": found.append((current_date,span,energy))
    assert [(a,b) for a,b,_ in found]==list(zip(ev.date,ev.interval))
    close([c for _,_,c in found],ev.emergency_kwh)
    close(ev.emergency_kwh.sum(),g.emergency_kwh.sum())
    result={"status":"passed","days":334,"realized_slots":len(g),"all_year_rolling_windows":len(audit),"verified_rolling_rows":len(w),"xlsx_sheets":book.sheetnames,"future_residual_perturbation_test":"passed","units_and_fee_boundary_tests":"passed","header_alignment":"00:00-00:10 ... 23:50-24:00"}
    (OUT/"problem3_verification.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=="__main__": main()
