"""Q4: causal price forecasts, empirical expected cost and interval-robust MILPs.

Reuse Q2/Q3 physics and load/PV information without editing their source/results.
Main outputs: robust radius multiplier 1, actual attachment-4 settlement prices.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import scipy

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "问题2/代码"))
sys.path.insert(0, str(ROOT / "问题3/代码"))
import solve_q2_risk_milp as q2
import solve_q3_mpc as q3
from price_forecast import build_prices, issued_price, diagnostics

OUT = ROOT / "问题4/结果"
N, DT = 144, 1/6


def read_inputs():
    data = q3.read_inputs()
    master = pd.read_csv(ROOT / "公共代码/数据/master_10min.csv").sort_values(["date", "slot"])
    prices = master.price_volatile_yuan_per_kwh.to_numpy().reshape(365, N)
    paths = build_prices(prices, data.dates)
    hist = pd.read_csv(ROOT / "问题2/结果/problem2_dispatch_all_2025.csv").sort_values(["date", "slot"])
    assert np.array_equal(hist[["date", "slot"]].to_numpy(), master[["date", "slot"]].to_numpy())
    q2_net = hist.net_forecast_kw.to_numpy().reshape(365, N)
    q2_reserve = hist.reserve_requirement_kw.to_numpy().reshape(365, N)
    q2_pv = hist.pv_forecast_kw.to_numpy().reshape(365, N)
    q3_buffer, _ = q3.risk_buffers(data, .8)
    return data, paths, q2_net, q2_reserve, q2_pv, q3_buffer


def run(data, paths, net2, reserve2, pv2, buffer3, variant, scale=1., mode="forecast", start_day=0, end_day=365, soc0=6000., keep_windows=False):
    rows, windows, audit = [], [], []
    hours = (0,) if variant == "4-2" else (0, 6, 12, 18)
    for day in range(start_day, end_day):
        original = None
        for phase, hour in enumerate(hours):
            start = hour * 6
            stop = hours[phase + 1] * 6 if phase + 1 < len(hours) else N
            n = min(N, (365 - day) * N - start)
            price = issued_price(paths, day, phase, n, scale, mode, data.price[0])
            if variant == "4-2":
                # The legacy numerical solver scales q,c,d in kW; every exposed
                # decision/output here is converted to kWh exactly once.
                solved = q2.plan_day(net2[day], reserve2[day], price, soc0)
                plan = {"q": solved["purchase"] * DT, "charge": solved["charge"] * DT, "discharge": solved["discharge"] * DT, "soc": solved["soc"], "emergency_proxy": solved["reserve_shortfall"] * DT, "gap": solved["mip_gap"], "objective": solved["objective"]}
                net = net2[day]
                risk = reserve2[day]
                pv_hat = pv2[day]
            else:
                net = data.net_hat[day, phase, :n]
                risk = buffer3[day, phase, :n]
                pv_hat = data.pv_hat[day, phase, :n]
                plan = q3.optimize(net, risk, price, soc0, baseline=None if hour == 0 else original[start:], terminal=6000.)
            if hour == 0:
                original = plan["q"].copy()
            issue = pd.Timestamp(data.dates[day]) + pd.Timedelta(hours=hour)
            nominal = paths.mean[day, phase, :n]
            delta = paths.radius[day, phase, :n]
            baseline = original[start:] if hour else np.zeros(n)
            m = len(baseline) if hour else 0
            # Monetary exposure, including the old plan constant in the current
            # day's future slots. All components are nonnegative.
            exposure = plan["q"].copy()
            if m:
                exposure[:m] = baseline + 1.5*np.maximum(plan["q"][:m]-baseline,0)+.5*np.maximum(baseline-plan["q"][:m],0)
            exposure += 5 * plan["emergency_proxy"]
            audit.append({"variant": variant, "date": data.dates[day], "issue_datetime": str(issue), "issue_hour": hour, "first_target_end": str(issue + pd.Timedelta(minutes=10)), "last_target_end": str(issue + pd.Timedelta(minutes=n*10)), "soc_measured_kwh": soc0, "planned_terminal_kwh": float(plan["soc"][-1]), "milp_gap": float(plan["gap"]), "price_mode": mode, "radius_scale": scale, "nominal_horizon_proxy_cost_yuan": float(nominal@exposure), "box_worst_horizon_proxy_cost_yuan": float((nominal+scale*delta)@exposure), "q2_pv_history_only": variant=="4-2", "load_history_through_date": data.load_audit.iloc[day].history_through_date})
            if keep_windows:
                for j in range(n):
                    windows.append({"issue_datetime": str(issue), "target_end": str(issue+pd.Timedelta(minutes=(j+1)*10)), "lead_slot": j+1, "committed": j<stop-start, "provisional_next_day": j>=N-start, "net_forecast_kw": net[j], "risk_requirement_kw": risk[j], "pv_forecast_kw": pv_hat[j], "price_raw_yuan_per_kwh": paths.raw[day,phase,j], "price_mean_yuan_per_kwh": nominal[j], "price_radius_yuan_per_kwh": delta[j], "price_used_for_optimization": price[j], "purchase_kwh": plan["q"][j], "charge_kwh": plan["charge"][j], "discharge_kwh": plan["discharge"][j], "risk_proxy_kwh": plan["emergency_proxy"][j], "soc_start_kwh": plan["soc"][j], "soc_end_kwh": plan["soc"][j+1], "monetary_exposure_kwh": exposure[j]})
            for t in range(start,stop):
                j=t-start
                purchase=float(plan["q"][j])
                charge,discharge,emergency,spill,soc1=q3.execute(purchase,data.load[day,t],data.pv[day,t],soc0)
                actual_price=paths.actual[day,t]
                up=max(0.,purchase-original[t]); down=max(0.,original[t]-purchase)
                plan_cost=actual_price*original[t]
                adj_cost=actual_price*(1.5*up+.5*down)
                emg_cost=5*actual_price*emergency
                rows.append({"variant":variant,"date":data.dates[day],"slot":t,"time_interval":q3.interval(t),"issue_datetime":str(issue),"issue_hour":hour,"target_end":str(pd.Timestamp(data.dates[day])+pd.Timedelta(minutes=(t+1)*10)),"price_yuan_per_kwh":actual_price,"price_forecast_yuan_per_kwh":nominal[j],"price_radius_yuan_per_kwh":delta[j],"price_optimization_yuan_per_kwh":price[j],"load_actual_kw":data.load[day,t],"pv_actual_kw":data.pv[day,t],"load_forecast_kw":data.load_hat[day,t],"pv_forecast_kw":pv_hat[j],"risk_requirement_kw":risk[j],"plan_kwh":original[t],"adjusted_purchase_kwh":purchase,"adjustment_up_kwh":up,"adjustment_down_kwh":down,"charge_kwh":charge,"discharge_kwh":discharge,"emergency_kwh":emergency,"spill_kwh":spill,"soc_start_kwh":soc0,"soc_end_kwh":soc1,"plan_cost_yuan":plan_cost,"adjustment_cost_yuan":adj_cost,"emergency_cost_yuan":emg_cost,"total_cost_yuan":plan_cost+adj_cost+emg_cost})
                soc0=soc1
        if day%61==0:
            print(f"{variant} {mode} scale={scale:g} {data.dates[day]} SOC={soc0:.2f}",flush=True)
    return pd.DataFrame(rows),pd.DataFrame(windows),pd.DataFrame(audit)


def summarize(frame):
    summary=q3.summarize(frame)
    daily=frame.groupby("date").total_cost_yuan.sum()
    cutoff=float(daily.quantile(.95))
    summary.update({"initial_soc_kwh":float(frame.soc_start_kwh.iloc[0]),"final_soc_kwh":float(frame.soc_end_kwh.iloc[-1]),"daily_cost_p95_yuan":cutoff,"daily_cost_upper5pct_mean_yuan":float(daily[daily>=cutoff].mean()),"days_with_emergency":int(frame.groupby("date").emergency_kwh.sum().gt(1e-6).sum())})
    return summary


def export_result(out,variant,frame,windows,audit,data,paths):
    frame.to_csv(out/f"problem{variant}_dispatch_all_2025.csv",index=False,float_format="%.10f")
    windows.to_csv(out/f"problem{variant}_rolling_windows.csv.gz",index=False,float_format="%.10f",compression={"method":"gzip","mtime":0})
    audit.to_csv(out/f"problem{variant}_information_audit.csv",index=False)
    checks=q3.validate(frame)
    delivered=frame[frame.date>="2025-02-01"].copy()
    daily=delivered.groupby("date",as_index=False)[q3.SUM_COLS].sum()
    daily["initial_soc_kwh"]=delivered.groupby("date").soc_start_kwh.first().to_numpy()
    daily["terminal_soc_kwh"]=delivered.groupby("date").soc_end_kwh.last().to_numpy()
    daily.to_csv(out/f"problem{variant}_daily_summary.csv",index=False)
    ev=q3.events(delivered)
    ev.to_csv(out/f"problem{variant}_emergency_events.csv",index=False)
    payload=q3.excel_payload(delivered,ev)
    # Critical: 334 different price rows, not the first-day vector tiled all year.
    payload["prices_by_day"]=paths.actual[31:].tolist()
    payload.pop("prices")
    q3.save_json(out/f"problem{variant}_excel_data.json",payload)
    summary={"variant":variant,"period":"2025-02-01 to 2025-12-31","days":334,"price_model":paths.selected_model,"price_interval_quantile":.9,"radius_scale":1.,"price_information":"past observations only; future actual prices used for settlement after execution","load_pv_model":"Q2 historical XGBoost/Ridge and battery reserve" if variant=="4-2" else "Q3 attachment3 rolling forecasts and net-load buffer","rolling_solves":len(audit),"scipy_version":scipy.__version__,"input_sha256":data.hashes,"checks":checks,"max_milp_gap":float(audit.milp_gap.max()),**summarize(delivered)}
    q3.save_json(out/f"problem{variant}_summary.json",summary)
    return delivered,summary


def write_forecast_outputs(out,paths,dates):
    paths.selection.to_csv(out/"price_model_january_validation.csv",index=False)
    paths.audit.to_csv(out/"price_information_audit.csv",index=False)
    diagnostics(paths).to_csv(out/"price_forecast_metrics.csv",index=False)
    records=[]
    flat=paths.actual.ravel()
    for day,date in enumerate(dates):
        for phase,hour in enumerate((0,6,12,18)):
            start=day*N+hour*6; n=min(N,len(flat)-start)
            issue=pd.Timestamp(date)+pd.Timedelta(hours=hour)
            for j in range(n):
                records.append({"issue_datetime":str(issue),"target_end":str(issue+pd.Timedelta(minutes=(j+1)*10)),"lead_slot":j+1,"raw_prediction":paths.raw[day,phase,j],"scenario_mean_prediction":paths.mean[day,phase,j],"radius":paths.radius[day,phase,j],"lower":max(1e-6,paths.mean[day,phase,j]-paths.radius[day,phase,j]),"upper":paths.mean[day,phase,j]+paths.radius[day,phase,j],"actual_price_for_evaluation_only":flat[start+j]})
    pd.DataFrame(records).to_csv(out/"price_forecasts_all_issues.csv.gz",index=False,float_format="%.10f",compression={"method":"gzip","mtime":0})


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--days",type=int,default=365)
    parser.add_argument("--out",type=Path,default=OUT)
    parser.add_argument("--skip-comparison",action="store_true")
    args=parser.parse_args(); args.out.mkdir(parents=True,exist_ok=True)
    data,paths,net2,reserve2,pv2,buffer3=read_inputs()
    print("January-only chosen price model:",paths.selected_model,flush=True)
    write_forecast_outputs(args.out,paths,data.dates)
    results={}; comparisons=[]
    for variant in ("4-2","4-3"):
        frame,windows,audit=run(data,paths,net2,reserve2,pv2,buffer3,variant,end_day=args.days,keep_windows=True)
        if args.days<365:
            print(variant,q3.validate(frame),flush=True)
            continue
        delivered,summary=export_result(args.out,variant,frame,windows,audit,data,paths)
        results[variant]=summary
        comparisons.append({"variant":variant,"strategy":"interval_robust","radius_scale":1.,**summarize(delivered)})
        if not args.skip_comparison:
            initial=float(delivered.soc_start_kwh.iloc[0])
            for name,mode,scale in [("scenario_expected","forecast",0.),("half_radius","forecast",.5),("wider_radius","forecast",1.5),("fixed_price_planning","fixed_price_planning",0.),("perfect_price_benchmark","perfect_price_benchmark",0.)]:
                alt,_,_=run(data,paths,net2,reserve2,pv2,buffer3,variant,scale=scale,mode=mode,start_day=31,soc0=initial)
                q3.validate(alt)
                comparisons.append({"variant":variant,"strategy":name,"radius_scale":scale,**summarize(alt)})
                alt.groupby("date",as_index=False)[q3.SUM_COLS].sum().to_csv(args.out/f"problem{variant}_{name}_daily.csv",index=False)
    if args.days==365:
        pd.DataFrame(comparisons).to_csv(args.out/"problem4_strategy_comparison.csv",index=False)
        q3.save_json(args.out/"problem4_summary.json",results)
        print({k:{col:round(v[col],2) for col in ["total_cost_yuan","emergency_kwh"]} for k,v in results.items()},flush=True)


if __name__=="__main__": main()
