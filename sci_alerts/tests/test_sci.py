"""Offline tests of the SCI reporting pipeline. Synthetic market prices ONLY."""
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from sci_alerts.core import (
    DataQualityError, MarketSeries, analyze, expected_last_session, indicators,
    normalize, partial_status, resample_ohlc, threshold_events, crossover_events,
    verify_fresh, verify_soxx_split, wilder_rsi,
)
from sci_alerts.runner import split_message


def bars(start="2020-01-02", count=1650):
    dates = pd.bdate_range(start, periods=count)
    x = np.arange(count)
    p = 100 + .023 * x + 4 * np.sin(x / 9.7) + np.cos(x / 42)
    return pd.DataFrame({"open": p-.1, "high": p+1, "low": p-1, "close":p}, index=dates)


def test_indicators_every_period():
    result = analyze(bars(),datetime(2026,11,3,tzinfo=timezone.utc))
    assert set(result) == {"D","W","M"}
    for item in result.values():
        latest = item["ind"].iloc[-1]
        assert latest.notna().all()
        assert 0 <= latest["K"] <= 100
        assert 0 <= latest["RSI"] <= 100
    assert len(result["M"]["ind"]) >= 60


def test_rsi_edge_cases():
    idx = pd.bdate_range("2024-01-01", periods=55)
    assert wilder_rsi(pd.Series(10,index=idx)).iloc[-1] == 50
    assert np.isclose(wilder_rsi(pd.Series(np.arange(55), index=idx)).iloc[-1],100)


def test_indicators_flat_case():
    idx=pd.bdate_range("2025-01-01", periods=40)
    d=pd.DataFrame({"open":15.,"high":20.,"low":10.,"close":15.},index=idx)
    r=indicators(d).iloc[-1]
    assert r["K"] == 50
    assert r["D"] == 50
    assert r["RSI"] == 50
    assert r["CCI"] == 0


def test_events_distinguish_slope_and_crossing():
    assert any("과매도 상향 이탈" in x for x in threshold_events(19,21,20,80,"K"))
    assert not any("과매도 상향 이탈" in x for x in threshold_events(18,19,20,80,"K"))
    assert any("아직 이탈 아님" in x for x in threshold_events(95,85,20,80,"K"))
    assert any("과매수 하향 이탈" in x for x in threshold_events(85,79,20,80,"K"))
    assert crossover_events(10,11,12,9,"K")
    assert not crossover_events(11,12,10,9,"K")


def test_invalid_ohlc_and_insufficient_history():
    data=pd.DataFrame({"date":["2026-11-02"],"open":[10],"high":[9],"low":[7],"close":[11]})
    with pytest.raises(DataQualityError,match="OHLC"):
        normalize(data,"test",enforce_history=False)
    data["high"]=12
    with pytest.raises(DataQualityError,match="이력 부족"):
        normalize(data,"test")


def test_split_blocking():
    df=pd.DataFrame({"close":[450,149]},index=pd.to_datetime(["2026-11-04","2026-11-05"]))
    with pytest.raises(DataQualityError,match="분할"):
        verify_soxx_split(df)
    df.loc[pd.Timestamp("2026-11-04"),"close"]=150
    verify_soxx_split(df)


def test_exchange_calendar():
    assert str(expected_last_session(datetime(2026,11,3,0,0,tzinfo=timezone.utc)))=="2026-11-02"
    assert str(expected_last_session(datetime(2026,11,2,15,0,tzinfo=timezone.utc)))=="2026-10-30"


def test_ohlc_aggregation_and_provisional():
    idx=pd.to_datetime(["2026-10-29","2026-10-30","2026-11-02"])
    d=pd.DataFrame({"open":[1,2,3],"high":[2,3,4],"low":[.5,1.5,2.5],"close":[1.5,2.5,3.5]},index=idx)
    weekly=resample_ohlc(d,"W")
    assert weekly.iloc[0]["open"]==1
    assert weekly.iloc[0]["close"]==2.5
    assert partial_status(d,"W")=="잠정"


def test_stale_series_rejected():
    s=MarketSeries("SOXX",bars(count=100),"mock","mock")
    with pytest.raises(DataQualityError, match="날짜 불일치"):
        verify_fresh(s,datetime(2026,11,2).date())


def test_telegram_message_splitting():
    parts=split_message("line\n"*1300,3500)
    assert len(parts)>1
    assert max(map(len,parts))<=3500


def test_full_report_offline(monkeypatch,tmp_path):
    from sci_alerts import runner
    raw=bars()
    vix=raw/10
    last=raw.index[-1].date()
    monkeypatch.setattr(runner,"fetch_soxx",lambda: MarketSeries("SOXX",raw,"mock://soxx","fixture"))
    monkeypatch.setattr(runner,"fetch_vix",lambda: MarketSeries("VIX",vix,"mock://vix","fixture"))
    monkeypatch.setattr(runner,"expected_last_session",lambda now:last)
    monkeypatch.setattr(runner,"crosscheck_stooq_close",lambda s:"mocked")
    monkeypatch.setattr(runner,"STATE",tmp_path/"state.json")
    monkeypatch.setattr(runner,"PLOT_DIR",tmp_path/"output")
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN",raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID",raising=False)
    report=runner.run("morning",dry_run=True)
    assert not report["sent"]
    assert report["row_counts"]=={"SOXX":1650,"VIX":1650}
    assert (tmp_path/"output"/"latest_report.txt").exists()
    assert len(report["charts"])==2
    for p in report["charts"]:
        assert __import__("os").path.getsize(p)>100_000
    assert not (tmp_path/"state.json").exists()
