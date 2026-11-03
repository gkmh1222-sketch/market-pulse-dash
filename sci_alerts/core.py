"""SOXX + Cboe spot VIX source validation and independently calculated SCI."""
from __future__ import annotations

import io
import math
import os
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import exchange_calendars as xcals
import requests

VIX_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"
YAHOO_ENDPOINTS = (
    "https://query1.finance.yahoo.com/v8/finance/chart/SOXX",
    "https://query2.finance.yahoo.com/v8/finance/chart/SOXX",
)
STOOQ_URL = "https://stooq.com/q/d/l/"
MIN_BARS = 1100
MIN_MONTHS = 60
NY = ZoneInfo("America/New_York")
KST = ZoneInfo("Asia/Seoul")


class DataQualityError(RuntimeError):
    pass


@dataclass
class MarketSeries:
    ticker: str
    data: pd.DataFrame  # index=date (pd.DatetimeIndex, naive), open/high/low/close
    source: str
    details: str


def get_response(url: str, *, params: dict | None = None, attempts: int = 3) -> requests.Response:
    last = None
    for attempt in range(attempts):
        try:
            r = requests.get(url, params=params, timeout=35, headers={
                "User-Agent": "Mozilla/5.0 (compatible; SCI-data-check/1.0)",
                "Accept": "text/csv,application/json,text/plain,*/*",
            })
            r.raise_for_status()
            return r
        except requests.RequestException as exc:
            last = exc
            if attempt < attempts - 1:
                time.sleep(2 ** attempt)
    raise DataQualityError(f"다운로드 실패: {url}: {last}") from last


def normalize(df: pd.DataFrame, source: str, *, enforce_history: bool = True) -> pd.DataFrame:
    cols = {str(c).strip().lower(): c for c in df.columns}
    required = ("date", "open", "high", "low", "close")
    missing = [c for c in required if c not in cols]
    if missing:
        raise DataQualityError(f"{source}: 필수 컬럼 누락 {missing}")
    out = df[[cols[x] for x in required]].copy()
    out.columns = required
    out["date"] = pd.to_datetime(out["date"], errors="coerce", format="mixed").dt.tz_localize(None).dt.normalize()
    for c in required[1:]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    if out.isna().any().any():
        raise DataQualityError(f"{source}: 날짜/OHLC 결측 또는 잘못된 값 {out.isna().sum().to_dict()}")
    if out["date"].duplicated().any():
        raise DataQualityError(f"{source}: 중복 거래일 {out.loc[out['date'].duplicated(),'date'].tolist()[:5]}")
    out = out.sort_values("date").set_index("date")
    if not out.index.is_monotonic_increasing:
        raise DataQualityError(f"{source}: 시계열 정렬 실패")
    o, h, l, c = [out[x] for x in ("open", "high", "low", "close")]
    invalid = (out <= 0).any(axis=1) | (h < pd.concat([o, c, l], axis=1).max(axis=1) - 1e-6) | (l > pd.concat([o, c, h], axis=1).min(axis=1) + 1e-6)
    if invalid.any():
        raise DataQualityError(f"{source}: OHLC 상하관계 오류 {out.index[invalid].date[:5].tolist()}")
    if enforce_history and (len(out) < MIN_BARS or out.index.to_period("M").nunique() < MIN_MONTHS):
        raise DataQualityError(f"{source}: 장기 이력 부족 {len(out)}봉, {out.index.to_period('M').nunique()}개월 (최소 {MIN_BARS}봉·{MIN_MONTHS}개월)")
    return out


def fetch_vix() -> MarketSeries:
    r = get_response(VIX_URL)
    # Official CSV: DATE,OPEN,HIGH,LOW,CLOSE (MM/DD/YYYY).
    df = normalize(pd.read_csv(io.StringIO(r.text)), "Cboe VIX_History")
    return MarketSeries("VIX", df, VIX_URL, "Cboe 현물지수 공식 OHLC; 선물/ETF/CFD 아님")


def fetch_soxx() -> MarketSeries:
    end = datetime.now(timezone.utc) + timedelta(days=4)
    start = end - timedelta(days=365 * 7)
    params = {
        "period1": int(start.timestamp()), "period2": int(end.timestamp()),
        "interval": "1d", "events": "div,splits", "includeAdjustedClose": "true",
    }
    errors = []
    for endpoint in YAHOO_ENDPOINTS:
        try:
            r = get_response(endpoint, params=params)
            body = r.json().get("chart", {})
            data = (body.get("result") or [None])[0]
            if not data or not data.get("timestamp"):
                raise DataQualityError(f"응답에 SOXX 시계열 없음: {body.get('error')}")
            quotes = data["indicators"]["quote"][0]
            raw = []
            for i, timestamp in enumerate(data["timestamp"]):
                vals = [quotes.get(k, [None] * len(data["timestamp"]))[i] for k in ("open", "high", "low", "close")]
                if any(v is None for v in vals):
                    continue
                dt = datetime.fromtimestamp(timestamp, timezone.utc).astimezone(NY).date()
                raw.append((dt, *vals))
            df = normalize(pd.DataFrame(raw, columns=["date", "open", "high", "low", "close"]), "Yahoo SOXX")
            verify_soxx_split(df)
            # Do NOT splice dividend-adjusted Adj Close into OHLC.
            return MarketSeries("SOXX", df, endpoint, "Yahoo Finance 동일 시계열의 OHLC; Adj Close 혼용하지 않음; 주식분할 연속성 검사")
        except (KeyError, IndexError, TypeError, ValueError, requests.RequestException, DataQualityError) as exc:
            errors.append(f"{endpoint}: {exc}")
    # Provider-level fallback, not a splice of dissimilar adjusted OHLC series.
    try:
        rr = get_response(STOOQ_URL, params={"s": "soxx.us", "i": "d", "d1": start.strftime("%Y%m%d"), "d2": end.strftime("%Y%m%d")})
        fallback = normalize(pd.read_csv(io.StringIO(rr.text)), "Stooq SOXX")
        verify_soxx_split(fallback)
        return MarketSeries("SOXX", fallback, rr.url, "Stooq 동일 제공처 OHLC (별도 배당/분할 보정 정책); Yahoo 수집 실패로 전 구간 대체")
    except Exception as exc:
        errors.append(f"Stooq: {exc}")
    raise DataQualityError("SOXX 장기 자료 확보 실패: " + "; ".join(errors))


def verify_soxx_split(df: pd.DataFrame) -> None:
    """Guard against a fake -67% SCI move across SOXX 2026-11-05 3:1 split.

    Yahoo's OHLC is generally split-normalized, but do not assume its backend
    has adjusted immediately on ex-date: explicitly check neighboring closes.
    """
    pre = df.loc[df.index < "2026-11-05", "close"]
    post = df.loc[df.index >= "2026-11-05", "close"]
    if len(pre) and len(post):
        ratio = float(pre.iloc[-1] / post.iloc[0])
        if ratio > 2.0:
            raise DataQualityError(f"SOXX 3:1 분할 전후 OHLC 미조정 의심 (직전/직후 종가 비율={ratio:.3f}). 지표 계산 보류")
        if ratio < 0.45:
            raise DataQualityError(f"SOXX 분할 전후 과잉 조정 의심 (종가 비율={ratio:.3f}). 지표 계산 보류")


def expected_last_session(now: datetime | None = None) -> date:
    now = now or datetime.now(timezone.utc)
    ny = now.astimezone(NY)
    cal = xcals.get_calendar("XNYS")
    valid = cal.sessions_in_range(ny.date() - timedelta(days=20), ny.date())
    sessions = [d.date() for d in valid]
    # SOXX closes at 16:00 ET; VIX spot publishes until 16:15 ET.
    # 17:00 ET is conservative and avoids premature close adoption.
    eligible = [d for d in sessions if d < ny.date() or (d == ny.date() and (ny.hour, ny.minute) >= (17, 0))]
    if not eligible:
        raise DataQualityError("미국 최근 정규장 날짜를 판단하지 못했습니다")
    return eligible[-1]


def verify_fresh(series: MarketSeries, expected: date) -> None:
    latest = series.data.index[-1].date()
    if latest != expected:
        raise DataQualityError(f"{series.ticker} 정규장 날짜 불일치: 실제 {latest}, 예상 {expected} (출처: {series.source})")


def crosscheck_stooq_close(series: MarketSeries) -> str:
    """Independent secondary check, never silently treats unavailable data as verified."""
    if "stooq.com" in series.source.lower():
        return "Stooq가 주 출처: Yahoo 별도 교차검증 미완료"
    last = series.data.index[-1].date()
    start = last - timedelta(days=12)
    params = {"s": "soxx.us", "i": "d", "d1": start.strftime("%Y%m%d"), "d2": last.strftime("%Y%m%d")}
    try:
        r = get_response(STOOQ_URL, params=params, attempts=2)
        alt = normalize(pd.read_csv(io.StringIO(r.text)), "Stooq SOXX 보조 검증", enforce_history=False)
        if last not in alt.index.date:
            return "보조 출처 최신 거래일 지연(교차검증 미완료)"
        other = float(alt.loc[pd.Timestamp(last), "close"])
        primary = float(series.data.iloc[-1]["close"])
        diff = abs(other / primary - 1)
        if diff > 0.003:
            raise DataQualityError(f"SOXX 최신 종가 출처 차이 {diff:.2%}: Yahoo {primary:.4f} vs Stooq {other:.4f}; 발송 보류")
        return f"Stooq 최신 종가 일치 (차이 {diff:.3%})"
    except DataQualityError as exc:
        # Material price disagreements are fatal; access failures are disclosed.
        if "출처 차이" in str(exc):
            raise
        return f"보조 교차검증 미완료: {exc}"
    except Exception as exc:
        return f"보조 교차검증 미완료: {type(exc).__name__}: {exc}"


def resample_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    # NYSE/US market sessions come with New York local trading dates.
    if rule == "D":
        return df.copy()
    if rule == "W":
        grp = df.resample("W-FRI", label="right", closed="right")
    elif rule == "M":
        grp = df.resample("ME", label="right", closed="right")
    else:
        raise ValueError(rule)
    return grp.agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()


def wilder_rsi(close: pd.Series, length: int = 14) -> pd.Series:
    a = close.to_numpy(dtype=float)
    out = np.full(len(a), np.nan, dtype=float)
    if len(a) <= length:
        return pd.Series(out, index=close.index)
    deltas = np.diff(a)
    gain = float(np.maximum(deltas[:length], 0).mean())
    loss = float(np.maximum(-deltas[:length], 0).mean())
    for pos in range(length, len(a)):
        if pos > length:
            ch = float(deltas[pos - 1])
            gain = (gain * (length - 1) + max(ch, 0.0)) / length
            loss = (loss * (length - 1) + max(-ch, 0.0)) / length
        out[pos] = 50.0 if gain == loss == 0 else (100.0 if loss == 0 else 100 - 100 / (1 + gain / loss))
    return pd.Series(out, index=close.index)


def indicators(df: pd.DataFrame) -> pd.DataFrame:
    lo5 = df["low"].rolling(5, min_periods=5).min()
    hi5 = df["high"].rolling(5, min_periods=5).max()
    spread = hi5 - lo5
    k = (100.0 * (df["close"] - lo5) / spread.where(spread != 0))
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    mean20 = tp.rolling(20, min_periods=20).mean()
    md20 = tp.rolling(20, min_periods=20).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    cci = (tp - mean20) / (0.015 * md20.where(md20 != 0))
    cci = cci.mask(md20.eq(0) & mean20.eq(tp), 0.0)
    rsi = wilder_rsi(df["close"], 14)
    return pd.DataFrame({
        "K": k, "D": k.rolling(3, min_periods=3).mean(),
        "CCI": cci, "CCI_signal": cci.rolling(10, min_periods=10).mean(),
        "RSI": rsi, "RSI_signal": rsi.rolling(9, min_periods=9).mean(),
    })


def threshold_events(prev: float, cur: float, lo: float, hi: float, title: str) -> list[str]:
    if not math.isfinite(prev) or not math.isfinite(cur):
        return []
    result = []
    if prev <= lo and cur > lo:
        result.append(f"{title} {lo:g}선 과매도 상향 이탈 ({prev:.2f}→{cur:.2f})")
    elif prev >= lo and cur < lo:
        result.append(f"{title} {lo:g}선 과매도 진입 ({prev:.2f}→{cur:.2f})")
    if prev >= hi and cur < hi:
        result.append(f"{title} {hi:g}선 과매수 하향 이탈 ({prev:.2f}→{cur:.2f})")
    elif prev <= hi and cur > hi:
        result.append(f"{title} {hi:g}선 과매수 진입 ({prev:.2f}→{cur:.2f})")
    elif prev > hi and cur > hi and cur < prev:
        result.append(f"{title} 과매수권 하락(아직 이탈 아님) ({prev:.2f}→{cur:.2f})")
    return result


def crossover_events(a_old: float, a_new: float, b_old: float, b_new: float, title: str) -> list[str]:
    if not all(math.isfinite(x) for x in (a_old, a_new, b_old, b_new)):
        return []
    if a_old <= b_old and a_new > b_new:
        return [f"{title} 상향 교차 (전 {a_old:.2f}/{b_old:.2f}, 후 {a_new:.2f}/{b_new:.2f})"]
    if a_old >= b_old and a_new < b_new:
        return [f"{title} 하향 교차 (전 {a_old:.2f}/{b_old:.2f}, 후 {a_new:.2f}/{b_new:.2f})"]
    return []


def classify(df: pd.DataFrame) -> list[str]:
    p, c = df.iloc[-2], df.iloc[-1]
    events = []
    for name, low, high in (("K", 20, 80), ("CCI", -100, 100), ("RSI", 30, 70)):
        events.extend(threshold_events(p[name], c[name], low, high, name))
    for name, sig in (("K", "D"), ("CCI", "CCI_signal"), ("RSI", "RSI_signal")):
        events.extend(crossover_events(p[name], c[name], p[sig], c[sig], name))
    return events


def partial_status(frame: pd.DataFrame, period: str, now: datetime | None = None) -> str:
    if period == "D":
        return "확정"
    now = (now or datetime.now(timezone.utc)).astimezone(NY)
    cal = xcals.get_calendar("XNYS")
    last_session = frame.index[-1].date()
    if period == "W":
        monday = last_session - timedelta(days=last_session.weekday())
        end = monday + timedelta(days=6)
    else:
        month = last_session.month
        year = last_session.year
        nextmonth = date(year + month // 12, month % 12 + 1, 1)
        end = nextmonth - timedelta(days=1)
        monday = last_session.replace(day=1)
    schedule = cal.sessions_in_range(monday, end)
    period_last_session = schedule[-1].date()
    return "확정" if last_session == period_last_session else "잠정"


def analyze(data: pd.DataFrame, now: datetime | None = None) -> dict:
    result = {}
    for period in ("D", "W", "M"):
        agg = resample_ohlc(data, period)
        ind = indicators(agg)
        if len(ind) < 30 or ind.iloc[-2:].isna().any().any():
            raise DataQualityError(f"{period}: SCI 계산 구간 부족/NaN")
        result[period] = {
            "ohlc": agg,
            "ind": ind,
            "events": classify(ind),
            "status": partial_status(data if period != 'D' else agg, period, now),
            "period_date": agg.index[-1].date().isoformat(),
            "asof_date": data.index[-1].date().isoformat(),
        }
    return result