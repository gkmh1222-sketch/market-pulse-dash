"""Scheduled independent SCI pipeline: fetch -> validate -> calculate -> report -> Telegram."""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import requests

from .core import (
    KST, DataQualityError, analyze, crosscheck_stooq_close,
    expected_last_session, fetch_soxx, fetch_vix, verify_fresh,
)

ROOT = Path(__file__).resolve().parent
STATE = ROOT / "state" / "last_state.json"
PLOT_DIR = ROOT / "output"
LINKS = {"SOXX": "https://www.tradingview.com/symbols/NASDAQ-SOXX/",
         "VIX": "https://www.tradingview.com/symbols/CBOE-VIX/"}
DISPLAY = {"K": "Stoch %K5", "D": "%D SMA3", "CCI": "CCI20", "CCI_signal": "CCI SMA10",
           "RSI": "RSI14", "RSI_signal": "RSI SMA9"}


def plot_report(symbol: str, data: dict, path: Path) -> None:
    fig, axs = plt.subplots(3, 3, figsize=(15, 12))
    defs = [("Stochastic 5,3,1", ["K", "D"], [20, 80]),
            ("CCI 20 + SMA10", ["CCI", "CCI_signal"], [-100, 100]),
            ("RSI Wilder14 + SMA9", ["RSI", "RSI_signal"], [30, 70])]
    for row, period in enumerate(("D", "W", "M")):
        item = data[period]
        ind = item["ind"]
        # More context for short timeframes, but monthly is naturally sparse.
        count = {"D": 45, "W": 24, "M": 18}[period]
        for col, (name, series, levels) in enumerate(defs):
            ax = axs[row, col]
            visible = ind.iloc[-count:]
            for s in series:
                ax.plot(visible.index, visible[s], linewidth=1.5, label=s)
                valid = visible[s].dropna()
                if len(valid):
                    v = valid.iloc[-1]
                    ax.scatter([valid.index[-1]], [v], s=10)
                    ax.annotate(f"{v:.1f}", (valid.index[-1], v), fontsize=8,
                                xytext=(-15, 7 if s == series[0] else -13),
                                textcoords="offset points")
            for level in levels:
                ax.axhline(level, color="gray", linestyle="--", linewidth=0.8, alpha=0.65)
            ax.set_title(f"{period} {name} | {'Confirmed' if item['status'] == '확정' else 'Provisional'}", fontsize=10)
            ax.grid(alpha=0.2)
            ax.legend(loc="best", fontsize=8)
            ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=5))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
            ax.tick_params(axis="x", labelrotation=30, labelsize=8)
    fig.suptitle(f"{symbol} SCI | US trading date {data['D']['asof_date']} | D / W / M", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def fmt(x: float) -> str:
    return "검증 불가" if pd.isna(x) else f"{float(x):.2f}"


def make_section(ticker: str, analyses: dict, source: str, details: str, extra: str) -> tuple[str, dict]:
    d = analyses["D"]
    pclose = float(d["ohlc"].iloc[-2]["close"])
    cclose = float(d["ohlc"].iloc[-1]["close"])
    title = f"[{ticker}] {d['asof_date']} 종가 {cclose:.3f} (직전 {pclose:.3f}, {(cclose/pclose-1)*100:+.2f}%)"
    lines = [title, f"출처: {source}", f"가격 기준: {details}", f"추가 검증: {extra}"]
    snapshot = {"source": source, "last_date": d["asof_date"], "last_close": cclose, "periods": {}}
    for period in ("D", "W", "M"):
        obj = analyses[period]
        ind = obj["ind"]
        before, now = ind.iloc[-2], ind.iloc[-1]
        lines += [f"\n{period}봉 ({obj['status']}, 기준 {obj['period_date']})"]
        for key in DISPLAY:
            prev, current = float(before[key]), float(now[key]); delta = current-prev
            direction = "상승" if delta > .01 else ("하락" if delta < -.01 else "횡보")
            lines.append(f"{DISPLAY[key]}: {prev:.2f}→{current:.2f} ({delta:+.2f}, {direction})")
        events = obj["events"]
        lines.append("이벤트: " + ("; ".join(events) if events else "신규 기준선 돌파/교차 없음"))
        snapshot["periods"][period] = {"status": obj["status"], "asof_date": obj["asof_date"],
            "period_date": obj["period_date"],
            "values": {key: float(now[key]) for key in DISPLAY}, "events": events}
    lines += [f"차트: {LINKS[ticker]}"]
    return "\n".join(lines), snapshot


def summarize(soxx: dict, vix: dict) -> str:
    """Qualitative, not a made-up numeric SCR score."""
    def direction(item):
        latest = item["D"]["ind"].iloc[-1]
        prev = item["D"]["ind"].iloc[-2]
        dirs = [latest[c] - prev[c] for c in ("K", "CCI", "RSI")]
        return sum(1 for x in dirs if x > 0.01), sum(1 for x in dirs if x < -0.01)
    so_up, so_down = direction(soxx)
    vi_up, vi_down = direction(vix)
    if so_up == 3 and vi_down == 3:
        verdict = "단기 롱 우세"
    elif so_down == 3 and vi_up == 3:
        verdict = "단기 숏 우세"
    elif (so_down >= 2 and vi_up >= 2) or (so_up >= 2 and vi_down >= 2):
        verdict = "전환 관찰"
    else:
        verdict = "혼조"
    return f"정성 판단: {verdict} — SOXX 상승은 주식시장에 우호적, VIX 상승은 위험회피 확대 방향입니다. 일·주·월 전체 지표가 반드시 일치한다는 뜻은 아닙니다. 특정 매매 지시나 수익 보장이 아닙니다."


def split_message(text: str, limit: int = 3700) -> list[str]:
    chunks = []
    buf = ""
    for line in text.splitlines(keepends=True):
        if len(buf) + len(line) > limit and buf:
            chunks.append(buf.rstrip())
            buf = ""
        while len(line) > limit:
            chunks.append(line[:limit]); line = line[limit:]
        buf += line
    if buf.strip():
        chunks.append(buf.rstrip())
    return chunks


def send_telegram(token: str, chat_id: str, text: str, images: list[Path]) -> None:
    base = f"https://api.telegram.org/bot{token}"
    for chunk in split_message(text):
        response = requests.post(base + "/sendMessage", json={"chat_id": chat_id, "text": chunk}, timeout=35)
        response.raise_for_status()
        if not response.json().get("ok"):
            raise RuntimeError("텔레그램 sendMessage API 오류")
    for path in images:
        with path.open("rb") as f:
            response = requests.post(base + "/sendPhoto", data={"chat_id": chat_id, "caption": path.stem},
                                     files={"photo": (path.name, f, "image/png")}, timeout=60)
        response.raise_for_status()
        if not response.json().get("ok"):
            raise RuntimeError("텔레그램 sendPhoto API 오류")


def run(session: str, *, dry_run: bool = False, now: datetime | None = None) -> dict:
    now = now or datetime.now(timezone.utc)
    kst = now.astimezone(KST)
    if not dry_run:
        # Guard against mistaken manual/UTC cron configuration. Morning Mon-Sat, evening Mon-Fri.
        permitted = (kst.weekday() <= 5 if session == "morning" else kst.weekday() <= 4)
        if not permitted:
            print(f"SKIPPED: KST weekday={kst.weekday()}, session={session}")
            return {"skipped": True, "reason": "outside dispatch weekday"}
    expected = expected_last_session(now)
    print(f"SCI begin: {kst.isoformat()}, session={session}, expected US close={expected}")
    s = fetch_soxx()
    v = fetch_vix()
    for series in (s, v):
        verify_fresh(series, expected)
        print(f"{series.ticker}: {len(series.data)} rows, first={series.data.index[0].date()}, last={series.data.index[-1].date()}, source={series.source}")
    soxx_xcheck = crosscheck_stooq_close(s)
    soxx = analyze(s.data, now)
    vix = analyze(v.data, now)
    prior = json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else {}
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    report_id = f"{session}:{expected.isoformat()}"
    if not dry_run and token and chat_id and prior.get("last_sent_report_id") == report_id:
        print(f"SKIPPED: duplicate Telegram report {report_id}")
        return {"skipped": True, "reason": "already sent"}
    previous_close_date = prior.get("last_verified_date")
    soxx_text, soxx_snap = make_section("SOXX", soxx, s.source, s.details, soxx_xcheck)
    vix_text, vix_snap = make_section("VIX", vix, v.source, v.details, "Cboe 공식 출처")
    if previous_close_date == expected.isoformat():
        compare = "이전 검증 이후 신규 확정봉 없음 / 값 유지"
    elif previous_close_date:
        compare = f"신규 확정봉 확인: 이전 {previous_close_date} → 이번 {expected}"
    else:
        compare = "첫 검증 완료: 과거 저장된 비교 기준 없음"
    title = f"SCI {session} | 한국 {kst.strftime('%Y-%m-%d %H:%M')} | 미국 {expected} 정규장 확정\n{compare}"
    text = "\n\n".join([title, summarize(soxx,vix), soxx_text, vix_text,
                          "수식: Fast Stoch K5/D SMA3; CCI HLC3(20)/SMA10; RSI Wilder14/SMA9. HTS 원본 수식과의 완전한 일치 여부는 미검증."])
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    paths = [PLOT_DIR / f"SCI_SOXX_{expected}.png", PLOT_DIR / f"SCI_VIX_{expected}.png"]
    plot_report("SOXX", soxx, paths[0])
    plot_report("VIX", vix, paths[1])
    (PLOT_DIR / "latest_report.txt").write_text(text, encoding="utf-8")
    updated = {"last_verified_date": expected.isoformat(), "source_checked_at_kst": kst.isoformat(),
               "last_sent_report_id": prior.get("last_sent_report_id", ""),
               "snapshots": {"SOXX": soxx_snap, "VIX": vix_snap}}
    if dry_run:
        print(f"DRY RUN - no Telegram or persistent state changes. {len(text)} report characters.")
    elif token and chat_id:
        send_telegram(token, chat_id, text, paths)
        updated["last_sent_report_id"] = report_id
        STATE.parent.mkdir(parents=True, exist_ok=True)
        STATE.write_text(json.dumps(updated, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print("Telegram delivery complete; state recorded")
    else:
        print("NO TELEGRAM SECRETS: report generated but NOT sent; state unchanged")
    return {"sent": bool(not dry_run and token and chat_id), "verified_date": str(expected),
            "row_counts": {"SOXX": len(s.data), "VIX": len(v.data)},
            "report_file": str(PLOT_DIR / "latest_report.txt"), "charts": [str(p) for p in paths],
            "crosscheck": soxx_xcheck}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", required=True, choices=["morning", "evening"])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        result = run(args.session, dry_run=args.dry_run)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except Exception as exc:
        # Fail closed: no stale/made-up values or misleading Telegram messages.
        print(f"SCI DATA VERIFICATION FAILED: {type(exc).__name__}: {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()