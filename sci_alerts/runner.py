"""Scheduled independent SCI pipeline: fetch -> validate -> calculate -> report -> Telegram."""
from __future__ import annotations

import argparse
import json
import math
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


def right_labels(ax, entries, levels=()):
    """Place labels in a separate right gutter, spaced in screen coordinates."""
    from matplotlib.colors import to_rgba
    ax.figure.canvas.draw()
    height = ax.get_window_extent().height
    gap = 28 / height
    low, high = 12 / height, 1 - 12 / height
    ordered = sorted(entries, key=lambda e: e[1])
    import numpy as np
    from itertools import combinations
    desired = [ax.transAxes.inverted().transform(ax.transData.transform((0, value)))[1]
               for _, value, _ in ordered]
    reserved = [ax.transAxes.inverted().transform(ax.transData.transform((0, v)))[1]
                for v in levels]
    candidates = [float(y) for y in np.linspace(low, high, 120)
                  if all(abs(y-r) >= 22 / height for r in reserved)]
    choices = (ys for ys in combinations(candidates, len(ordered))
               if all(b-a >= gap for a,b in zip(ys,ys[1:])))
    positions = min(choices, key=lambda ys: sum((y-d)**2 for y,d in zip(ys,desired)))
    labels = []
    for (name, value, color), y in zip(ordered, positions):
        # Threshold ticks occupy the inner gutter; boxed values occupy the outer one.
        label = ax.annotate(f"{name} {value:.2f}", xy=(1, value),
            xycoords=ax.get_yaxis_transform(), xytext=(1.18, y),
            textcoords=ax.transAxes, ha="left", va="center", fontsize=9,
            color=color, fontweight="bold", annotation_clip=False,
            bbox=dict(boxstyle="round,pad=0.3", fc=to_rgba(color, .09), ec=color, lw=.7),
            arrowprops=dict(arrowstyle="-", color=color, lw=.65))
        labels.append(label)
    return labels


def plot_report(symbol: str, data: dict, path: Path, period: str = "M") -> list[dict]:
    from matplotlib.patches import Rectangle
    from matplotlib import font_manager
    if any(f.name == "Malgun Gothic" for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = "Malgun Gothic"
    else:
        plt.rcParams["font.family"] = "NanumGothic" if any(
            f.name == "NanumGothic" for f in font_manager.fontManager.ttflist) else "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False
    item = data[period]
    count = {"D": 45, "W": 24, "M": 24}[period]
    visible = item["ind"].iloc[-count:]
    prices = item["ohlc"].iloc[-count:]
    fig, axs = plt.subplots(4, 1, figsize=(9, 12), sharex=True,
                            gridspec_kw={"height_ratios": [2, 1, 1, 1]})
    fig.subplots_adjust(left=.09, right=.70, top=.91, bottom=.07, hspace=.20)
    defs = [("Close", prices, ["close"], []),
            ("Stochastic 5,3,1", visible, ["K", "D"], [20, 80]),
            ("CCI 20 + SMA10", visible, ["CCI", "CCI_signal"], [-100, 100]),
            ("RSI Wilder14 + SMA9", visible, ["RSI", "RSI_signal"], [30, 70])]
    names = {"close": "Close", "K": "%K", "D": "%D", "CCI": "CCI",
             "CCI_signal": "SMA10", "RSI": "RSI", "RSI_signal": "SMA9"}
    audit, all_labels, threshold_labels = [], [], []
    for ax, (title, frame, series, levels) in zip(axs, defs):
        entries = []
        for i, key in enumerate(series):
            value = float(frame[key].iloc[-1])
            if not pd.notna(value) or not math.isfinite(value):
                raise DataQualityError(f"Non-finite latest chart value: {symbol}/{period}/{key}")
            color = "#252525" if i == 0 else "#1675ca"
            color = "#252525" if i == 0 else "#2196f3"
            if key == "close":
                x = mdates.date2num(prices.index.to_pydatetime())
                width = (float(pd.Series(x).diff().median()) if len(x)>1 else 1)*.58
                for date, bar in zip(x, prices.itertuples()):
                    candle_color = "#ff4747" if bar.close >= bar.open else "#2878e0"
                    ax.vlines(date, bar.low, bar.high, color=candle_color, lw=.7)
                    ax.add_patch(Rectangle((date-width/2,min(bar.open,bar.close)), width,
                        max(abs(bar.close-bar.open),.001), facecolor="white",
                        edgecolor=candle_color, lw=.8))
                for window, ma_color in ((5,"#252525"),(10,"#2196f3"),(20,"#d000ff")):
                    ma = item["ohlc"]["close"].rolling(window).mean().reindex(prices.index)
                    ax.plot(prices.index, ma, color=ma_color, lw=1, label=f"MA {window}")
            else:
                ax.plot(frame.index, frame[key], color=color, lw=1,
                        ls="--" if i and key != "D" else "-", label=names[key])
            entries.append((names[key], value, color))
            audit.append({"symbol": symbol, "period": period, "series": key,
                          "value": value, "display": f"{value:.2f}"})
        if levels:
            main = frame[series[0]]
            lower, upper = levels
            upper_color = {"K":"#e600ff", "CCI":"#ff7777", "RSI":"#d8200b"}[series[0]]
            lower_color = "#2475bf" if series[0] != "RSI" else "#6b80ff"
            ax.fill_between(frame.index, upper, main, where=main>upper,
                            interpolate=True, color=upper_color, alpha=.9 if series[0]=="K" else .65)
            ax.fill_between(frame.index, lower, main, where=main<lower,
                            interpolate=True, color=lower_color, alpha=.35)
            for level, line_color, caption in ((lower,lower_color,"과매도"),(upper,upper_color,"과매수")):
                ax.axhline(level, color=line_color, lw=.9)
                threshold_labels.append(ax.text(1.18,level,f"{caption} {level:g}",
                    transform=ax.get_yaxis_transform(),color=line_color,
                    fontsize=9,fontweight="bold",va="center",clip_on=False))
            if series[0] in ("K","RSI"):
                ax.set_ylim(-2,108)
                ax.set_yticks([0,25,50,75,100])
            else:
                ax.set_yticks([-100,0,100,200,300])
        ax.yaxis.tick_right()
        ax.axvline(frame.index[-1],color="#ff7777",lw=.6,alpha=.8)
        ax.margins(x=.025)
        ax.set_title(title, loc="left", fontsize=10)
        ax.grid(alpha=.2, linestyle=":")
        ax.legend(loc="upper left", fontsize=7, ncol=3, frameon=False)
        all_labels.extend(right_labels(ax, entries, levels))
    period_name = {"D":"일봉","W":"주봉","M":"월봉"}[period]
    fig.suptitle(f"{symbol} {period_name} SCI | {item['status']}\n자료 기준 {item['asof_date']}", fontsize=13)
    axs[-1].xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=7))
    axs[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y/%m/%d"))
    axs[-1].tick_params(axis="x", labelrotation=25, labelsize=8)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = [label.get_bbox_patch().get_window_extent(renderer) for label in all_labels]
    if any(a.overlaps(t.get_window_extent(renderer)) for a in boxes for t in threshold_labels):
        raise DataQualityError("Latest-value label overlaps threshold caption")
    if any(a.overlaps(b) for i, a in enumerate(boxes) for b in boxes[i+1:]):
        raise DataQualityError("Latest-value chart labels overlap")
    if any(not fig.bbox.contains(b.x0, b.y0) or not fig.bbox.contains(b.x1, b.y1) for b in boxes):
        raise DataQualityError("Latest-value chart label clipped")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return audit


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


def send_telegram(token: str, chat_id: str, text: str, images: list[Path]) -> list[dict]:
    receipts = []
    base = f"https://api.telegram.org/bot{token}"
    for chunk in split_message(text):
        response = requests.post(base + "/sendMessage", json={"chat_id": chat_id, "text": chunk}, timeout=35)
        if not response.ok or not response.json().get("ok"):
            raise RuntimeError("Telegram sendMessage rejected (credentials redacted)")
        receipts.append({"kind": "text", "message_id": response.json()["result"]["message_id"]})
    for path in images:
        with path.open("rb") as f:
            response = requests.post(base + "/sendPhoto", data={"chat_id": chat_id, "caption": path.stem},
                                     files={"photo": (path.name, f, "image/png")}, timeout=60)
        if not response.ok or not response.json().get("ok"):
            raise RuntimeError("Telegram sendPhoto rejected (credentials redacted)")
        receipts.append({"kind": "photo", "file": path.name, "message_id": response.json()["result"]["message_id"]})
    return receipts


def run(session: str, *, dry_run: bool = False, now: datetime | None = None, resend: bool = False) -> dict:
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
    if not resend and not dry_run and token and chat_id and prior.get("last_sent_report_id") == report_id:
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
    paths, label_audit = [], []
    for symbol, analysis in (("SOXX", soxx), ("VIX", vix)):
        for period in ("D", "W", "M"):
            path = PLOT_DIR / f"SCI_{symbol}_{period}_{expected}.png"
            label_audit.extend(plot_report(symbol, analysis, path, period))
            paths.append(path)
    (PLOT_DIR / "label_audit.json").write_text(
        json.dumps(label_audit, indent=2), encoding="utf-8")
    print(f"Chart labels verified: {len(label_audit)} values across {len(paths)} charts; no overlap/clipping")
    (PLOT_DIR / "latest_report.txt").write_text(text, encoding="utf-8")
    updated = {"last_verified_date": expected.isoformat(), "source_checked_at_kst": kst.isoformat(),
               "last_sent_report_id": prior.get("last_sent_report_id", ""),
               "snapshots": {"SOXX": soxx_snap, "VIX": vix_snap}}
    if dry_run:
        print(f"DRY RUN - no Telegram or persistent state changes. {len(text)} report characters.")
    elif token and chat_id:
        try:
            receipts = send_telegram(token, chat_id, text, paths)
        except requests.RequestException:
            raise RuntimeError("Telegram transport failed (credentials redacted)") from None
        (PLOT_DIR / "telegram_receipts.json").write_text(
            json.dumps(receipts, indent=2), encoding="utf-8")
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
    parser.add_argument("--resend", action="store_true", help="Explicit manual resend; bypass report deduplication only")
    args = parser.parse_args()
    try:
        result = run(args.session, dry_run=args.dry_run, resend=args.resend)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except Exception as exc:
        # Fail closed: no stale/made-up values or misleading Telegram messages.
        print(f"SCI DATA VERIFICATION FAILED: {type(exc).__name__}: {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
