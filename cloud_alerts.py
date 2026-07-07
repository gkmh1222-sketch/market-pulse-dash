# -*- coding: utf-8 -*-
"""
cloud_alerts.py — GitHub Actions(클라우드)에서 24시간 도는 알림 체커.
PC가 꺼져 있어도 이게 대신 돌면서: 사이트 명령(control_topic) 수신 → 알림 등록,
설정 포인트 도달 확인 → 폰 푸시(ntfy). 상태는 alerts_config.json(같은 폴더)에 저장.
"""
import os
import json
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
CFG = os.path.join(HERE, "alerts_config.json")
ALIAS = {
    "NQ": ("NQ=F", "미니나스닥"), "ES": ("ES=F", "미니S&P"), "YM": ("YM=F", "미니다우"),
    "RTY": ("RTY=F", "미니러셀"), "SPX": ("^GSPC", "S&P500"), "IXIC": ("^IXIC", "나스닥"),
    "NDX": ("^IXIC", "나스닥"), "DJI": ("^DJI", "다우"), "VIX": ("^VIX", "VIX"),
}


def load():
    with open(CFG, encoding="utf-8") as f:
        return json.load(f)


def save(c):
    with open(CFG, "w", encoding="utf-8") as f:
        json.dump(c, f, ensure_ascii=False, indent=2)


def push(topic, title, body, priority="high", tags="rotating_light"):
    requests.post(f"https://ntfy.sh/{topic}", data=body.encode("utf-8"),
                  headers={"Title": title, "Priority": priority, "Tags": tags}, timeout=15)


def price(yf, tk):
    try:
        m = yf.download(tk, period="1d", interval="1m", progress=False)
        c = m["Close"].squeeze().dropna()
        if len(c):
            return float(c.iloc[-1])
    except Exception:
        pass
    d = yf.download(tk, period="5d", progress=False)
    c = d["Close"].squeeze().dropna()
    return float(c.iloc[-1]) if len(c) else None


def poll_control(cfg):
    ct = cfg.get("control_topic")
    if not ct:
        return False
    last = cfg.get("last_cmd_ts", 0)
    try:
        r = requests.get(f"https://ntfy.sh/{ct}/json", params={"poll": "1", "since": "12h"}, timeout=15)
    except Exception:
        return False
    changed, newest = False, last
    for line in r.text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            m = json.loads(line)
        except Exception:
            continue
        if m.get("event") != "message":
            continue
        ts = m.get("time", 0)
        if ts <= last:
            continue
        newest = max(newest, ts)
        p = m.get("message", "").strip().split()
        if not p:
            continue
        op = p[0].upper()
        try:
            if op == "ADD" and len(p) >= 4 and p[1].upper() in ALIAS:
                tk, name = ALIAS[p[1].upper()]
                lvl = float(p[2].replace(",", "")); di = p[3].lower()
                if di in ("above", "below"):
                    cfg["alerts"].append({"ticker": tk, "name": name, "level": lvl,
                                          "direction": di, "fired": False})
                    changed = True
                    push(cfg["topic"], "Alert Updated",
                         f"알림 추가: {name} {lvl:,.0f} {'이상' if di=='above' else '이하'}",
                         "default", "gear")
            elif op == "DEL" and len(p) >= 2:
                i = int(p[1]) - 1
                if 0 <= i < len(cfg["alerts"]):
                    rm = cfg["alerts"].pop(i); changed = True
                    push(cfg["topic"], "Alert Updated", f"알림 삭제: {rm['name']}", "default", "gear")
            elif op == "CLEAR":
                cfg["alerts"] = []; changed = True
                push(cfg["topic"], "Alert Updated", "알림 전체 삭제", "default", "gear")
        except Exception:
            pass
    if newest != last:
        cfg["last_cmd_ts"] = newest
        changed = True
    return changed


def main():
    cfg = load()
    changed = poll_control(cfg)
    alerts = cfg.get("alerts", [])
    topic = cfg.get("topic")
    if topic and alerts:
        import yfinance as yf
        px = {}
        for a in alerts:
            if a["ticker"] not in px:
                px[a["ticker"]] = price(yf, a["ticker"])
        for a in alerts:
            cur = px.get(a["ticker"])
            if cur is None:
                continue
            up = a["direction"] == "above"
            hit = cur >= a["level"] if up else cur <= a["level"]
            if hit and not a.get("fired"):
                push(topic, "Market Pulse Alert",
                     f"{a['name']} {a['level']:,.0f} {'▲ 이상 도달' if up else '▼ 이하 도달'}\n현재가 {cur:,.2f}")
                a["fired"] = True; changed = True
            elif (not hit) and a.get("fired"):
                a["fired"] = False; changed = True
    if changed:
        save(cfg)
    print("cloud_alerts done. alerts:", len(alerts))


if __name__ == "__main__":
    main()
