# SOXX · Cboe spot VIX — SCI scheduled reports

This module is independent of the existing `cloud_alerts.py` and `alerts.yml`.

## Data and calculations
- VIX: Cboe spot index official CSV `https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv`; **never** futures, CFDs or ETFs.
- SOXX: Yahoo Finance 7-year chart OHLC, with Stooq full-series fallback and split discontinuity check. Always disclose split/adj-close limitations. No fabricated numbers.
- Require at least 1,100 daily rows spanning 60 calendar months, no duplicate/missing/invalid OHLC, and newest expected completed U.S. regular session. If invalid, **fail closed** rather than send stale or misleading alerts.
- Indicators independently calculated on daily/weekly/monthly OHLC: Stochastic Fast %K5/D SMA3; CCI(HLC3) 20 + SMA10; Wilder RSI14 + SMA9. Crossovers are distinct from threshold crossings and slope changes. In-progress weekly/monthly bars are provisional.
- Chart images have 3 timeframes × 3 indicator axes for each instrument.
- This is **not validated to match a proprietary HTS formula**.

## Secrets and verification
Set two GitHub Actions repository secrets in Settings → Secrets and variables → Actions:
- `SCI_TELEGRAM_BOT_TOKEN` — the bot token
- `SCI_TELEGRAM_CHAT_ID` — destination Telegram chat ID

Never paste secrets in chat, issues, source files, or commits.

In Actions → SOXX-VIX SCI verified Telegram → Run workflow, select `morning` and `dry_run=true` to validate downloads, arithmetic and chart output. Inspect workflow logs and artifact. After successful dry-run and source verification, set `dry_run=false` to send one test Telegram message. If secrets are missing, the report is generated but not sent. Confirm its arrival before treating production deployment as complete.

GitHub Actions cron uses UTC:
- `0 0 * * 1-6` = KST Mon-Sat 09:00 (morning)
- `30 9 * * 1-5` = KST Mon-Fri 18:30 (evening)

GitHub Actions schedule is **not exactly-on-time guaranteed**. The Python runner checks KST day before sending. Avoid changing other active market alert automations.

## Local checks
```bash
pip install -r sci_alerts/requirements.txt pytest
pytest -q sci_alerts/tests
python -m sci_alerts.runner --session morning --dry-run
```
The last command requires outgoing internet access to Yahoo, Cboe, and, optionally, Stooq. Integration tests use synthetic data and **do not verify live prices**.

Source: https://www.cboe.com/tradable_products/vix/vix_historical_data
