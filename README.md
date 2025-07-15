for vol_cli: example python vol_cli.py --symbol ETHUSDT --interval 15m --start 2025-06-09T00:00:00 --end 2025-07-07T00:00:00


for hedging_analyzer: python hedging_analyzer.py --symbol "AVAXUSDT"


for tz_dist: python analyze.py \
  --symbol ETHUSDT \
  --start-date 2025-05-01 \
  --end-date   2025-07-09 \
  --tz-defs    '{"asia":["23:00","03:00"],"europe":["03:00","15:00"],"usa":["15:00","23:00"]}'
