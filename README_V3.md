# WeatherBet v3 改造版

这是针对 `alteregoeth-ai/weatherbot` 的保守改造原型，仍然只做 paper trading。

## 主要变化

1. 每个温度 bucket 使用连续概率分布，不再把点预测映射为 0% 或 100%。
2. 同一个 Event 的所有 bucket 概率会统一归一化为 1。
3. 当天读取过去 24 小时 METAR，并以“已观测最高温”排除不可能结果。
4. 从 Polymarket 公共 CLOB `/book` 读取 Yes token 的真实 bids/asks。
5. 按 ask 深度计算 VWAP，不再把 Gamma `outcomePrices` 误作 bid/ask。
6. 概率先做 haircut，再用 0.15 fractional Kelly，并设置 Event 和天气组合上限。
7. 现金、持仓成本和 realized PnL 分开记录，避免 balance 重复累加。
8. 删除机械价格止损；后续退出应基于重新计算后的概率是否低于可卖价格。

## 使用

```bash
pip install requests
python weatherbot_v3.py init
python weatherbot_v3.py scan
python weatherbot_v3.py status
python weatherbot_v3.py run
```

数据写入 `data_v3/state.json`。

## 当前限制

- 市场规则仍需要人工复核。自动解析只处理常见整数、区间、or above/or below 文案。
- Open-Meteo 的 ECMWF/GFS 仍是点预报加误差分布，不是真正 ensemble 成员。
- METAR 可能不是每个市场最终指定的官方结算源。
- 尚未实现结算、信息止损、maker 挂单和真实交易。
- 每次 scan 对单个 Event 最多开一个仓位，防止重复下注。

## 推荐下一步

先保存 200–500 个已结算 Event 的预测快照，按城市、提前量和季节估计 bias/sigma；之后再实现信息止损与 maker-first 执行。不要在校准完成前开启真实下单。
