# Hyperliquid Python SDK 已知问题

_更新: 2026-04-30_

## 问题 1: Testnet `Info()` 初始化崩溃 (IndexError)

- 受影响版本: `hyperliquid-python-sdk==0.22.0`
- 当前项目版本: `hyperliquid-python-sdk==0.23.0`
- 官方状态: upstream 已有 issue #275 和 duplicate #283，`0.23.0` release note 写明 `Fix constructing Info object`；不要再提交重复 issue。
- 环境: Testnet (`https://api.hyperliquid-testnet.xyz`)
- 旧错误: `IndexError: list index out of range` at `info.py:48`

### 根因

`Info.__init__()` 遍历 `spotMeta["universe"]`，用 `tokens[base]` 和 `tokens[quote]` 索引访问 token 信息。
Testnet 的 `spotMeta` 数据不一致：

- `tokens` 数组: 1599 个元素 (max index 1598)
- `universe` 引用的最大 token index: 1845
- 有 24 个 spot pair 引用了越界的 token index（从 universe[1246] 开始）

SDK 0.22.0 把 token index 当数组位置使用，所以越界崩溃。SDK 0.23.0 改为按 token 的 `index` 字段构建 lookup，修复了这个初始化问题。

### 影响

- 只影响 testnet，mainnet 数据一致性应该没问题
- 所有需要 `Info()` 的操作都受影响（查余额、持仓、挂单等）
- 如果你只用 perp，这个 bug 也会阻止你

### 当前项目处理

`cta-forge` 现在使用 `hyperliquid-python-sdk==0.23.0`，并在初始化时预取 `meta` / `spotMeta` 后同时传给 `Info` 和 `Exchange`，避免重复拉取 metadata。代码仍保留一个轻量防御：按 token 的 `index` 字段过滤掉引用不存在 token 的 spot pair，防止 testnet 脏数据影响 perp 初始化。

旧版本 workaround 曾经需要 monkey-patch `Info.__init__`，现在已经不需要。

### 旧版本绕过方案

1. 不用 SDK，直接调 REST API：

```python
import httpx

BASE = "https://api.hyperliquid-testnet.xyz"

# 查持仓
resp = httpx.post(f"{BASE}/info", json={
    "type": "clearinghouseState",
    "user": "0xYOUR_ADDRESS"
})
state = resp.json()
positions = [p for p in state["assetPositions"]
             if float(p["position"]["szi"]) != 0]

# 查挂单
resp = httpx.post(f"{BASE}/info", json={
    "type": "openOrders",
    "user": "0xYOUR_ADDRESS"
})
orders = resp.json()
```

2. 用 `skip_ws=True` + 传入预过滤的 `spot_meta`：

```python
from hyperliquid.info import Info
from hyperliquid.api import API

api = API("https://api.hyperliquid-testnet.xyz")
raw_meta = api.post("/info", {"type": "spotMeta"})

# 按 token 的 index 字段过滤掉引用不存在 token 的 universe 条目。
# 不要假设 token index 等于 tokens 数组位置。
token_indexes = {token["index"] for token in raw_meta["tokens"]}
raw_meta["universe"] = [
    u for u in raw_meta["universe"]
    if all(t in token_indexes for t in u["tokens"])
]

info = Info(
    base_url="https://api.hyperliquid-testnet.xyz",
    skip_ws=True,
    spot_meta=raw_meta,
)
```

### 是否应该提 issue?

暂时不要提。官方已有同类 issue，且 0.23.0 已修复构造 `Info` 的核心问题。除非在 0.23.0+ 上复现新的崩溃，并且能提供最小复现和版本信息，否则继续观察即可。

---

## 通用建议

- Testnet 和 Mainnet 数据一致性不保证，SDK 对 testnet 的兼容性较差
- 只需要 perp 功能时，考虑直接用 REST API 而不是 SDK
- SDK 的 `Info.__init__()` 会自动拉取 spotMeta 并初始化 WebSocket，开销不小
- 如果只需要查询（不下单），用 `httpx` 直接调 `/info` 端点更轻量
