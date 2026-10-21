# 新旧对跑签字报告（P4b）

这份报告是 runner 切换（P1c）的合入门槛（wave2-plan.md 第 2 节 P1c、P4）。对齐的是 `unify-phase2` 的 `55b0082`。

## 1. 对跑的是什么

- **代码**：分支 `unify2-p4a`，基于 `unify-phase2` 的 `55b0082`。已合入：
  - kernel-fix 第三到五轮（决定 27、31）、KF7；
  - p2 全部，所以 `DefaultHarness` 实现了 KernelAgent，X6–X8 会出现；
  - p5b 第二、三轮（决定 33、38）；
  - 外壳的第 47、48 条；
  - policy-fix（决定 25）；
  - stream-policy 给 `events.py` 加的字段。

  p1c 还没合入，所以旧循环就是 `runner.run_stream`。
- **三方**：
  - 离散事件模拟（`simulated.py`）；
  - 外壳 `run_episode`（`drivers.shell`）；
  - 旧循环 `run_stream`（`drivers.legacy`，参数照 `main()` 传）。

  外壳和旧循环用同一个回放世界：真 agent、回放桩、`FakeEnvironment` 或 `GuestEnvironment` 的子类。墙钟下每局一个新的假时钟。
- **范围**：只覆盖 HTTP 类 API。墙钟下是 chat_completions，token 和暂停下是 openrouter；live 编码器不进对跑（R15）。

## 2. 结论

1. **外壳对离散事件模拟：完全一致。** 投影里的浮点按 1e-9 的容差比（X4）。53 个具名脚本、2016 个随机种子（0–2015）都一致。
2. **外壳对旧循环：除了清单内的差异，只剩两条。**
   - **U7（清单外，生产代码的问题）**：外壳写的每条 InferenceCompleted，`frame_usage` 都是 0。外壳在内核接纳结果之前取用量差（`evaluation/shell.py` `_Shell._settle`），而 p2 要到 accept 时才累加用量（`DefaultHarness.accept`）。p1b 报告 L8 留下的约定没对上。按要求没改生产代码，写成规则 U7，记在第 6 节。
   - **U5（待 p1c 修）**：墙钟执行失败、读快照又要时间时，外壳先投环境锁存的 Invalidated、再投执行的回答。内核因此在读完快照以后才结算这个动作，tick 是读完以后的。具名脚本按旧循环的结局断言，标了 `xfail(strict=True, reason="p1c 修 U4/U5")`；随机脚本在修好之前避开这种组合（第 6 节）。
3. **连跑 20 次**：同时跑 20 次、依次跑 20 次，都没有偶发失败（第 5 节）。
4. **全量 rtest**：`RTEST TOTAL: 1819 passed, 0 failed or errored`，另有 1 个 xfailed（U5 的具名脚本），合计 1820，等于本地 `pytest --collect-only` 收集到的 1820 个。

## 3. 怎么比

- **外壳对模拟**：比投影，不套任何规则，必须一致。投影的内容：
  - 每个决策：观测的 tick、额度、提议的动作、两次时钟推进、落地记录、wait 记录、环境执行了什么、回报、lifecycle 结局；
  - 整局：采帧、token 下每次查结束的 tick、松键、放弃生成的次数、判无效、RunInvalidated 的条数、终止记录、快照落在第几个回报之后。
- **旧循环对外壳**：
  - 旧投影按编号规则（`rules.py`）改写以后要和外壳一致；
  - 另外整份比请求（按回放基线的归一化），比 agent 的用量，并按决策比投影外的四种轨迹记录：ObservationCaptured、InferenceStarted、OutputChunkReceived、InferenceCompleted；
  - 规则删掉的决策（内核结束以后旧循环才开始的），连同它们的请求、回复记录和用量一起不比。
- **读状态按内核的读法编号**（决定 31）。
  - 内核同一帧最多读一次，动作执行完再读一次；点按里读到结束就结束（决定 27）。
  - 旧循环多出来的读（循环开头和落地前在已读的帧上、结束以后、点按看到结束以后），不编号、哪里也不失败，照实际读任务。但在内核读失败的那一帧，这些读回答"没结束"，因为内核把这一帧当成读过了。所以"第 n 次读失败"在两边指同一次读（KF5f）。
  - 墙钟下旧循环的读里，执行后那次和每帧开头的第一次，对应内核也做的读，只有它们按时间窗失败。
  - 外壳路径下每次读都是内核的读。
- **回报**：外壳经 `commit(outcome)` 回报（p2），旧循环经 `commit_action`，两处都记下。
- **归一化掉的**（方案"投影"一节）：
  - KernelRecord（X5）；
  - 序号和真实时间字段；
  - token 一侧采帧的 `capture_latency_s`，它在运维时钟上量，而 token 一侧的运维时钟是真实时间；
  - wait 记录里推进失败的 `error`：外壳写整段 traceback，环境写异常文本（p1b 报告第 4 节第 6 条），只比消息那一行。
- **放弃生成的次数**：旧循环数 `agent.abort` 的调用，外壳和模拟数内核的放弃（`GenerationAbandoned`）。外壳只中止还在跑的生成，同一刻结果回来时调不调 `abort` 取决于调度。
- **模拟照外壳的做法**：
  - 墙钟下 lifecycle 的 `action_output_tokens` 写 None；
  - 执行失败（环境锁存无效、`execute()` 照常返回）先以 Invalidated 交给内核，再交回答；
  - 推进失败只交带错误的回答（`_Shell._env`、`_failure`）。
- **同刻竞争（R13）**：假时钟上，外壳线程、worker、墙钟松键定时器同一刻醒来时，谁先谁后由调度决定；读数的 ulp 误差也会改变两个同刻定时的先后。
  - 随机脚本先在模拟上检测（`simulated.ties`），有这类同刻就重抽。
  - 具名脚本断言没有同刻。

## 4. 逐条

"出现"是指对跑里观察到，而且规则改写以后两边一致。随机一栏是种子 0–2015 里触发的个数。

| 编号 | 结果 | 触发它的具名脚本 | 随机 | 说明 |
|---|---|---|---|---|
| K-a | 出现 | `frame_rates-wall-60_1`、`frame_rates-wall-30_7`、`frame_rates-wall-2_2`、`runner-late_reply`、`runner-done_reply`、`runner-bounded_call_timeout`、`runner-long`、`wall-result-at-the-deadline`、`wall-retry-across-the-deadline` | 311 | |
| K-b | 出现 | `wall-ends-during-a-reply`、`wall-ends-while-pacing`、`wall-wait-cut-by-the-end` | 120 | |
| K-c | 出现 | `wall-wait-cut-by-the-end` | 12 | 已并入 KF3 和 KF3 补充：被任务结束截短时标志为 True，时长算到结束的定日 |
| K-d | 出现 | `token-skipped-decision-after-the-end`、`token-think-budget-and-the-end` | 9 | |
| K-e | 出现 | `token-advance-fails-while-approaching`、`token-advance-fails-while-pacing` | 33 | |
| K-f | 出现 | `token-missing-usage` | 47 | |
| K-g | 出现 | 墙钟各脚本，例如 `runner-late_reply`、`wall-ends-during-a-reply`、`wall-a-wait-lands-after-the-end` | 384 | |
| K-h | 出现 | `wall-frozen-stream`、`token-frozen-stream` | 19 | |
| K-i | 出现 | `wall-think-budget-and-the-end`、`token-think-budget-and-the-end` | 4 | |
| K-j | 出现 | `token-lease-ends-inside-a-tap` | 4 | 外壳下环境照旧在点按内部按租约松键；外壳一侧的松键取内核 ReleaseKeys 的时刻 |
| K-k | 不可观察 | — | — | 墙钟下总帧数就是截止 |
| K-l | 不可观察 | `wall-per-move-cap`（会出"expected, not shown"的警告） | — | 回放里采帧到开始生成之间没有时间流逝 |
| K-m | 被 K-a 吸收 | — | — | kernel-fix 第五轮中 2 以后重新成立 |
| K-n | 不再出现 | — | — | 内核不在落地前读了（决定 31，KF5b、KF5d） |
| K-poll | 不可观察 | — | — | `FakeEnvironment` 本来就落在整 tick 上 |
| K-rel | 出现 | `wall-hold-pressed-after-the-guards` | — | 墙钟按住至少按 1/60 秒（决定 38）也照着做了；执行记录的 `lease_seconds` 不在投影里 |
| KF1 | 作废 | — | — | fps-fix 已让旧循环一致 |
| KF2 | 出现 | `token-horizon-a-few-ulps-past-a-tick` | — | 旧循环驱动照 `main()` 传 `max_minutes * 60 * env_fps`；只做了有限 B，暂停模式下旧循环会死循环 |
| KF3 | 并入 K-c | `wall-wait-cut-by-the-end` | — | |
| KF4 | 出现 | `token-the-end-seen-inside-a-tap`、`token-no-read-after-the-end-seen-inside-a-tap` | 1 | 原来的 X12 补充 c 作废 |
| KF5（a、b、c、f） | 出现 | token 各脚本，例如 `frame_rates-token-60_1`、`token-hold-between-frames` | 1294 | U3 并进 KF5a。KF5c 在零耗时的墙钟上看不出来。KF5f 体现在读的编号上，见第 3 节 |
| KF5d | 出现 | `runner-environment_done_during_reply`、`wall-a-wait-lands-after-the-end`、`wall-a-refused-action-lands-after-the-end` | 15 | 分普通动作、wait（含 done 和解析不出的回复）、被拒绝的动作三种 |
| KF5e | 没单独出现 | — | — | 零耗时的墙钟上，结束在帧内出现时，由 K-b 或 KF5d 解释 |
| KF6、KF7、D47、KF8 | 没有脚本 | — | — | KF6 要 agent 的 accept 抛异常；KF7 要整理结果有耗时，而回放里整理时假时钟不动；D47 要墙钟查结束有耗时；KF8 要时限不在整 tick 上，旧循环暂停模式会死循环 |
| X1 | 出现 | `frame_rates-token-*`、`runner-done_reply`、`runner-long`、`wall-wait-across-the-deadline`、`wall-wait-cut-by-the-end`、`token-a-wait-whose-own-step-fails` | 1083 | |
| X2 | 出现 | 很多，例如 `runner-environment_done`、`wall-hold-between-frames`、`wall-early-end-snapshot-past-the-deadline` | 802 | |
| X3 | 出现 | `frame_rates-wall-*`、`runner-late_reply`、`runner-done_reply`、`runner-bounded_call_timeout`、`runner-long`、`wall-result-at-the-deadline`、`wall-retry-across-the-deadline`、`wall-ends-during-a-reply`、`wall-third-failed-read-during-a-reply` | 402 | 第 48 条以后：截止放弃的生成，外壳记成一次放弃的调用（一条 `timed_out` 的 InferenceCompleted，没有输出块），这里核对它的形状；任务结束或判无效时放弃的，什么都不写。被放弃的生成都不进 agent 的用量（第 37 条） |
| X4 | 出现 | `frame_rates-wall-30_7`、`wall-per-move-cap`、`token-horizon-a-few-ulps-past-a-tick` | 50 | |
| X5 | 归一化 | — | — | 投影去掉 KernelRecord |
| X6 | 出现 | `wall-per-move-cap` | — | 只写了超时前没有已接纳记录的情形，见第 8 节；随机脚本不生成墙钟单步上限 |
| X7 | 出现 | `runner-done_reply` | 252 | 请求里的反馈改成 `actual=wait` 加 NOTE |
| X8 | 不比 | — | — | 在 agent 自己的事件日志里；对跑没有 Decision API |
| X9 | 这次看不到 | — | — | P1d |
| X10 | 不进对跑 | — | — | live（R15） |
| X11、X14、X15、X16 | 两边相同 | — | — | 两条路径用同一个环境；回放里键盘事件和动作不耗时，锁没人抢，收尾松键不失败 |
| X12 | 出现 | token：`token-two-failed-reads-in-a-row-are-tolerated`、`token-third-failed-read-after-the-action`、`token-third-failed-read-inside-a-tap`、`token-a-read-inside-a-tap-resets-the-count`、`token-a-failed-read-inside-a-tap-counts`、`token-a-tap-goes-on-after-the-third-failed-read`、`token-a-tap-goes-on-after-the-third-failed-read-to-the-end`；墙钟：`wall-two-failed-reads-are-tolerated`、`wall-third-failed-read-during-a-reply`、`wall-a-successful-read-resets-the-count` | 122 | 规划按内核的读来排（第 3 节）。补充 a：同一读数下按读的先后排；补充 b：没覆盖（第 8 节） |
| X13 | 没有脚本 | — | — | 回放的读状态耗时就是脚本给的时间，到不了 10 秒 |
| U2 | 出现，进清单 | `runner-environment_failure`、`token-advance-fails-*`、`token-a-wait-whose-own-step-fails`、`token-third-failed-read-after-the-action` 等 | 157 | 环境自己锁的无效，内核也写一条 RunInvalidated。依据：内核的事件日志本来就记每次判无效 |
| U3 | 进清单，就是 KF5a | 见 KF5 | — | 依据：第 31 条"截止后不读"和"同一帧最多读一次" |
| U4 | 消掉了 | `token-a-wait-whose-own-step-fails`（按旧循环的结局断言，通过） | — | kernel-fix 第五轮中 1，加上 p1b `d2b38b7`：推进失败只在回答里带错误 |
| U5 | 待 p1c 修 | `wall-a-failed-action-settles-after-the-snapshot-read`（xfail strict） | — | 见第 6 节 |
| U6 | 消掉了 | `wall-frozen-stream`、`token-frozen-stream` | — | p1b L3 `e9d5a9d`：判冻结的那一帧不再写 ObservationCaptured。依据（协调者定的）：只影响无效局的记录 |
| U7 | 出现，清单外 | 除 `wall-per-move-cap` 以外的所有具名脚本 | 1895 | 见第 6 节 |

## 5. 连跑 20 次和随机扫描

在 VM 上跑默认集合（`tests/differential`：53 个具名脚本、25 个随机种子、1 个缩减器自测，共 79 个）：

- 20 个进程同时跑：20 次都是 `78 passed, 1 xfailed`，每次 198–203 s。
- 20 次依次跑：20 次都是 `78 passed, 1 xfailed`，每次 127.32–145.96 s。
- 单独跑一次：`78 passed, 1 xfailed in 133.28s`，当时 VM 负载 7–26，不是空闲时的数。

随机种子 0–2015（32 个进程）全部通过。各规则触发的次数：

> U7 1895、KF5 1294、X1 1083、X2 802、X3 402、K-g 384、K-a 311、X7 252、U2 157、X12 122、K-b 120、X4 50、K-f 47、K-e 33、K-h 19、KF5d 15、K-c 12、K-d 9、K-j 4、K-i 4、KF4 1；4 个种子没有任何差异。

## 6. 清单外和待修的

- **U7：外壳下 InferenceCompleted 的 `frame_usage` 全是 0。**
  - 外壳整理结果时取 `usage_after`，在交给内核、内核发 AcceptResult 之前（`_Shell._settle`）。p2 以后 agent 只在 `accept` 时累加用量，所以取到的差总是 0。
  - 影响：只在轨迹记录里。整局用量、lifecycle、请求都两边一致。
  - 修法要 p1b 和 p2 约定：在 AcceptResult 之后取 `frame_usage`，或者让 `generate` 返回前就累加好。修好以后 U7 不再触发，具名脚本会出"expected, not shown"的警告，提醒去掉。
- **U5：墙钟执行失败，读快照又要时间时，失败的动作在读完快照以后才结算。**
  - 原因：环境锁存无效、`execute()` 照常返回 failed，外壳先投锁存的 Invalidated、再投回答（`_Shell._env`）。内核先结束这一局并冻结快照，执行中的决策等回答到了才结算（`TurnPolicy.on_end`）。
  - p1c 修：命令自己带错误回答时，先投回答、再投锁存的 Invalidated。
  - 修好以前：具名脚本 `wall-a-failed-action-settles-after-the-snapshot-read` 按旧循环的结局断言，标 `xfail(strict=True, reason="p1c 修 U4/U5")`，修好以后翻成 XPASS 提醒去掉；随机脚本不同时生成墙钟执行失败和有耗时的快照读（`generate.py`），修好以后放开。

## 7. 怎么跑

- 默认集合：`pytest tests/differential`。会话结束时打印每个脚本触发了哪些规则。
- 放大随机种子：`DIFF_SEEDS=500` 跑种子 0–499，`DIFF_SEEDS=500:1000` 跑一段。失败的随机脚本会被贪心缩减，最小的脚本打印出来，可以直接贴成具名脚本。

## 8. 没覆盖的

- X12 补充 b，以及墙钟下各操作有耗时的一般情形（KF7、D47）：墙钟规则按"操作不耗时"写，只有 K-rel、K-g 和 U5 的具名脚本用到耗时。
- X6 只写了"超时前没有已接纳记录"的情形。先有接纳的记录、再超时时，旧循环会改写那条记录的反馈，这牵扯 agent 怎么渲染历史，没写。
- 重试、墙钟读状态失败、墙钟下的思考预算和单步上限：随机脚本不生成，只有具名脚本。
- 一次点按不止一步（60 fps 以上）、有按住的键时点按内部读失败：只有具名脚本，或者没有。
- 暂停模式下的 KF2、KF8：旧循环会死循环。
- 投影不比的部分：timeline、日志、工具记录、agent 的事件日志（X8）、轨迹里除上面四种以外的记录。
