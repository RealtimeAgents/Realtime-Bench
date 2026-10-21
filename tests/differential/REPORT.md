# 新旧对跑签字报告（P4b）

这份报告是 runner 切换（P1c）的合入门槛（wave2-plan.md 第 2 节 P1c、P4）。

## 1. 对跑的是什么

- **代码**：分支 `unify2-p4a`，基于 `unify-phase2` 的 `b5eae69`。kernel-fix 两轮、外壳 p1b、p5b、fps-fix 都已合入。
- **agent**：外壳那一方用 W0 的默认 KernelAgent，见 `entity/harness.py` `StreamingAgent.generate`、`commit`，它们跑的是旧接口。p2 的 `DefaultHarness` 实现还没合入，合入后要重跑一遍，X6–X8 到那时才会出现。
- **范围**：只覆盖 HTTP 类 API。墙钟下是 chat_completions，token 和暂停下是 openrouter；live 编码器不进对跑（R15）。

## 2. 结论

1. **外壳对离散事件模拟：完全一致。** 投影里的浮点按 1e-9 的容差比（X4）。51 个具名脚本、2016 个随机种子（0–2015）都一致。
2. **外壳对旧循环：除了清单内的差异，还有 5 条清单外的差异（U2–U6），每条都有规则解释，也都有具名脚本，见第 4 节。** 是进清单还是改代码，要协调者定。
   - U4、U5 的原因相同：环境自己锁了无效时，外壳先投 Invalidated，再投命令的回答（`evaluation/shell.py` `_Shell._env`）。
   - U4 让 kernel-fix 第 9 条的修正在外壳下不成立：token wait 自己那一步推进失败时，结局变回被截短的 wait。建议在 P1c 之前定。
3. **连跑 20 次：** 同时跑 20 次、依次跑 20 次，全部通过（第 5 节）。

## 3. 怎么比

同一份脚本跑三遍：

- 离散事件模拟（`simulated.py`）；
- 外壳 `run_episode`（`drivers.shell`）；
- 旧循环 `run_stream`（`drivers.legacy`，参数照 `main()` 传）。

外壳和旧循环用同一个回放世界：真 agent、回放桩、`FakeEnvironment` 或 `GuestEnvironment` 的子类。墙钟下每局一个新的假时钟。

- **外壳对模拟**：比投影，不套任何规则，必须一致。投影的内容：
  - 每个决策：观测的 tick、额度、提议的动作、两次时钟推进、落地记录、wait 记录、环境执行了什么、回报、lifecycle 结局；
  - 整局：采帧、token 下每次查结束的 tick、松键、放弃生成的次数、判无效、RunInvalidated 的条数、终止记录、快照落在第几个回报之后。
- **旧循环对外壳**：
  - 旧投影按编号规则（`rules.py`）改写后，要和外壳的投影一致；
  - 另外整份比请求（按回放基线的归一化，`tests/replay/baseline.py` `Normalizer`），比 agent 的用量，并按决策比投影外的四种轨迹记录：ObservationCaptured、InferenceStarted、OutputChunkReceived、InferenceCompleted；
  - 规则从投影里删掉的决策（内核结束以后旧循环才开始的），连同它们的请求一起不比：墙钟下是内核结束以后才发出的请求，token 下是这些决策的尝试次数。
- **归一化掉的**（方案"投影"一节）：
  - KernelRecord（X5）；
  - 序号和真实时间字段；
  - InferenceStarted 里只有外壳写的 `generation`、`remaining_clock_tokens`（p1b 报告 4.2 第 3 条）；
  - token 一侧采帧的 `capture_latency_s`，它在运维时钟上量，而 token 一侧的运维时钟是真实时间（`sandbox/actions.py` `GuestEnvironment.capture`）。
- **放弃生成的次数**：
  - 旧循环数 `agent.abort` 的调用；外壳和模拟数内核的放弃（`GenerationAbandoned` 记录，`AbortGeneration` 命令）。
  - 原因：外壳只中止还在跑的生成（`_Shell._abort`，p1b 4.2 第 6 条）；结果和中止同一刻时，调不调 `abort` 取决于调度。
- **模拟照外壳的写法投影**：
  - 墙钟下 lifecycle 的 `action_output_tokens` 写 None（`_Shell._commit`）。所以 P4a 报告里的 U1 已经不存在。
  - 环境自己锁的故障先以 Invalidated 交给内核，再交命令的回答（`ScriptedSimulation`，照 `_Shell._env`）。
- **同刻竞争（R13）**：假时钟上，外壳线程、worker 和墙钟松键定时器同一刻醒来时，谁先谁后由调度决定；读数的 ulp 误差也会改变两个同刻定时的先后。
  - 随机脚本先在模拟上检测（`simulated.ties`），有这类同刻就重抽：2016 个种子里，第一次抽到的有 53 个带同刻。
  - 具名脚本断言没有同刻。
  - 回复正好和结束同一刻、只有这次回复的记录有竞争时，这个决策的回复记录和用量不比（X3）。

## 4. 逐条

"出现"是指对跑里观察到，而且规则改写以后两边一致。

| 编号 | 结果 | 触发它的具名脚本 | 说明 |
|---|---|---|---|
| K-a | 出现 | `frame_rates-wall-60_1`、`frame_rates-wall-30_7`、`frame_rates-wall-2_2`、`runner-late_reply`、`runner-done_reply`、`runner-bounded_call_timeout`、`runner-long`、`wall-result-at-the-deadline`、`wall-retry-across-the-deadline` | |
| K-b | 出现 | `wall-ends-during-a-reply`、`wall-ends-while-pacing`、`wall-wait-cut-by-the-end` | |
| K-c | 出现 | `wall-wait-cut-by-the-end` | 已并入 KF3：墙钟 wait 被任务结束截短时，`truncated_by_terminal_condition` 为 True。kernel-fix 要删的 token 补充已删，但外壳下同样的结局又出现了，见 U4 |
| K-d | 出现 | `token-skipped-decision-after-the-end`、`token-think-budget-and-the-end` | |
| K-e | 出现 | `token-advance-fails-while-approaching`、`token-advance-fails-while-pacing` | |
| K-f | 出现 | `token-missing-usage` | |
| K-g | 出现 | 墙钟各脚本，例如 `runner-late_reply`、`wall-ends-during-a-reply`、`wall-think-budget-and-the-end`、`wall-third-failed-read-during-a-reply` | |
| K-h | 出现 | `wall-frozen-stream`、`token-frozen-stream` | |
| K-i | 出现 | `wall-think-budget-and-the-end`、`token-think-budget-and-the-end` | |
| K-j | 出现 | `token-lease-ends-inside-a-tap` | 外壳下环境照旧在点按内部按租约松键；外壳一侧的松键取内核 ReleaseKeys 的时刻 |
| K-k | 不可观察 | — | 墙钟下总帧数就是截止 |
| K-l | 不可观察 | `wall-per-move-cap`（会出"expected, not shown"的警告） | 回放里采帧到开始生成之间没有时间流逝 |
| K-m | 被 K-a 吸收 | — | |
| K-n | 出现 | `token-third-failed-read-before-landing`、`wall-third-failed-read-is-todays-first` | |
| K-poll | 不可观察 | — | `FakeEnvironment` 本来就落在整 tick 上 |
| K-rel | 出现 | `wall-hold-pressed-after-the-guards` | |
| KF1 | 作废 | — | fps-fix 已把旧循环的截断落点改成 T（kernel-fix 第二轮第 5 节） |
| KF2 | 出现 | `token-horizon-a-few-ulps-past-a-tick` | 旧循环驱动照 `main()` 传 `max_minutes * 60 * env_fps`。只做了有限 B：暂停模式下旧循环会死循环 |
| KF3 | 并入 K-c | `wall-wait-cut-by-the-end` | |
| X1 | 出现 | `frame_rates-token-*`、`runner-done_reply`、`runner-long`、`wall-wait-across-the-deadline`、`wall-wait-cut-by-the-end`、`token-a-wait-whose-own-step-fails` | |
| X2 | 出现 | 很多，例如 `runner-environment_done`、`wall-hold-between-frames`、`wall-early-end-snapshot-past-the-deadline` | |
| X3 | 出现 | `frame_rates-wall-*`、`runner-late_reply`、`runner-done_reply`、`runner-bounded_call_timeout`、`runner-long`、`wall-result-at-the-deadline`、`wall-retry-across-the-deadline`、`wall-ends-during-a-reply`、`wall-third-failed-read-during-a-reply` | 照 p1b 报告第 5 节第 5 条的做法：被放弃的生成不写回复记录，截止以后也不再重试。用量：回复在中止之前回来时照记，而中止排在读快照之后（`kernel/core.py` `Kernel.end`） |
| X4 | 出现 | `frame_rates-wall-30_7`、`wall-per-move-cap`、`token-horizon-a-few-ulps-past-a-tick` | 外壳和旧循环用同一个环境，环境一侧的浮点运算相同；剩下的只有内核和 runner 写法不同的那几个算式 |
| X5 | 归一化 | — | 投影去掉 KernelRecord |
| X6–X8 | 这次看不到 | — | p2 没合入 |
| X9 | 这次看不到 | — | P1d |
| X10 | 不进对跑 | — | live（R15） |
| X11、X14–X16 | 两边相同 | — | 两条路径用同一个环境。回放里键盘事件和动作不耗时，锁没人抢，收尾松键不失败 |
| X12 | 出现 | token：`token-two-failed-reads-in-a-row-are-tolerated`、`token-third-failed-read-before-landing`、`token-third-failed-read-inside-a-tap`、`token-a-read-inside-a-tap-resets-the-count`、`token-a-failed-read-inside-a-tap-counts`、`token-a-failed-read-after-the-end-seen-inside-a-tap`、`token-a-tap-goes-on-after-the-third-failed-read`、`token-a-tap-goes-on-after-the-third-failed-read-to-the-end`；墙钟：`wall-two-failed-reads-are-tolerated`、`wall-third-failed-read-during-a-reply`、`wall-a-successful-read-resets-the-count` | kernel-fix 第二轮的三条补充：(a) 同一读数下按读的先后计数，零耗时脚本的规划照读的先后排；(b) 读状态有耗时的墙钟脚本没写规则，没覆盖（第 7 节）；(c) 就是 `token-a-failed-read-after-the-end-seen-inside-a-tap`，kernel-fix 第三轮（决定 27）合入后要改 |
| X13 | 没有脚本 | — | 回放的读状态耗时就是脚本给的时间，到不了 10 秒 |

### 清单外的差异

| 编号 | 内容 | 具名脚本 | 来源 |
|---|---|---|---|
| U1 | 消失 | — | 外壳墙钟下 lifecycle 的 `action_output_tokens` 写 None，和旧循环一样 |
| U2 | 环境自己锁的无效，内核也写一条 RunInvalidated | `runner-environment_failure`、`token-advance-fails-*`、`token-a-wait-whose-own-step-fails`、`wall-a-failed-action-settles-after-the-snapshot-read` 等 | 就是 p1b 4.2 第 2 条 |
| U3 | token 下环境自己查到结束以后，旧循环还会再读一次状态 | `token-ends-while-pacing`、`token-skipped-decision-after-the-end` 等 | P4a 报告 3.4 |
| U4 | token 下 wait 自己那一步推进失败：内核按"被结束截短"记 `wait`（执行后结束、写 ActionApplied），旧循环记 `invalid_run` 和 `{"error": …}` | `token-a-wait-whose-own-step-fails` | 新发现，见下 |
| U5 | 墙钟执行失败、读快照又要时间时：内核在读完快照以后才结算这个决策，`environment_tick` 和 `next_observation_tick` 是读完以后的 tick | `wall-a-failed-action-settles-after-the-snapshot-read` | 新发现，见下 |
| U6 | 外壳给冻结流看门狗判冻结的那一帧也写 ObservationCaptured，旧循环不写 | `wall-frozen-stream`、`token-frozen-stream` | 就是 p1b 4.2 第 1 条 |

**U4、U5 的原因。** 环境自己锁了无效时（`sandbox/actions.py` `GuestEnvironment._advance_clock`、`_execution_failed`），外壳在命令的回答之前先投 Invalidated（`_Shell._env`）。Invalidated 和带错误的 ClockStepped、ActionExecuted 优先级相同，按投递的先后处理，所以内核先结束这一局：
- 正在等 wait 的决策，在 `TurnPolicy.on_end` 里当场按被截短的 wait 结算。推进失败的那个回答到的时候已经没人等了，`_step_failed` 本来会记下这次失败，让 `_waited` 记成 `invalid_run`。这就是 kernel-fix 第 9 条要的结局，但只在回答先到时成立。
- 正在执行的决策，等执行的回答到了才结算，而这时快照已经冻结了（`Kernel.end` 先发 FreezeSnapshot）。

模拟本来只投命令的回答，所以 P4a 里没有这两条。现在模拟照外壳的顺序投，外壳和模拟一致。要消掉这两条，可以让外壳在命令自己带着错误回答时不再另投 Invalidated，或者让内核在这两个阶段先处理命令的回答。都不在本包范围内。

**p1b 4.2 里其余几条**：
- 第 3 条：已归一化（第 3 节）。
- 第 6 条：按内核的放弃记录数（第 3 节）。
- 第 4、5、9 条：不在投影里。分别是 timeline 的 ulp、`action_resolved_unix_s` 这个真实时间字段、墙钟下显式写 0 的丢帧数。
- 第 7、10、11 条：回放里看不到，或者没写脚本。分别是 `configure_decision` 的先后、非 dict 的结果、日志。
- 第 8 条：没有哪个脚本在同一局里判两次无效。

## 5. 连跑 20 次

在 VM 上跑默认集合（`tests/differential`：51 个具名脚本、25 个随机种子、1 个缩减器自测，共 77 个）：

- 20 个进程同时跑：20 次都是 `77 passed`，每次 203–214 s。
- 20 次依次跑：20 次都是 `77 passed`，每次 126.72–134.74 s。
- 单独跑一次：`77 passed in 137.67s`，当时 VM 负载约 16–39，不是空闲时的数。

另外在 VM 上扫了随机种子 0–2015（32 个进程），全部通过。各规则触发的次数：

> X1 1079、X2 804、X3 404、K-g 385、U3 348、K-a 313、U2 167、X12 132、K-b 120、X4 51、K-f 47、K-e 31、U6 19、K-h 19、K-c 12、K-d 9、K-n 7、U5 4、U4 4、K-j 4、K-i 4；190 个种子没有任何差异。

## 6. 怎么跑

- 默认集合：`pytest tests/differential`。会话结束时打印每个脚本触发了哪些规则。
- 放大随机种子：`DIFF_SEEDS=500` 跑种子 0–499，`DIFF_SEEDS=500:1000` 跑一段。失败的随机脚本会被贪心缩减，最小的脚本打印出来，可以直接贴成具名脚本。

## 7. 没覆盖的

- X12 补充 b，以及墙钟下各操作有耗时的情形：墙钟规则按"操作不耗时"写，只有 K-rel、K-g、U5 的具名脚本用到耗时。
- 重试、墙钟读状态失败、墙钟下的思考预算和单步上限：随机脚本不生成，只有具名脚本。
- 一次点按不止一步（60 fps 以上）、有按住的键时点按内部读失败：只有具名脚本，或者没有。
- 暂停模式下的 KF2：旧循环会死循环。
- 投影不比的部分：timeline、日志、工具记录、轨迹里除上面四种以外的记录。
