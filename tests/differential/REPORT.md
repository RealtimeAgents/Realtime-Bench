# 新旧对跑签字报告（P4b）

这份报告是 runner 切换（P1c）的合入门槛（wave2-plan.md 第 2 节 P1c、P4）。对齐的是 `unify-phase2` 的 `cc2c853`，加上本分支把 token 那一半旧循环代码搬出生产环境的两个提交（`d4ecfd6`、`1b8c589`）。签字跑完时主线到了 `377b167`，多出的 20 个提交都在流式部分：其中 `core.py` 按 `ActionExecuted.finished_at` 归档动作的回答，而回合制外壳和模拟都不填这个字段，所以不影响这里的对跑。

## 1. 对跑的是什么

- **代码**：分支 `unify2-p4a`，基于 `unify-phase2` 的 `cc2c853`。和对齐 `55b0082` 的那次签字相比，多合入了：
  - p1c 的切换：
    - `main()` 跑在外壳上（`dffd397`）；
    - 外壳先投动作自己的回答，再投环境锁存的 Invalidated（`3f7fcb8`，N6）；
    - 接纳以后才取用量（`73ca9a3`）；
    - 旧循环原样搬到 `tests/differential/legacy_loop.py`（`883c735`，第 11 条），runner 去掉了内核没有的截止宽限；
  - kernel-fix 第六、七轮和第九到十一轮：
    - 墙钟动作在外壳 accept 之后的读数上落地（N25）；
    - 墙钟落地前在新读数上照读（N26 取 (b)）；
    - 读失败判无效记在发问时刻（N44）；
  - p1b 第四到七轮：
    - L1–L8；
    - 任务结束、判无效时放弃的生成合成记录（N10），这些记录不进 token 统计（N34）；
    - 外壳兜住内核和读数抛的异常（N11）；
  - token 回放替身的运维时钟随世界时间走（`a3a8d76`）；
  - 只有旧循环用的环境代码搬出生产的 `GuestEnvironment`（第 11 条），放进 `tests/differential/legacy_environment.py`：
    - 墙钟那一半（截止）由 p1c 第七轮搬，叫 `LegacyWallClock`；
    - token 那一半（`done()`、内核不计读时点按和步进里的读）由本分支搬，叫 `LegacyStateReads`；
  - N43、原生流式 agent 和 OpenAI Realtime、Gemini Live 的修正轮、stream-wire 第三四轮、流式报表，以及默认 harness、回放归一化的几处改动。
- **三方**：
  - 离散事件模拟（`simulated.py`）；
  - 外壳 `run_episode`（`drivers.shell`）；
  - 旧循环 `legacy_loop._run_stream_legacy`（`drivers.legacy`，参数照切换前的 `main()` 传）。

  外壳和旧循环用同一个回放世界：真 agent、回放桩，墙钟下是 `FakeEnvironment` 的子类，token 和暂停下是混入了 `LegacyStateReads` 的 `GuestEnvironment` 子类（`drivers.shell` 给它设了 `kernel_counts_reads`，走生产的那条路）。墙钟下每局一个新的假时钟。
- **范围**：只覆盖 HTTP 类 API。墙钟下是 chat_completions，token 和暂停下是 openrouter；live 编码器不进对跑（R15）。

## 2. 结论

1. **外壳对离散事件模拟：完全一致。** 62 个具名脚本、2016 个随机种子（0–2015）都一致。投影里的浮点按 1e-9 的容差比（X4）。第四轮起比较的范围扩大了：
   - 墙钟下每次读状态的时刻和答案；
   - 内核放弃的生成，外壳有没有中止；
   - 每个决策的记录有没有写两遍。
2. **外壳对旧循环：除了清单内的差异，没有清单外的差异。** `b3288a2` 那次签字剩下的 U14 已由 N44 消掉（第 7 节）。这一轮多抽了执行耗时，又发现一处疑似内核问题，只影响记录，用 PENDING 脚本钉住（第 7 节）。
3. **清单本身的变化**：
   - U4、U5 去掉了：p1c 按 N6 修了外壳，两个具名脚本按旧循环的结局断言，不再标 xfail。
   - `55b0082` 那次签字时自编的 U7（`frame_usage` 全为 0）在 `73ca9a3` 以后消失了，这个编号现在是 p1b 的。
   - X3 不再出现：放弃的生成，外壳都合成"放弃的调用"记录，归 K-a（截止）、U10（任务结束）、U11（判无效）。
   - N26 (b) 以后，KF5d 作废，K-n 又出现了。第四轮中途自编过一条 U13（旧循环落地前多读那一次的耗时），也随 N26 (b) 消失了；U13 这个编号现在是 p1b 第七轮的。
   - U14（读失败判无效的结束时刻）由 N44 消掉：内核和旧循环都记在发问时刻。
   - 补了 N29 要的读状态、采帧有耗时的具名脚本：K-b 扩写的推迟、D47 都有脚本钉住。耗时覆盖到哪见第 4 节。
4. **连跑 20 次**：同时跑 20 次、依次跑 20 次，都没有偶发失败（第 6 节）。
5. **全量 rtest**：`RTEST TOTAL: 2346 passed, 0 failed or errored`，等于本地 `pytest --collect-only` 收集到的 2346 个。
6. **kernel-fix 审查修复这一轮**：
   - 具名脚本 78 个：77 个通过；1 个记在 `scenarios.PENDING` 里，按旧循环的结局断言，xfail strict（第 7 节）。
   - 随机生成器在墙钟下加了读状态和执行的耗时、单步上限、思考预算（第 4 节）。种子 0–999 本地全部通过（第 6 节）。

## 3. 怎么比

- **外壳对模拟**：比投影，不套任何规则，必须一致。投影的内容：
  - 每个决策：观测的 tick、额度、提议的动作、两次时钟推进、落地记录、wait 记录、环境执行了什么、回报、lifecycle 结局；
  - 整局：
    - 采帧、token 下每次查结束的 tick、松键；
    - 放弃生成的次数、判无效、RunInvalidated 的条数；
    - 终止记录、快照落在第几个回报之后；
    - 写了两遍的记录（应当没有）。
  - 只在这两方之间比的部分（`Projection.kernel`）：
    - 墙钟下快照以前每次读状态的时刻和答案；
    - 内核放弃了、外壳却没中止也没跑完的生成（应当没有）。
- **旧循环对外壳**：
  - 旧投影按编号规则（`rules.py`）改写以后要和外壳一致。
  - 另外整份比请求（按回放基线的归一化），比 agent 的用量和已回报决策的 token 小计（U17），并按决策比投影外的四种轨迹记录：ObservationCaptured、InferenceStarted、OutputChunkReceived、InferenceCompleted。
  - 第四轮新加两项：
    - 除 wait 以外每个动作的 `execution_result`；
    - token 下任务时钟收到的键事件。
  - 规则删掉的决策（内核结束以后旧循环才开始的），连同它们的请求、回复记录和用量一起不比。
  - 内核放弃的生成，外壳写的是"放弃的调用"记录。比的是它的关键字段：
    - `timed_out` 和 `stale`、`usage` 为空、`frame_usage` 全 0；
    - 缺用量的原因、`harness_abort_reason` 按放弃原因区分；
    - 没有输出块。

    它的 ObservationCaptured、InferenceStarted 在放弃以前写，和别的决策一样整条比。`latency` 不比，由 `tests/shell/test_commands.py` 钉住。
- **读状态按内核的读法编号**（决定 31、N26）：
  - token 下，内核同一 tick 最多读一次，动作执行完再读一次；点按里读到结束就结束（决定 27）。
  - 墙钟下，内核也做旧循环的这几次读：执行后那次、落地前在新读数上那次、每帧开头的第一次，只有它们按时间窗失败。旧循环在帧中间的循环开头多读的那次不编号，哪里也不失败。
  - 旧循环多出来的读照实际读任务。在内核读失败的那一帧，它们回答"没结束"，因为内核把这一帧当成读过了。所以"第 n 次读失败"在两边指同一次读（KF5f）。
  - 外壳路径下每次读都是内核的读。
- **回报**：外壳经 `commit(outcome)` 回报，旧循环经 `commit_action`，两处都记下。
- **归一化掉的**（方案"投影"一节）：
  - KernelRecord（X5），包括 N25 新加的 `landing` 定时记录；
  - 序号和真实时间字段；
  - token 一侧采帧的 `capture_latency_s`；
  - wait 记录里推进失败的 `error`，只比消息那一行：外壳写整段 traceback，旧循环和模拟写异常文本（N22）；
  - `FakeEnvironment` 自己的执行序号：执行了哪些动作由投影的 `executed` 比。
- **放弃生成的次数**：旧循环数 `agent.abort` 的调用，外壳和模拟数内核的放弃（`GenerationAbandoned`）。外壳只中止还在跑的生成，所以另外核对：外壳的每一次放弃，要么调了 `agent.abort`，要么那次生成在放弃时已经返回。
- **模拟照外壳的做法**：
  - 墙钟下 lifecycle 的 `action_output_tokens` 写 None；
  - 执行失败时先交动作的回答（带错误），再交环境锁存的 Invalidated（N6，`_Shell._execute`）；推进失败只交带错误的回答（`_Shell._failure`）；
  - 键名用环境的叫法（`GuestEnvironment._key_identity`）；
  - 墙钟按下被耽搁的键至少再按住 min(时长, 1/60 秒)（第 38 条，N9）。
- **同刻竞争（R13）**：假时钟上，外壳线程、worker、墙钟松键定时器同一刻醒来时，谁先谁后由调度决定；读数的 ulp 误差也会改变两个同刻定时的先后。随机脚本先在模拟上检测（`simulated.ties`），有这类同刻就重抽；具名脚本断言没有同刻。
  - N25 以后多了一种：回复正好落在帧格点上时，落地定时和定时读只差几个 ulp。外壳把它们并成一个读数处理，省掉落地前那次读；模拟当成两个读数，多读一次。
  - `frame_rates-wall-30_7` 第二次回复照搬回放测试是 0.2 秒，正好落在 11/30 秒，所以这里改成 0.21 秒。

## 4. 操作耗时覆盖到哪（N29）

- **随机脚本**：
  - 读快照有耗时；
  - 墙钟下各有两成的局，读状态花 0.143 或 0.317 个环境帧，执行花 0.121 或 0.269 个环境帧。采帧不耗时。这几个数和回复耗时、wait 时长（都在八分之一帧的格点上，30/7 下是 30/56 帧的格点）加起来，二十次操作以内落不到环境帧的格点上。落在格点上时，两边取 tick 差一个 ulp。原来取 0.1、0.3 帧时，五次操作就会落上去（种子 499）。
  - wait、DONE、FAIL 不交给客体，不花执行时间（`GuestEnvironment.execute`），墙钟替身照这个做。
  - 耗时会让旧循环在内核以外的时刻读到结局时，这一局先把读状态改回不耗时，还不行再把执行也改回不耗时（`simulated.timed_apart`）。包括三种情形：
    - 读状态有耗时，决策在帧中间回报；
    - 执行有耗时，被拒的动作执行完已过任务结束的时刻，决策在帧中间回报（种子 754）；
    - 结果碰上一次定时读，读完时已经进了下一个输入帧。

    这几种都列在下面"覆盖不到的"里。
- **具名脚本**：
  - `scenarios.TIMED` 的 7 个墙钟脚本：读状态 0.01 或 0.05 秒，采帧 0.02 秒；
  - `scenarios.ACROSS` 的 11 个墙钟脚本：截止落在落地前那次读、动作后那次读、采帧、执行里，执行跨过帧开头的定时读；其中 1 个是第 7 节的 PENDING；
  - K-rel 的 2 个：动作 0.125、0.2 秒；
  - 读快照有耗时的若干个。
- **规则成立的前提**：
  - 读一次比一帧短；
  - 没有回复恰好在一次读发问的同一刻回来；
  - 每个决策落地后等到下一帧。
- **截止或帧开头落在一次同步操作里**：外壳做完手上的读、采帧或执行，内核才处理截止或到期的定时读。规则照这个改写：
  - 截止落在落地前那次读里（答回在截止当刻或之后），旧循环记 `not_applied_tick_limit`，内核记 `not_applied_time_limit`（K-k）；
  - 截止落在采帧里，旧循环在截止后开始生成，内核不开始（K-a）；
  - 快照在那次操作返回时读（K-g）；动作后那次读在截止前发问的，内核先回报再冻结快照（X2）；
  - 执行跨过帧开头，那一帧的定时读由动作后那次读、或被拒时补的那次读来做，在执行返回时（K-b）。
- **覆盖不到的**：
  - 结果晚于帧边界才回来，或 wait 结束在帧中间：旧循环在循环开头多读一次（内核在已读过的帧里不读，KF5c），耗时会带进下一个决策。执行有耗时的话，这次读还可能先于内核读到任务结束：被拒的动作执行完已过结束时刻，两边都不在执行后读，内核要到下一帧开头才读到。没写规则，也没进具名脚本；随机脚本避开（种子 82、754）。
  - 结果碰上一次定时读，K-b 推迟的接纳跨进下一个输入帧：内核从那一帧起等下一帧，旧循环从回来的那一帧起，之后的决策整体错开。没写规则；随机脚本避开（种子 245）。
  - 动作执行跨过截止、读快照有耗时：疑似内核问题，见第 7 节；随机脚本避开。
  - 读一次比一帧长：60 fps 下读 0.05 秒。另外，墙钟替身在格点上取 tick 会差一个 ulp（`1024.05 - 1024` 乘 60 得 2.999…）。
  - 单步上限落在一次读状态期间（p1b 的 U9）：只写了前面的决策都没有被接纳的结果时的情形（`wall-the-per-move-cap-falls-during-a-read`），和 X6 一样；截止那一种是 D47。
  - 墙钟随机脚本的单步上限只放弃开头的几步，它们的回复改成挂住、客户端放弃时才放开（`generate._capped`）。回放桩一次只答一个请求（`tests/replay/stub.py`），放弃以后才回来的回复会拖住下一次请求，这是替身的局限，不是两边的差异。
  - 整理结果、accept、回报有耗时（KF7、KF10）：回放里这些都不花时间。
  - 生产环境 `GuestEnvironment` 的墙钟代码：墙钟回放用的是 `FakeEnvironment`，所以 X13 的 10 秒上限、按住键的租约在这里看不到，靠 `tests/sandbox` 守着。

## 5. 逐条

"出现"是指对跑里观察到，而且规则改写以后两边一致。随机一栏是种子 0–2015 里触发的个数，那是审查修复以前的扫描；这一轮种子 0–999 的计数见第 6 节。

| 编号 | 结果 | 触发它的具名脚本 | 随机 | 说明 |
|---|---|---|---|---|
| K-a | 出现 | `frame_rates-wall-*`、`runner-late_reply`、`runner-done_reply`、`runner-bounded_call_timeout`、`runner-long`、`wall-result-at-the-deadline`、`wall-retry-across-the-deadline`、`wall-reads-and-captures-take-time` | 313 | 截止放弃的生成，外壳合成一条 `timed_out` 的记录（第 48 条），和旧循环有界等待超时写的一样，逐字段比。旧循环在宽限期内等到了回复的，那条回复的记录由 K-a 换成"放弃的调用"。采帧跨过截止时，旧循环在截止以后还开始生成，内核不开始（`wall-a-capture-runs-past-the-deadline`）。旧循环把等这次调用的秒数计进思考预算（`_charge`），内核只计接纳的结果（`TurnPolicy._accept`），所以 summary 的 `think_spent_s` 少这一次；投影不比这个字段 |
| K-b | 出现 | `wall-ends-during-a-reply`、`wall-ends-while-pacing`、`wall-wait-cut-by-the-end`、`wall-a-reply-back-during-a-periodic-read`、`wall-ends-during-a-reply-and-reads-take-time`、`wall-a-wait-cut-by-the-end-and-reads-take-time`、`wall-the-per-move-cap-falls-during-a-read` | 120 | 按 N29 扩写：结果回来时一次定时读正在进行，就等它读完再接纳，再在新读数上读一次才落地。动作、帧龄、动作时刻比旧循环晚"读完时刻 − 回来时刻"，最多一次读的耗时。执行跨过帧开头时，那一帧的定时读推迟到执行返回，由动作后那次读（被拒时补的那次读）来做，所以不在帧开头找到结束（`wall-*-runs-into-the-next-frame-and-the-end`，两边相同） |
| K-c | 出现 | `wall-wait-cut-by-the-end`、`wall-a-wait-cut-by-the-end-and-reads-take-time` | 12 | 已并入 KF3 和 KF3 补充。wait 从交给环境的时刻算起，落地前那次读有耗时就晚一次读 |
| K-d | 出现 | `token-skipped-decision-after-the-end`、`token-think-budget-and-the-end` | 9 | |
| K-e | 出现 | `token-advance-fails-while-approaching`、`token-advance-fails-while-pacing` | 33 | |
| K-f | 出现 | `token-missing-usage` | 47 | |
| K-g | 出现 | 墙钟 18 个脚本，例如 `runner-late_reply`、`wall-ends-during-a-reply`、`wall-the-deadline-falls-during-a-read` | 385 | 快照在内核处理结束时读：读到结束或第三次读失败的那次读返回以后（结束记在发问时刻，N44）；截止落在一次读、采帧或执行里时，那次操作返回以后 |
| K-h | 出现 | `wall-frozen-stream`、`token-frozen-stream` | 19 | |
| K-i | 出现 | `wall-think-budget-and-the-end`、`wall-think-budget-at-the-deadline`、`token-think-budget-and-the-end` | 4 | 思考预算用完和任务结束、墙钟截止相撞时，先到的那个定停止原因。预算的结束两边都记在超预算那个决策回报时，在它等到下一帧之后；这段等待撞上截止时，内核记 `time_limit`，决策在手上（X2） |
| K-j | 出现 | `token-lease-ends-inside-a-tap` | 4 | |
| K-k | 出现 | `wall-the-read-before-landing-runs-past-the-deadline`、`wall-the-read-before-landing-answers-at-the-deadline` | — | 墙钟下总帧数在截止处用完。落地前那次读在截止前发问、在截止当刻或之后答回时，旧循环读前问截止（还没到）、读完数帧，记 `not_applied_tick_limit`；内核只问截止，由截止的定时器记 `not_applied_time_limit` |
| K-l | 不可观察 | — | — | 回放里采帧到开始生成之间没有时间流逝 |
| K-m | 被 K-a 吸收 | — | — | |
| K-n | 又出现了 | `wall-third-failed-read-is-todays-first`、`wall-a-failed-read-takes-time` | — | N26 以后墙钟落地前又读了；这次读失败，内核记 `invalid_run`，旧循环记 `not_applied_terminal`。随机脚本不生成墙钟读失败 |
| K-poll | 不可观察 | — | — | `FakeEnvironment` 本来就落在整 tick 上 |
| K-rel | 出现 | `wall-hold-pressed-after-the-guards`、`wall-a-hold-pressed-past-its-lease` | — | 租约从 Execute 算起；按下被耽搁的，至少再按住 min(时长, 1/60 秒)（第 38 条，N9） |
| KF1 | 作废 | — | — | |
| KF2 | 出现 | `token-horizon-a-few-ulps-past-a-tick` | — | 依据：kernel-fix 第一轮第 13 条让内核像环境时钟一样把 1e-9 内的时限吸附到整 tick，同一报告第 5 节列成 KF2；旧循环没改 |
| KF3 | 并入 K-c | `wall-wait-cut-by-the-end` | — | |
| KF4 | 出现 | `token-the-end-seen-inside-a-tap`、`token-no-read-after-the-end-seen-inside-a-tap` | 1 | |
| KF5（a、b、c、f） | 出现 | token 25 个脚本，例如 `frame_rates-token-60_1` | 1292 | p1c 记作 X18。U3 并进 KF5a（N5）。墙钟上只剩 KF5c（kernel-fix 第七轮 KF9） |
| KF5d、KF5e | 作废 | `wall-a-wait-lands-after-the-end`、`wall-a-refused-action-lands-after-the-end`、`runner-environment_done_during_reply` 现在两边相同 | — | N26 (b)：落地前在新读数上照读 |
| KF6、KF7、KF8 | 没有脚本 | — | — | KF6 要 accept 抛异常；KF7 要整理结果有耗时；KF8 下旧循环暂停模式会死循环 |
| KF10 | 不可观察 | — | — | N25：落地在 accept 之后的读数上；回放里 accept 不耗时，`landing` 定时记录是 KernelRecord |
| D47 | 出现 | `wall-the-deadline-falls-during-a-read` | — | 截止落在一次读期间，期间回来的结果在读完以后才整理，已过截止。外壳不回报这一步，记录是这次调用自己的，`frame_usage` 为 0，用量不计（第 37 条）。旧循环照常落地 |
| X1 | 出现 | `frame_rates-token-*`、`runner-done_reply`、`runner-long`、`wall-wait-across-the-deadline`、`wall-wait-cut-by-the-end`、`token-a-wait-whose-own-step-fails` | 1083 | |
| X2 | 出现 | 43 个，例如 `runner-environment_done`、`wall-hold-between-frames`；另有第 7 节那个 PENDING 脚本 | 801 | |
| X3 | 不再出现 | — | — | 外壳给每个放弃的生成都合成记录（第 48 条、N10），归 K-a、U10、U11 |
| X4 | 出现 | `frame_rates-wall-30_7`、`wall-per-move-cap`、`token-horizon-a-few-ulps-past-a-tick` 等 7 个 | 51 | |
| X5 | 归一化 | — | — | |
| X6 | 出现 | `wall-per-move-cap`、`wall-over-the-cap-and-the-think-budget`、`wall-the-per-move-cap-falls-during-a-read` | 见第 6 节 | 只写了开头几步被上限放弃、之前没有已接纳结果的情形：每一步的反馈开下一帧，请求只留系统消息和这一帧；第一个被接纳的帧，在它之后的请求里还带着这些反馈，它的回合保留在历史里（`RealtimeRuntime._retained`） |
| X7 | 出现 | `runner-done_reply` | 252 | |
| X8 | 不比 | — | — | agent 自己的事件日志 |
| X9 | 这次看不到 | — | — | P1d |
| X10 | 不进对跑 | — | — | live（R15） |
| X11、X14、X15、X16 | 两边相同 | — | — | 两条路径用同一个环境 |
| X12 | 出现 | token 7 个、墙钟 3 个（`wall-two-failed-reads-are-tolerated`、`wall-third-failed-read-during-a-reply`、`wall-a-successful-read-resets-the-count`） | 122 | 点按继续走完时，键也在点按结束处松开 |
| X13 | 不可观察 | — | — | 上限在 `GuestEnvironment.check_end` 里，墙钟回放的 `FakeEnvironment` 不设上限；`tests/sandbox/test_shell_environment.py` 守着 |
| X17 | 不比 | — | — | token 下 tick 从整数变成等值浮点（p1c 的编号）；投影按数值比 |
| N22 | 归一化 | `token-a-wait-whose-own-step-fails` | — | 见第 3 节 |
| U2 | 出现，在清单内（N5） | 12 个，例如 `runner-environment_failure`、`token-a-wait-whose-own-step-fails` | 157 | |
| U3 | 在清单内（N5），就是 KF5a | 见 KF5 | — | |
| U4、U5 | 去掉了（N6） | `token-a-wait-whose-own-step-fails`、`wall-a-failed-action-settles-after-the-snapshot-read`，按旧循环的结局断言，通过 | — | p1c `3f7fcb8` |
| U6 | 消掉了 | `wall-frozen-stream`、`token-frozen-stream` | — | p1b `e9d5a9d` |
| U7、U8、U12 | 没有脚本 | — | — | p1b 第四到六轮的编号，依次要这些情形，回放里都不会发生：轨迹行写失败、采帧期间松键失败、内核或读数抛错 |
| U9 | 出现一半 | `wall-the-per-move-cap-falls-during-a-read` | — | 结果在单步上限之前回来，但正赶上一次定时读、上限落在这次读里：外壳读完才整理，已过上限，内核放弃这次生成，记 `skipped_timeout`，按上限扣思考预算。结果的记录是这次调用自己的，用量哪里都不计（第 37 条），历史里也没有这次往来（X6）。lifecycle 的 `model_latency_s` 是结果自己的耗时、`harness_timeout_reason` 为空；结果晚于上限回来的，记上限和 `per_move_cap`。上限 1.5 秒时，1.4 秒回来的记 1.4 和空，更晚回来的记 1.5 和 `per_move_cap`。投影不比这两个字段。另一半"整理本身跨过上限"没有脚本：回放里整理不花时间 |
| U10 | 出现 | `wall-ends-during-a-reply`、`wall-ends-during-a-reply-and-reads-take-time` | 91 | 任务结束时放弃的生成，外壳合成"放弃的调用"记录（`stale`，`harness_abort_reason: task_end`，缺用量原因 `generation_aborted_at_task_end`，不进 token 统计）；旧循环那条是真实回复（N10、N34） |
| U11 | 出现 | `wall-third-failed-read-during-a-reply` | — | 同上，判无效时放弃（`invalid_run`）。随机脚本不生成墙钟读失败 |
| U13 | 不比 | — | — | p1b 第七轮：已整理、后被内核丢弃的结果，轨迹行类型是 `decision`。对跑不比轨迹行 |
| U14 | 已由 N44 消掉 | `wall-a-failed-read-takes-time` 现在两边的结束和快照时刻相同 | — | 读失败判无效，内核原来记在读返回时、旧循环记在发问时；N44 以后内核也记在发问时（`98a60cd`） |
| U17 | 出现 | 墙钟 15 个，例如 `runner-late_reply`、`wall-ends-during-a-reply`、`wall-third-failed-read-during-a-reply` | 未重扫 | 内核放弃的生成（截止、任务结束、判无效），回复不进已回报决策的 token 小计（`reported_decisions`，`TokenAccounting.leave_unreported`）；规则删掉的决策，内核没有开始。旧循环每个开始了的决策记一次回复，不写小计，小计就是总数。规则从旧的小计里减掉这些决策的回复和 `clock_output_tokens`。投影取 `summary()` 的 `reported_decisions`，没有就取整份，和 `reporting.py` 的读法一样。默认 25 个种子里也出现 |
| U18 | 没有脚本 | — | — | agent 的 accept、discard 抛错：外壳在自己的线程上跑它们，经 `_raised` 记成 `runner:run_exception`；旧循环在有界调用的工作线程上、`stream_generate`（legacy_turn.py）里跑（`accept`、`discard`），accept 那一半抛错记成 `provider:generation_error`，已被放弃的工作线程上抛的没人读，这一局有效。回放里 agent 不抛错（审查低 7） |

## 6. 连跑 20 次和随机扫描

在 VM 上跑默认集合 `tests/differential`，共 104 个：62 个具名脚本、25 个随机种子、4 个自测（缩减两个、签名一个、种子参数一个）、13 个旧循环环境的测试（`test_legacy_environment.py`）。

- 20 个进程同时跑：20 次都是 `104 passed`，每次 164.82–169.90 s。
- 20 次依次跑：20 次都是 `104 passed`，每次 145.95–156.12 s。

随机种子 0–2015（32 个进程）全部通过。各规则触发的次数：

> KF5 1292、X1 1083、X2 801、K-g 385、K-a 313、X7 252、U2 157、X12 122、K-b 120、U10 91、X4 51、K-f 47、K-e 33、K-h 19、K-c 12、K-d 9、K-j 4、K-i 4、KF4 1；15 个种子没有任何差异。

kernel-fix 审查修复这一轮：

- 默认集合 120 个：78 个具名脚本（其中 1 个 PENDING，xfail），其余同上。在 VM 上跑一次：`119 passed, 1 xfailed in 159.57s`。修复以前是 `104 passed`，151.89–158.55 s。
- 默认 25 个种子：本地修复以前 29.33 s，以后 29.38–31.83 s。
- 随机种子 0–999 在本地全部通过（`1004 passed in 1058.56s`）。335 个墙钟局里，抽到的耗时和限制保留下来的局数：读状态有耗时 36 个，执行有耗时 75 个，单步上限 25 个，思考预算 30 个。各规则触发的次数：

> KF5 634、X1 540、X2 408、U17 200、K-g 180、K-a 163、X7 112、U2 77、X12 64、X4 54、K-b 46、U10 37、K-f 30、K-i 19、K-e 18、X6 17、K-rel 15、K-h 8、K-c 4、K-d 3、KF4 1；8 个种子没有任何差异。

  K-i、X6、K-rel 是随机脚本第一次触发。

## 7. 清单外的差异

`b3288a2` 那次签字时一条不剩。这一轮随机脚本多抽了执行耗时，发现一处，疑似内核问题，没有写成规则：

- **动作执行跨过截止，读快照又有耗时**：回合制外壳交给内核的 `ActionExecuted` 不填 `finished_at`。截止的定时器先到期，内核先结束这一局、冻结快照，快照读完才归档这次执行的回答。所以 `ActionResolved` 的 `environment_tick` 和 outcome 的 `next_observation_tick` 取的是快照读完时的 tick，越过了时限。旧循环取的是执行返回时的 tick。
- 流式那一半的 `core.py` 已经按 `finished_at` 归档动作的回答（第 1 节开头）。
- 只影响记录，不影响结局和快照。具名脚本 `wall-an-action-runs-past-the-deadline-and-the-snapshot-takes-time`（0.2 秒的点击 1.9 秒落地，截止 2.0 秒，读快照 1.5 秒）记在 `scenarios.PENDING` 里：旧循环记 tick 2，内核记 3。随机脚本避开这种情形（`simulated.timed_apart`）。

- 前几次签字的清单外差异的去向：
  - U2、U3、U6 由 N5 收进清单；
  - U4、U5 由 p1c 按 N6 修了外壳；
  - 自编的 U7 在 `73ca9a3` 以后消失；
  - 自编的 U13 在 N26 (b) 以后消失；
  - U14 由 N44 消掉。
- U14 是什么：墙钟下读状态失败判无效时，内核原来把结束记在读返回的时刻，旧循环记在发问的时刻。读到任务结束时两边都记在发问时刻，所以那是内核自己前后不一致的一处。kernel-fix 第九到十一轮让 `Kernel.invalidate` 取那次读的发问时刻（`98a60cd`），具名脚本 `wall-a-failed-read-takes-time` 现在两边的结束和快照时刻相同，规则也删了。

另外要知道的：

- 墙钟回放替身 `WallEnvironment.done()` 原来先锁存读失败、再花读的时间，已经改成和旧循环的环境（`LegacyStateReads.done()`）一样读完才锁存。这只影响 `tests/differential`。
- 这一轮随机脚本抽到执行耗时、单步上限以后，替身又改了几处，都只影响 `tests/differential`：
  - `WallEnvironment.execute` 原来给 wait、DONE、FAIL 也花执行耗时，现在照生产的 `GuestEnvironment.execute` 不花；
  - 回放桩的回复表里，挂住的请求不再占一条回复（`stub.py ChatCompletionsStub._next`）；
  - 投影的 facts 多了 `think_exhausted`：读快照失败把停止原因改成 `invalid_run` 以后，规则据此认出这一局其实是思考预算先用完，不是撞上截止。
- `tests/differential/legacy_loop.py` 是主线原样搬来的，`ruff check` 报 43 处（行长等）。本分支没改它，lint 只查框架自己的文件。
- 生产 `GuestEnvironment.execute` 原来在 `honor_fail` 时把 DONE、FAIL 记成 `_done`，只有 `LegacyStateReads.done()` 读它。内核把 DONE、FAIL 当 wait 跑，不交给环境（`turns.py` `_accept`、`streaming.py` `_arrived`），旧循环也在 `_decide` 里先把它换成 wait，所以哪边都走不到这一段。它已经连同 `honor_fail`、`_done`、`_terminal_action` 和 `GuestClient` 里 WAIT、DONE、FAIL 的分支删掉，`LegacyStateReads.done()` 也不再读 `_done`。

## 8. 怎么跑

- 默认集合：`pytest tests/differential`。会话结束时打印每个脚本触发了哪些规则。具名脚本预期的差异没有出现，这个脚本就失败（随机脚本不要求）。
- 放大随机种子：
  - `DIFF_SEEDS=500` 跑种子 0–499，`DIFF_SEEDS=500:1000` 跑一段；
  - 写错或者得到空集合，`test_the_seeds_are_given_right` 会失败；
  - 失败的随机脚本会被贪心缩减，打印出最小的脚本，可以直接贴成具名脚本。缩减的方法：删任一决策、操作耗时清零、时间吸附到帧上；比的是规则解释不了的差异和外壳对模拟的差异，两处的位置都要相同。

## 9. 没覆盖的

- 操作耗时的一般情形，见第 4 节。
- X12 补充 b、补充 d。
- X6 只写了"开头几步被上限放弃、之前没有已接纳结果"的情形；墙钟随机脚本的单步上限也只放在开头几步（第 4 节）。
- 重试、墙钟读状态失败：随机脚本不生成，只有具名脚本。
- U18（accept、discard 抛错）：没有脚本。
- 动作执行跨过截止、读快照有耗时：只有 PENDING 的那个具名脚本（第 7 节）。
- 一次点按不止一步（60 fps 以上）、有按住的键时点按内部读失败：只有具名脚本，或者没有。
- 暂停模式下的 KF2、KF8：旧循环会死循环。
- 投影不比的部分：timeline、日志、工具记录、agent 的事件日志（X8）、轨迹里除上面四种以外的记录。
- token 账只比已回报决策的小计里的回复数和 `clock_output_tokens`（U17）。总数不比：任务结束、判无效时合成的记录，外壳的总数不算（N34），截止时合成的那条算（第 48 条），旧循环的总数算的是真实回复。其他计数和成本也不比。
- 默认 25 个种子只触发一部分规则；靠具名脚本和 2016 个种子的扫描补足。
