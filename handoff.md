# ResponseLab Handoff

## 当前任务

- task_id: `responselab-latest-hardening-20260811`
- 目标：把独立仓恢复到总仓最新已验证的稳定代码，并修复本轮审计仍存在的 GUI、I/O、DSP、内存和交付问题。
- 权威仓库：`/Users/mac/PycharmProjects/RinysProject/codex_projects/frequency_response_compensator`，GitHub `ZhenlongYou/response-lab`。
- 写入分支：`codex/responselab-latest-hardening-20260811`；合入目标：`main`；最终需保留 `project/response-lab`。
- 起点：`main@993903d6f16485600eab79bbf9cf231edca1d5c3`，产品 tree `e988d549c7076919a45b6d9af5f6dc64c2c5c420`。
- 迁移来源：父仓 `43d5303b3e402a85175991ed5b6e947cdb716c2d` 的项目 tree `d95481cce5b4516f4693b3775e2878ac47be84af`。
- status: reviewed_ready_for_project_branch

## 已经完成

- 已确认当前独立仓与 `origin/main` 一致，但拆分时遗漏父仓 `d7b4914..43d5303` 的 32 个稳定化提交；218 项测试不是此前 422 项稳定快照。
- 已真实复现：同窗高采样率切到低采样率失败、AG10 被裸 float32 静默误读、packed CSV 失败、合法大 RFFT 轴误拒、Nyquist 端点错误、无限增益与硬边缘振铃。
- 已确认单一 canonical worktree、仓库身份和 GitHub remote；交付编排写入 lane 已由本任务认领。
- 按用户最终确认，仅删除主窗口“数据输入”标题下的常驻 CSV 格式说明及其专用空白；眼图算法与显示未改。目标回归先失败再通过，完整 `tests/test_ui_workflow.py` 通过；真实离屏窗口截图位于 `/Users/mac/Desktop/test/ResponseLab_删除数据输入说明后.png`，标题到首卡片间距为 10 px。
- 按用户确认，影响频段的 `M=32` 改为可直接使用的真实默认值；用户修改后立即生效，换文件不再要求额外确认。已删除确认提示、隐藏确认 worker 与来源令牌，未改眼图算法。
- 复核此前易用性/原理结论后，新增响应诊断 CSV 按 4096 行分块可取消导出；GUI 会在成功、失败或取消伴随清理残留时显示需要人工检查的路径；启动新的影响分析时立即清除旧曲线/候选，防止同路径源文件改写或新任务失败后继续显示旧结果。
- 普通比较/补偿在文件或参数变化后会协作中断旧 worker；影响频段的自动频带建议、工作区准备、Vpp 周期 FFT、候选 IFFT 和眼图卷积已贯穿取消回调。算法数值和眼图绘制口径未改变。
- 增益上限、raised-cosine、跨 Fs Nyquist、内存预检与多种子稳定性等旧审查项已由当前实现覆盖，未重复修改。导出并发合同明确限定为普通实例与良性并发，不宣称抵抗同一操作系统账户下故意抢占随机临时名的恶意进程。
- 当前最终快照验证：全套 `503 passed, 1 warning`；Ruff、compileall、diff-check 通过；项目 `.venv` 自检与 GUI smoke、系统 `python3 main.py --gui-smoke-test` 均 PASS。真实离屏窗口确认 M 默认 32、无二次确认、无数据输入格式赘述。

## 当前状态或阻塞

- 完整稳定 tree 与本轮已确认修改已冻结并通过最终验收；本轮先保存到持久项目分支
  `project/response-lab`。`main` 只有在仓库交付门禁与平台验收均闭环后才能更新。
- 当前没有外部阻塞；朗视厂商格式仍需真实 reader/writer 和读回证据，不能由 Keysight AG10 验证替代。

## 下一步计划

1. Windows 真实机器的窗口、打包、文件锁/取消/导出仍需按平台清单验收；macOS 结果不能替代。
2. 平台验收和仓库交付门禁闭环后，再把持久项目分支合入并推送 `main`。

## 不要再踩的坑

- 不要用 `git pull` 或零散补丁掩盖“拆分选错源码树”；先恢复完整稳定化提交链。
- 不要仅按扩展名猜 BIN/CSV 格式；未知格式必须 fail-closed。
- 不要把自动相位带、无限增益、硬频带边缘或固定单种子当作可信工程默认；M 的界面默认值固定为 32 并直接生效，结果摘要继续供用户核对 Fs/Rs/M/UI。
- 不要把虚拟眼/Vpp 宣称为 CDR、BER、噪声/抖动或标准合规测量。
- 用户已全盘否定眼图门限/测量标注预览；不得据此修改产品。当前示例上下眼“少一段”已确认来自窄脉冲形状而非显示抽样，用户最终只授权删除数据输入说明文字。
- 不要用测试数量替代目标 RED、独立数值 oracle、真实 GUI、内存和 Windows 交付证据。

建议后续继续使用：`rinysproject-delivery-orchestrator`、`tdd`、`signal-processing-review`、`test-effectiveness-gate`、`reviewer-subagents-gate`、`delivery-acceptance-gate`、`github-code-handoff`。
