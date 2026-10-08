# Cosmos Task21：5090 部署开发参考

可复用的是 `backend=training_bundle` 路径，已在 4090 上完成真实 checkpoint 推理及仿真闭环联调。5090 尚未实测。部署代码随本仓库提交；新机器拉取同一分支后按下文准备外部模型文件。

## 代码入口

| 文件 | 用途 |
| --- | --- |
| `policy/Cosmos/bundle_policy.py` | 实际使用的模型封装、当前状态输入、推理计时与显存统计 |
| `policy/Cosmos/contract.py` | 三视角、52 维关节状态及按名字映射的输入约束 |
| `policy/Cosmos/deploy_policy.py` | `get_model` 的 training_bundle 分支及 `LocalSession`；其余 Robolab 路径是早期方案，不是本次 baseline 的验证路径 |
| `script/policy_model_server.py` | 模型进程入口 |
| `script/policy_rpc.py` | 同机 TCP 请求/返回动作块；无需云端 |
| `deployment/cosmos/client_example.py` | 独立客户端示例，读真实观测 NPZ、请求动作、保存 NPY，不驱动机器人 |
| `tools/prepare_cosmos_bundle.py` | 在目标机器重新生成绝对路径配置 |
| `tools/check_cosmos_flash_attention.py` | 在目标 GPU 上核对实际 Flash2 四类算子的数值 |
| `tools/check_cosmos_local_inference.py` | 从真实三视角 HDF5 检查输入或执行模型推理 |
| `run_policy.py`、`tools/run_cosmos_local_sim.sh` | 仿真客户端；运行时需另行准备 IsaacLab 环境及场景资产 |

外部训练配套源码 `source/cosmos_framework/inference/robot_policy/bench2dex.py` 实现实际 batch 构造和关节映射；`adapters.py` 负责模型加载和采样。仓库保存 `policy/Cosmos/bundle_compat.patch`，用于修补原始配套源码；本机源码已经修补，不要重复应用。这不是最新版 NVIDIA 官方部署实现。

## 固定数据接口

- `joint_names`：52 个唯一名字。`joint_action.vector`：对应顺序的当前实测关节角，float32、弧度。输入按名字转到训练顺序，输出转回请求顺序。
- `observation.cam_overhead.rgb`、`cam_wrist_left.rgb`、`cam_wrist_right.rgb`：每路 uint8 `[480,640,3]` RGB；OpenCV BGR 必须转 RGB。应取同一时刻的观测。
- 三视角拼成宽 640、高 720；训练预处理后视频张量为 `[3,33,640,640]`。仅首帧填观测，未来帧占位。
- 动作首行是当前实测 qpos 条件；32 个未来目标为 **52 维绝对关节角**，不是 delta、速度或末端位姿。模型内部 pad 到 64 维不改变机器人动作维度。
- 此 baseline 训练、推理均不归一化；不要额外反归一化。模型适配器已丢弃首行状态，客户端不要再删一行。
- 20 Hz、完整预测 32 步；配置 `execute_steps` 决定返回前几步，当前默认 4。`num_steps=4` 是扩散采样步数，含义不同。
- 实际任务文字来自 `$BUNDLE/metadata/manifest.json` 的 `task_text`；当前 backend 不支持通过每次请求的 `language` 切换任务。
- 默认 EMA 权重、BF16、不量化、关闭 compile；仍联合生成视频 latent，但不解码未来 RGB。该 baseline 不输入 PointFlow/FK。

## 搬到 5090

拉取仓库后，还需要已有的原始 `cosmos_task21_inference_bundle.tar` 中的 `model_assets/`（VAE、tokenizer 等）及训练 checkpoint 的 `model/` 目录。这些外部文件和 Python 环境不提交到 Git；原始配套 tar 同时提供匹配 checkpoint 的训练源码与配置。

原始配套 tar 的 SHA256：
`2e60255c64a7a24d36d94f5d344b18d0919f9db772badfe266ca3662b727e13b`。

将已有原始 tar 解压到 `outputs/cosmos_local/`，得到 `cosmos_task21_inference_bundle/`（包含 `source/`、`training/`、`metadata/`、`model_assets/` 和 `prepare_local.py`）。不要从本机复制 `checkpoint_compat/` 的绝对路径软链接；传原始 checkpoint，在目标机器按需转换元数据。

本机验证环境为 Python 3.11 / Torch 2.7.0+cu128 / FlashAttention 2.8.3。`.venvs/cosmos-local` 继承了 IsaacLab 环境，不能直接搬运。优先复用 5090 现有兼容环境；下述命令不会安装、升级或下载 Torch。4090 的 wheel 测试不能证明 5090 内核可用，先在目标机器跑下面的算子检查，再跑真实模型。

以下从本仓库根目录运行，`PY` 指向目标机器已经准备好的推理 Python，`CKPT` 指向原始 DCP `model/`：

```bash
export PY=/path/to/inference-env/bin/python
export CKPT=/path/to/iter_000020000/model
export BUNDLE="$PWD/outputs/cosmos_local/cosmos_task21_inference_bundle"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0

# 仅对新解压、未修补的原始源码执行以下两条；dry-run 失败先检查是否已修补。
patch --dry-run -d "$BUNDLE" -p1 < policy/Cosmos/bundle_compat.patch
patch -d "$BUNDLE" -p1 < policy/Cosmos/bundle_compat.patch
"$PY" "$BUNDLE/prepare_local.py"
"$PY" tools/prepare_cosmos_bundle.py --bundle "$BUNDLE" --checkpoint "$CKPT"
"$PY" tools/check_cosmos_flash_attention.py --source "$BUNDLE/source" \
  --output outputs/flash_attention_check.json
```

若加载旧 DCP 时出现 `pathlib._local` / `pathlib._abc` 元数据兼容错误，再执行以下转换并重新生成配置；不修改或复制权重张量：

```bash
"$PY" tools/prepare_cosmos_dcp_compat.py --source "$CKPT" --destination checkpoint_compat/model
"$PY" tools/prepare_cosmos_bundle.py --bundle "$BUNDLE" --checkpoint checkpoint_compat/model
```

配置生成在 `outputs/cosmos_local/deploy_baseline.json`，同目录另有 `baseline_contract.json`。先用真实三视角样本测试单次推理：

```bash
"$PY" tools/check_cosmos_local_inference.py --config outputs/cosmos_local/deploy_baseline.json \
  --hdf5 /path/to/observations.hdf5 --frame 1 --run-model \
  --output outputs/inference_check.json
```

HDF5 需要 `robot/joint_names`、`robot/qpos`、`meta/instruction` 和三路 `cameras/<name>/rgb` JPEG 数据集。首轮加载与稳态推理分开计时；多取真实观测检查动作幅度与连续性。

随后启动模型服务：

```bash
"$PY" script/policy_model_server.py --config outputs/cosmos_local/deploy_baseline.json --host 127.0.0.1 --port 9000
```

观测 NPZ 包含 `qpos`、Unicode 数组 `joint_names` 和以三个相机名命名的 RGB 数组，可用 `np.savez` 写入。在另一个终端执行：

```bash
"$PY" deployment/cosmos/client_example.py --observation /path/to/observation.npz \
  --contract outputs/cosmos_local/baseline_contract.json --manifest "$BUNDLE/metadata/manifest.json" \
  --output outputs/actions.npy
```

如果不需要进程隔离，可直接 `get_model(config)`、`model.reset(seed)`、`model.get_action(observation)`。此路径推理时无需启动仿真环境。

## 可选动作平滑

RPC 模型配置可以加入 `"action_smoothing": {"mode": "ema", "alpha": 0.3}`；省略此字段或使用 `mode: none` 保持原始输出。处理发生在 `LocalSession`，因此直接调用 `get_model()` 会绕过平滑；同进程需要平滑时使用 `LocalSession.get_action_chunk()`。

平滑作用于已经反归一化、按运行时关节顺序排列的绝对角度：每个控制步执行 `filtered = previous + alpha * (target - previous)`。首步以实际关节状态初始化，后续动作块沿用上个块最后的平滑指令，episode `reset()` 时清空历史。只处理实际返回执行的动作，不消费未执行的预测；调用方应完整执行返回动作块，中途丢弃后需重置 session。

在 20 Hz 下，`alpha=0.3` 的低频等效延迟约 0.117 秒。平滑可能降低抖动，也可能影响接触和抓取时机；它不代替速度/加速度限制。启用后 `<output_dir>/action_smoothing.jsonl` 同时保存原始和实际返回的指令，用于核对。实现与测试分别在 `policy/Cosmos/action_smoothing.py` 和 `tests/test_cosmos_action_smoothing.py`。

## 真机开发边界与已测结果

现有仿真会暂停物理推进，等模型返回再执行默认 4 步。真实机器人不会暂停时间，因此真机控制器需要独立的固定频率执行线程、带采集时刻的观测和动作块、对过期动作的处理，以及基于实际关节限制的速度/加速度约束和超时停止。当前代码没有实现这些，也没有机器人驱动。不要把模型输出直接当成已经可执行的真机控制策略。

当前 RPC 为简单同步 JSON/base64 TCP，共享一个模型 session，按单客户端使用；没有多机器人 session 隔离、认证或实时调度。它适合参考接口及同机联调。客户端示例不会自动执行动作。

4090 完整联调：1000 控制步、250 次查询，推理中位数约 1.145 秒；1 秒间隔采样的整卡显存峰值约 16.83 GiB（模型＋Isaac）。任务失败、0/3 放置阶段完成，动作存在抖动。因此已验证的是推理/仿真链路，不是策略质量。训练旧 RGB 与本地仿真 anchor 的场景外观不同，正式 benchmark 需对齐配置。本机 `outputs/cosmos_local/closed_loop_episode01/` 保留原始验证报告（不提交到 Git）；不能将这些数值当作 5090 测量。

下一步在 5090 先通过算子检查和单次推理，再测连续请求的延迟分布与动作质量，最后实现异步控制器。增大 `execute_steps` 可以延长单次动作覆盖时间，但必须评估更长开环执行对成功率和抖动的影响。

## 仿真当前帧 PointFlow＋FK 推理

新增后端 `pointfk_bundle`，保留 `training_bundle` baseline。使用包含 Point/FK 分支的训练源码及 checkpoint；当前适配的是 `sim_v3_94train_6val_handguided1024_stage2_8gpu_20k` 的 `iter_000003000`。源码来自补充的当前工作区快照，不能据此证明与训练启动时逐字节一致。

处理顺序：`run_policy.py --live-pointfk` 在动作队列为空时，从同一物理时刻的 overhead RGB、光轴深度、相机参数、关节状态和物体位姿生成 mask、当前点云和 42 个 FK 点。只有 overhead 开启深度，RGB 保留三个视角。mask 使用缓存的场景实际网格、实际 scale 和位姿，并与相机深度比较排除遮挡。1024 点预算采用训练源码相同的 global512＋左右手各256（5cm 范围，不足则全局补齐）选择；可见候选不足时保留实际点数，不复制凑数。

`policy/Cosmos/sim_geometry.py` 负责现场数据，`live_geometry.py` 负责几何，`anchor_selection.py` 是训练选择函数的独立副本；Isaac 无需导入完整 Cosmos。RPC 的 `current_geometry` 携带几何、帧号及关节快照，模型拒绝与动作 state 不同步的输入。几何计算只发生在新动作块查询时。

`policy/Cosmos/pointfk_policy.py` 将当前输入接入源码正式的 `generate_samples_from_batch`，联合采样 `[vision | action | pointflow | FK]`。未来几何标签只填零并将有效掩码设为 false；源码推理准备阶段还会清除监督标签。第一帧视频和当前归一化 state 为条件，未来四模态从噪声生成。动作由源码反归一化一次，去掉首行 state，按运行时关节顺序返回。默认预测32步、执行16步，4次去噪、CFG=3、shift=5；不启用动作平滑。

本路径不运行 CoTracker/Track4World 或 DA3。训练点来自跨帧轨迹在当前帧的观测；实时点来自当前深度网格，因此候选点分布仍有差异，不应称为与训练输入逐点相同。

准备独立运行目录（复用已有 baseline bundle 的 tokenizer/VAE 和 Bench2Dex 图像/动作适配器，不复制 checkpoint 张量）：

```bash
PY="$PWD/.venvs/cosmos-local/bin/python"
ASSETS=/home/fangkaipeng/dexmanip_scj/checkpoints/sim_ckpt/sim_v3_94train_6val_handguided1024_stage2_8gpu_20k
BUNDLE="$PWD/outputs/cosmos_local/cosmos_task21_inference_bundle"
WORK="$PWD/outputs/cosmos_local/pointfk_3000"
"$PY" tools/setup_cosmos_pointfk_runtime.py \
  --assets "$ASSETS" --baseline-bundle "$BUNDLE" --output "$WORK"

COSMOS_PYTHON="$PY" COSMOS_CONFIG="$WORK/policy.yaml" \
BASELINE_ANCHOR_HDF5="$PWD/outputs/sim_pointflow/task21_replay100/sources/episode_000000.hdf5" \
COSMOS_RUN_DIR="$WORK/live_episode_new" \
  bash tools/run_cosmos_local_sim.sh --live-pointfk
```

脚本启动并监管本机模型和仿真进程，模型退出时不继续等待；运行目录保存 `model.log`、`sim.log`、`gpu.csv`、`metrics/live_geometry.jsonl`、`metrics/first_live_geometry.npz` 及 `episodes/{success,failure}/*.hdf5`。模型耗时在 policy YAML 指定的 `output_dir/local_inference.jsonl`，同一配置多次运行会追加。导出录像：

```bash
"$PY" tools/render_cosmos_episode.py \
  --episode "$WORK/live_episode_new/episodes/failure/episode_000001.hdf5" \
  --output "$WORK/live_episode_new/episode_000001.mp4" --three-views --fps 20 --threads 2
```

本机新增依赖的 wheel 持久保存在 `.cache/sonata_wheels/`，含 SHA256SUMS；通过 `pip install --no-index --no-deps .cache/sonata_wheels/*.whl` 离线复用。已验证组合为 Python3.11、Torch2.7.0+cu128、FlashAttention2.8.3、spconv-cu126 2.3.8、cumm-cu126 0.7.11、torch-scatter 2.1.2+pt27cu128、timm1.0.25。换 Python/Torch/CUDA 时需重新检查二进制兼容性。

已通过 Sonata 严格权重加载与4090 GPU前向、真实观测四模态联合推理和64控制步闭环冒烟测试。同步仿真会等待推理完成再推进物理时间；这里的“当前帧处理”不等于已经满足真机20Hz实时控制。

2026-10-07 完整验证：`outputs/cosmos_local/pointfk_3000/live_episode_diagnostic/`，1000控制步、63次联合推理、无 episode error，50秒三视角视频为 `episode_000001.mp4`。预热后几何处理中位数157ms（其中mask33ms），模型调用中位数1.839秒；模型峰值分配8.40GiB，整卡采样峰值16.85GiB。几何计时不含相机渲染/RPC，不能把两项直接当完整端到端延迟。任务失败；这是链路验证，不是成功率结论。相关输入/遮挡/原baseline回归测试共31项通过。

首轮完整测试曾在36次查询后发生仿真停顿，无法响应SIGINT/SIGTERM；相同场景、种子的重跑完成了全程，根因仍未确定。`--live-pointfk` 已加入单个物理步超过60秒自动打印堆栈的诊断，不自动杀进程、不改变benchmark判定。原停顿日志保留在 `live_episode/`，不能将此问题标为已修复。

后续输入审查修正了 Point/FK 的视角提示词：使用训练 `SimPointFKSFTDataset` 中的 `Head on top; left wrist bottom-left; right wrist bottom-right.`，替换从旧 baseline 继承的描述。任务文本与训练数据一致；回归测试对比训练端完整 transform 生成的 JSON prompt，包括缓存视频的33帧时长。`live_episode_diagnostic` 仍是修改前的结果，不应当作修改后测评。

同时修复 `setup_cosmos_pointfk_runtime.py` 的重复执行问题：GNU patch 的 `--batch` 会自动猜测反向补丁，原来再次准备环境可能撤销已安装的兼容修改，使 Torch2.7 启动时报缺少 `torch.distributed.checkpoint.hf_storage`。现在前向检查和应用均使用 `--forward`，已安装状态只作反向 dry-run 验证，不修改文件；连续执行及不兼容源码拒绝测试均通过。无需升级 Torch。

后续两次复测保留在 `live_episode_prompt_aligned_retry` 和 `live_episode_prompt_trace`：前者在2208物理步出现 `ndarray.items` 异常（未定位），后者在54次推理后停在 mask 矩阵乘法，14:52:04 内核再次在CPU4出现同类BUG。现在 episode 异常会输出完整堆栈；启动脚本默认 `OPENBLAS_NUM_THREADS=1`、`MKL_NUM_THREADS=1`（可显式覆盖），模型服务和端口检查允许复用已关闭连接的地址，仍拒绝占用中的监听端口。

针对本机CPU4/5反复内核异常的局部规避复测，可在上述启动命令的 `bash` 前加 `taskset -c 0-3,6-31`，并使用新的 `COSMOS_RUN_DIR`。这只限制该次运行的CPU亲和性，不修改系统，不应复制CPU编号到其他机器，也不代表内核／硬件根因已修复。该次记录在 `live_episode_prompt_cpu_guard`。

`live_episode_prompt_cpu_guard` 已完整结束：None、seed102100000、3000物理步／1000控制步、63次联合推理，episode error为空，任务失败、LSCR=0。单条通过只验证链路可完成，不证明间歇性故障已根治。

Point/FK 后端现在把每次正式联合预测保存为未压缩NPZ，不在线渲染。启动脚本默认保存到 `$COSMOS_RUN_DIR/predictions/episode_<index>_seed_<seed>/query_<index>.npz`；可用 `COSMOS_PREDICTION_DIR` 指定其他目录。文件含当前锚点、Point和FK各32步的米制位移、点ID、head内参、仿真时间和实际返回动作。源码采样器已完成位移反归一化，离线位置必须是 `anchor_xyz + displacement[t]`，不能再乘scale或累加。每次查询独立选点，点ID不保证跨查询一致。

完成推理后，用现有Cosmos环境离线生成MP4（不启动模型／仿真，不用GPU）：

```bash
"$PY" tools/render_cosmos_pointfk_predictions.py \
  --episode "$RUN/episodes/failure/episode_000001.hdf5" \
  --predictions "$RUN/predictions/episode_000001_seed_102100000" \
  --output "$RUN/predicted_pointfk.mp4"
# 成功时将 failure 改为 success；只看一次完整32步预测，再加 --query-index 0。
```

默认视频左侧为实际head RGB，右侧为同一RGB上的最新一次Point/FK预测；按20Hz对齐，重规划时明确标记当前锚点／新查询。`--query-index` 固定一条预测并显示最多33帧（含条件帧），即使控制器之后已重规划也不替换此预测。背景不是模型生成视频；正深度和图像边界只用于投影过滤，不冒充遮挡预测或GT轨迹。录制stride不是1时必须传 `--record-stride`。完整32步始终保存在NPZ，即使闭环每次只执行16步。

保存预测的完整验证位于 `outputs/cosmos_local/pointfk_3000/live_episode_predictions`：None、seed102100000、63次查询、1000控制步，全部63份预测有效、无episode error；单条任务失败，latched stage completion为2/3（味精未完成）。`predicted_pointfk.mp4` 为1000帧／50秒完整叠加，`query_000010_full_prediction.mp4` 为第10号查询的条件帧＋未来32帧。已逐帧解码校验并人工检查首／中／末帧；预测文件约28MB，数值落盘中位数0.61ms，绘图和编码全部离线。此结果与前一条的阶段完成情况不同，不能据此把改善归因于记录功能或提示词修正。

默认20FPS MP4呈现的是仿真时间，**不包含现实中的推理等待时长**。启用录制后，`run_policy.py` 另外保存 `metrics/episode_000001_wall_timing.json`：同一单调时钟下的逐帧采集时刻、几何准备／RPC区间及控制循环结束时间。可将同一轮、stride=1的RGB或预测叠加视频离线转换为含等待的实际时间录像：

```bash
"$PY" tools/render_cosmos_walltime.py \
  --video "$RUN/predicted_pointfk.mp4" \
  --timing "$RUN/metrics/episode_000001_wall_timing.json" \
  --output "$RUN/predicted_pointfk_walltime.mp4"
```

等待期间保持最新采集帧，按实测时间戳而不是平均推理耗时重复画面；顶部显示实际经过时间、仿真时间和等待阶段。20FPS输出的时间量化误差小于一帧。范围是首张观测到控制循环结束，排除启动及最后文件编码；这是按观测时间重建的录像，不是桌面录屏。旧运行没有这些时间戳，不能精确恢复旧视频的全部等待。

### 异步预测32步，第16步预取

在远端策略的 policy YAML 中设 `execute_steps: 32`（必须返回完整32步），启动仿真时增加 `--async-inference`。例如复用上述环境和场景路径：

```bash
COSMOS_PYTHON="$PY" COSMOS_CONFIG="$ASYNC_POLICY_YAML" \
BASELINE_ANCHOR_HDF5="$PWD/outputs/sim_pointflow/task21_replay100/sources/episode_000000.hdf5" \
COSMOS_RUN_DIR="$WORK/live_episode_async_new" \
  taskset -c 0-3,6-31 bash tools/run_cosmos_local_sim.sh \
  --live-pointfk --async-inference --seed 100000000 --episode-steps 1000 --early-stop
```

`--seed` 是种子命名空间基值；task21 会增加2100000，上例实际episode seed为102100000。每次运行应使用独立 YAML `output_dir`，避免混合模型计时日志。

`policy/Cosmos/async_actions.py` 使用单个后台线程发送RPC。观测和Point/FK仍在仿真线程采集，提交前复制整个快照；推理期间继续执行动作队列，不并发发送 `update_obs`，避免同一个socket的请求/响应错配。下一episode重置前会等待工作线程结束。

时间对齐规则：首个观测在控制步0，获得32步预测；执行完16步后，在控制步16取得真实观测并请求下一次32步预测，继续执行原预测的后16步。到控制步32切换时，下一预测相对其观测已经过去16步，所以跳过前16个过期动作，执行其后16个动作，并立即用控制步32的真实观测预取下一次预测。稳定后通常每16步发起一次请求；这是预测32步、延迟接入，不是每批重新从动作0执行。结果提前返回时也保留到旧队列耗尽再接入。

若队列耗尽但结果未返回，当前仿真实现阻塞等待，物理时间暂停、当前目标不变；不是让真实机器人在等待期间继续自由演化。没有把32步训练预测扩展成64步，也没有实现延迟条件训练或动作前缀约束，因此新旧预测衔接仍可能不连续。异步不会自动保证真机20Hz或任务成功。

`metrics/episode_000001_wall_timing.json` 的 `async_inference` 为true，每次query记录 `observation_step`、几何准备起止、`rpc_end_sec`、`adopt_step`、`skipped_actions`、`retained_actions`、`queue_wait_sec`。后者是真正因队列空而阻塞的时间，不能把整个RPC时长都当作阻塞；首轮无动作可重叠，需要单独统计。常规20FPS录像仍按仿真时间播放。已有“最新预测”Point/FK叠加工具不区分异步结果的返回/接入时刻，不能用其叠加图判断当前实际执行的是哪次预测。

2026-10-07 异步完整测试：`outputs/cosmos_local/pointfk_3000/live_episode_async32_prefetch16/`，与 `live_episode_exec32_same_seed` 使用相同场景配置和实际seed102100000。1000控制步、62次查询、无运行错误；除首次外，每次RPC期间均采集到了后续16个控制帧，接入时均跳过16个过期动作。模型调用中位数2.538秒，空队列阻塞中位数0.746秒、P95 0.859秒，累计阻塞46.433秒（含首次）；完整控制循环168.745秒，与同步32步的169.721秒基本持平。同步32只有32次调用，异步更新更频繁且同卡渲染/推理竞争，因此不能把局部等待下降等同于总耗时大幅下降。两条任务均失败、阶段完成0/3；相关调度及录像时间轴测试10项通过。

### 只预处理真实首帧

本地 `BundlePolicy`（包括 Point+FK 后端）默认启用 `preprocess_observed_frame_only: true`。实现位于 `policy/Cosmos/observed_frame_batch.py`：只对当前真实首帧执行原来的三视角拼接、缩放和补边，然后建立同尺寸的零占位未来帧。最终仍是33帧输入、32步预测，原有尺寸元数据、归一化、提示词、序列计划和Point/FK映射保持一致。该优化用于单观测推理，不用于真实多帧历史或训练视频。

如需与原始 WorldAct `_build_batch` 对照，可在 policy YAML 设置 `preprocess_observed_frame_only: false`。实现保存在本仓库的适配层，无需手动修改导出的训练源码，既有部署配置默认生效。

`tests/test_cosmos_observed_frame_batch.py` 对比上游完整batch：真实录像第0/250/500/999帧，以及黑/白/随机图像；逐元素比较视频、动作、尺寸、提示词和序列计划等。与Point/FK输入契约测试合计13项通过。运行时须将配套 WorldAct source加入PYTHONPATH。

2026-10-07 成对实测保存在 `outputs/cosmos_local/pointfk_3000/preprocess_optimized_v3/comparison.json`：4090、Point+FK3000、同一真实观测、相同采样seed及Python/NumPy/Torch随机状态，预热后3组对照，完整策略调用中位数由2.104秒降至1.725秒（减少0.379秒，约18%）。包含归一化和Point/FK字段的实际模型输入batch逐元素完全相同。计时不含Isaac/RPC/现场几何准备。

输出可重复性限制：原版重复推理自身也有差异（本次最大动作差约0.042–0.060rad），新旧对照约0.044–0.048rad，不能声称模型输出逐位一致。仅重置采样seed不够；补齐所有常见RNG后仍有此现象，具体根因尚未定位，不应仅归因于Sonata随机排序。此优化已证明保持完整输入一致，但本次未重跑闭环任务成功率。

### 索引复用与Transformer编译试验（未默认开启）

2026-10-07 在首帧预处理优化已启用的前提下，以同一保存观测测试 `tools/try_cosmos_compile.py`。普通推理中位数1.784秒；请求内注意力分组索引缓存使8次构建变为2次构建＋6次复用，字段与原实现相同，3项回归测试通过，但整次推理1.789秒，无明显收益。这只是分组索引缓存，不是完整序列模板或文本KV复用；后两者仍受当前Point/FK状态刷新及local RoPE的memory限制，未强行打开。

对 `model.net.language_model.forward` 使用 `torch.compile(fullgraph=True, mode='reduce-overhead', dynamic=False)` 成功，热启动整次推理中位数1.451秒（约快19%），首次含编译101.606秒。编译28层生成MLP，热启动1.721秒，收益较小。没有编译整个策略、Sonata或仿真，未开启旧的 `pad_for_cuda_graphs` 开关。

`tools/check_cosmos_compiled_transformer.py` 捕获条件/无条件两套输入，对完全相同的Transformer输入做对照：原版重复输出逐元素相同；普通编译生成特征相对RMS差异约1.36–1.47%，文本特征约3.18–3.46%。增加 `--preserve-precision` 使用当前Torch的 `emulate_precision_casts=True` 和 `force_same_precision=True`，生成特征差异约1.18–1.25%，文本特征约3.42–4.79%，仍非数值等价。保留精度版本热启动整策略1.401秒（独立运行3次中位数），首次整策略107.175秒；未将此小幅差别认定为比普通编译更快的严格成对结论。

因此编译版本仍仅为实验候选，未接入默认策略。特征差异不是动作误差或任务失败率；尚未验证跨点簇形状的重编译开销和闭环成功率，不能据此承诺效果不变。实测目录：`outputs/cosmos_local/pointfk_3000/compile_probe_v1`、`compile_equal_input_v1`、`transformer_repeat_control`、`compile_precision_v1`。编译缓存持久保存在 `.cache/torchinductor_cosmos`、`.cache/triton_cosmos`，无需下载或升级Torch。

2026-10-07 编译版实际异步闭环测试：`live_episode_async_compiled_v1`。设置 `compile_transformer: true` 可在Point/FK后端选择上述保留精度转换的静态编译路径，默认仍为false。该轮相同场景/实际seed102100000，预测32、第16步预取；仅完成84/1000控制步，因 `FailOnRecompileLimitHit` 中止。5次返回调用耗时30.60、108.06、1.93、109.47、111.66秒，分别生成2/2/0/2/2个新图；第6次触及重编译上限8，最后guard为 `pack['max_sample_len'] == 4309`。唯一无重编译调用的空队列等待近0，但只有1个样本且仿真非实时，不能作为稳定延时或真机实时性结论。控制循环364.29秒；不是完整任务测评。当前静态编译配置不能直接默认用于闭环，须先解决变化的序列长度及数值/任务质量验证问题。


### 动态编译修复及完整闭环实测（2026-10-07）

静态编译中止的问题已通过动态形状处理解决。当前可选配置：

```yaml
preprocess_observed_frame_only: true
compile_transformer: true
compile_dynamic: true
compile_mode: default
execute_steps: 32
```

仿真启动时同时加 `--async-inference`，第16步预取。编译总开关默认仍为false；开启后默认dynamic=true、mode=default。`reduce-overhead` 在10个变化观测上虽然只生成1个FX图，但多数新观测仍约3.2秒；关闭CUDA Graph的default模式下，首轮编译85.18秒，余下9次1.37–1.47秒、中位数1.379秒，全部在编译路径、没有回退。此离线测量不含现场几何、RPC或Isaac渲染。

`policy/Cosmos/compiled_forward.py` 对已知编译器错误记录明确回退信息，并永久切回eager以完成当前调用，防止每帧重复编译失败。其他模型/输入/CUDA异常仍传播。每次调用在 `model/compile_calls.jsonl` 记录new_graphs、mode、compiled_active、fallback；回退不能算编译加速成功。相关编译回退与异步队列测试10项通过。

完整单场景验证：`outputs/cosmos_local/pointfk_3000/live_episode_async_dynamic_v1/`。Point+FK3000、None、同一episode_000000初始化、实际seed102100000，4次去噪，32步预测、16步预取。计划最多1000控制步；第638步达到stable_success而正常提前结束，3/3阶段完成，40次模型返回，无运行错误、无重编译、无回退，全程1个FX图。最后一条预测因成功结束未接入动作队列。完整三视角录像为该目录 `episode_000001.mp4`，31.9秒、20FPS。

热启动模型调用中位数1.860秒、P95 1.908秒；RPC中位数1.899秒；现场几何准备中位数0.098秒。相比旧异步版本模型调用中位数2.538秒，下降约27%。两条轨迹和结束长度不同，不能把整个episode用时直接当作加速比例。首轮模型调用含编译35.337秒、阻塞35.385秒；其余38次接入的空队列等待均小于0.003毫秒。控制循环127.144秒，首轮RPC完成后的控制时间91.453秒。这里仿真执行速度慢于实时，足以覆盖RPC耗时；近零空队列等待不能证明真机实时性。16步仅覆盖0.8秒，1.86秒模型调用仍不满足这一时间预算。单条任务成功也不能证明成功率提高或编译数值等价。

最终动态编译配置的相同输入数值对照：dynamic_compile_equal_input_v1。两套CFG输入的生成特征相对RMS差异1.25%–1.37%，文本特征3.02%–3.42%；索引完全相同，输出全部有限，但不是逐元素等价。该比例不是动作误差，也不是任务失败率。编译保持可选，不能因单条成功就宣称效果无损。
