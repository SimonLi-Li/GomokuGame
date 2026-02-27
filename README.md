# 从零构建 AlphaZero 五子棋 AI：技术解析与代码详解

## 项目概述

本项目实现了一个完整的 **AlphaZero 风格的 13×13 五子棋 AI 系统**，涵盖了从深度学习模型设计、蒙特卡洛树搜索（MCTS）、自我对弈训练，到 Web 前端交互的全链路。用户可以通过浏览器与训练好的 AI 对弈，也可以观看两个 AI 之间的对决。

### 技术栈

| 层级 | 技术 | 用途 |
|------|------|------|
| 深度学习框架 | **PyTorch** | 策略价值网络的构建与训练 |
| 数值计算 | **NumPy** | 棋盘状态表示、概率计算 |
| Web 后端 | **Flask + Gunicorn** | HTTP API 服务、多线程并发处理 |
| Web 前端 | **原生 HTML/CSS/JavaScript** | 棋盘渲染、用户交互 |
| 硬件加速 | **MPS (Apple Silicon) / CUDA / CPU** | 模型推理与训练加速 |

### 项目结构

```
GomokuGame/
├── game.py          # 五子棋游戏逻辑
├── model.py         # AlphaZero 策略价值神经网络
├── mcts.py          # 蒙特卡洛树搜索算法
├── train.py         # 自我对弈训练脚本
├── hello.py         # Flask Web 服务端
├── templates/
│   └── index.html   # 前端页面（棋盘 UI）
├── checkpoints_gomoku_13x13/  # 模型权重存档
├── requirements.txt # Python 依赖
└── venv/            # 虚拟环境
```

---

## 一、游戏引擎 —— `game.py`

### 1.1 核心设计

`GomokuGame` 类封装了 13×13 五子棋的完整游戏逻辑。棋盘用一个 NumPy 二维数组表示，其中 `0` 代表空位，`1` 代表黑子（先手），`-1` 代表白子（后手）。

```python
class GomokuGame:
    def __init__(self, board_size=13, n_in_row=5, max_pieces_per_player=0, max_moves=200):
        self.board = np.zeros((board_size, board_size), dtype=np.int8)
        self.current_player = 1  # 黑子先手
```

### 1.2 落子与规则判定

`make_move(x, y)` 方法处理落子逻辑：

1. **合法性校验**：检查越界、重复落子、游戏是否已结束
2. **棋子数量管理**：支持可选的棋子数量上限（使用 `deque` 队列实现 FIFO 淘汰机制——超出上限时自动移除最早的棋子）
3. **胜负判定**：调用 `_check_winner()` 检查四个方向（横、纵、主对角线、副对角线）是否有五子连珠
4. **平局检测**：棋盘满或达到最大回合数（200手）

```python
def _check_winner(self, x, y):
    directions = [
        [(0, 1), (0, -1)],   # 横向
        [(1, 0), (-1, 0)],   # 纵向
        [(1, 1), (-1, -1)],  # 主对角线
        [(1, -1), (-1, 1)]   # 副对角线
    ]
    for direction in directions:
        count = 1
        for dx, dy in direction:
            nx, ny = x + dx, y + dy
            while 0 <= nx < self.board_size and 0 <= ny < self.board_size and self.board[nx, ny] == player:
                count += 1
                nx += dx
                ny += dy
        if count >= self.n_in_row:
            return True
    return False
```

从落子位置出发，沿每个方向的正反两端延伸计数，只要总数 >= 5 即判定获胜。

### 1.3 神经网络输入编码

`get_state_for_network()` 将棋盘状态编码为 4 通道的张量，供神经网络消费：

| 通道 | 含义 |
|------|------|
| 0 | 当前玩家的棋子位置（二值矩阵） |
| 1 | 对手的棋子位置（二值矩阵） |
| 2 | 最后落子位置（单点标记） |
| 3 | 当前玩家标识（全 1 表示黑棋，全 0 表示白棋） |

这种编码方式使网络始终从"当前玩家"的视角看棋盘，简化了学习任务。

---

## 二、策略价值网络 —— `model.py`

### 2.1 网络架构

本项目采用 AlphaZero 的经典**双头网络**架构：共享的残差网络主干 + 策略头 + 价值头。

```
输入 (4, 13, 13)
    │
    ▼
初始卷积层 (4 → 128 通道, 3×3)
    │
    ▼
残差塔 (5 个 ResidualBlock)
    │
    ├──→ 策略头 → softmax → 落子概率分布 (169 维)
    │
    └──→ 价值头 → tanh → 局面胜率评估 [-1, 1]
```

### 2.2 残差块 (ResidualBlock)

```python
class ResidualBlock(nn.Module):
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))  # Conv → BN → ReLU
        out = self.bn2(self.conv2(out))          # Conv → BN
        out += residual                           # 残差连接
        out = F.relu(out)
        return out
```

残差连接解决了深层网络的梯度消失问题，使信息可以跨层直接传递。每个残差块包含两个 3×3 卷积层，通道数保持 128 不变。

### 2.3 策略头与价值头

**策略头**（Policy Head）输出每个棋盘位置的落子概率：
```python
# 1×1 卷积降维 (128 → 16) → 全连接 → 169 维 logits
self.policy_conv = nn.Conv2d(num_channels, 16, kernel_size=1)
self.policy_fc = nn.Linear(16 * 13 * 13, 13 * 13)
```

**价值头**（Value Head）输出当前局面的胜率评估：
```python
# 1×1 卷积降维 (128 → 16) → 全连接 (2704 → 64 → 1) → tanh
self.value_fc1 = nn.Linear(16 * 13 * 13, 64)
self.value_fc2 = nn.Linear(64, 1)
# 输出经过 tanh 激活，范围 [-1, 1]（-1=必败, 0=平局, 1=必胜）
```

### 2.4 损失函数 (AlphaZeroLoss)

总损失由三部分组成：

$$L = \underbrace{(z - v)^2}_{\text{价值损失 (MSE)}} - \underbrace{\pi^T \log p}_{\text{策略损失 (交叉熵)}} + \underbrace{c \|\theta\|^2}_{\text{L2 正则化}}$$

- **价值损失**：预测胜率 $v$ 与真实对局结果 $z$ 的均方误差
- **策略损失**：MCTS 搜索策略 $\pi$ 与网络输出策略 $p$ 的交叉熵
- **L2 正则化**：防止过拟合，权重衰减系数 $c = 10^{-4}$

---

## 三、蒙特卡洛树搜索 —— `mcts.py`

MCTS 是 AlphaZero 的核心搜索算法，它利用神经网络指导搜索，在有限时间内找到接近最优的落子。

### 3.1 搜索树节点 (MCTSNode)

每个节点存储四个关键统计量：

| 属性 | 含义 |
|------|------|
| `N` | 访问次数 |
| `W` | 累积价值 |
| `Q` | 平均价值 (`W/N`) |
| `P` | 先验概率（来自神经网络） |

### 3.2 MCTS 搜索的三个阶段

每次模拟包含三个阶段：

#### 阶段 1：选择 (Selection)

从根节点开始，沿着 **UCB 值最大**的子节点路径向下遍历，直到到达叶子节点：

$$UCB(s, a) = Q(s, a) + c_{puct} \cdot P(s, a) \cdot \frac{\sqrt{N(s)}}{1 + N(s, a)}$$

- **Q 项**（利用）：倾向于选择平均价值高的动作
- **P·√N/(1+N) 项**（探索）：倾向于选择先验概率高但访问次数少的动作
- `c_puct = 0.8`：控制探索与利用的平衡

#### 阶段 2：扩展 (Expansion)

到达叶子节点后，用神经网络评估该局面：
- 获取策略概率分布 `policy_probs` 和局面价值 `value`
- 过滤非法动作，重新归一化概率
- 为所有合法动作创建子节点
- 在根节点添加 **Dirichlet 噪声**（$\alpha = 0.3$，混合比例 0.25），增加探索多样性

```python
if self.add_dirichlet_noise and node == self.root:
    noise = np.random.dirichlet([0.3] * len(action_priors))
    for i, move in enumerate(action_priors):
        action_priors[move] = 0.75 * action_priors[move] + 0.25 * noise[i]
```

#### 阶段 3：反向传播 (Backpropagation)

将叶子节点的价值沿路径回传，更新所有经过节点的统计信息。注意每层交替取负，因为对手的收益就是己方的损失：

```python
while node is not None:
    node.update(-value)
    value = -value
    node = node.parent
```

### 3.3 动作选择与温度参数

完成所有模拟后（默认 400 次），根据根节点子节点的访问次数生成概率分布：

- **温度 T = 1**（训练时）：按访问次数比例采样，保持探索性
- **温度 T → 0**（对弈时）：贪心选择访问次数最多的动作

概率计算使用 **log-space 技巧**避免数值溢出：

```python
log_visits = np.log(visits + 1e-10)
scaled_log_visits = log_visits / temperature
max_log = np.max(scaled_log_visits)
scaled_log_visits -= max_log         # 减去最大值避免 exp 溢出
visits_temp = np.exp(scaled_log_visits)
probs = visits_temp / np.sum(visits_temp)
```

### 3.4 搜索树复用

`update_with_move()` 方法在每次落子后将对应子节点提升为新的根节点，避免重复搜索已探索的部分，显著提升搜索效率。

---

## 四、训练流程 —— `train.py`

### 4.1 AlphaZero 训练循环

`AlphaZeroTrainer` 实现了完整的自我对弈训练管线，每次迭代包含三个步骤：

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐
│  1. 自我对弈  │ ──→ │  2. 训练网络   │ ──→ │  3. 保存模型   │
│  收集数据     │     │  梯度更新      │     │  定期存档      │
└─────────────┘     └──────────────┘     └──────────────┘
       │                                          │
       └──────── 循环 1000 次迭代 ────────────────┘
```

### 4.2 自我对弈 (Self-Play)

AI 与自己对弈，同时扮演黑白双方。每一步都通过 MCTS 搜索决定落子，并记录三元组 `(棋盘状态, MCTS策略, 最终胜负)`：

```python
def self_play_game(self, mcts_simulations, temperature=1.0):
    while not game.game_over:
        state = game.get_state_for_network()
        actions, probs = mcts.get_action_probs(game, temperature=temperature)
        # 记录训练数据
        states.append(state)
        mcts_probs.append(prob_map)
        current_players.append(game.current_player)
        # 按概率采样落子
        action_idx = np.random.choice(len(actions), p=probs)
        game.make_move(action[0], action[1])
```

温度参数随训练进度动态调整：
- 迭代 1-10：T = 1.0（充分探索）
- 迭代 10-30：T = 0.8
- 迭代 30+：T = 0.5（逐渐减少随机性）

### 4.3 数据增强

利用棋盘的旋转和翻转对称性，将每条数据扩展为 **8 条**（4 种旋转 × 2 种翻转），有效增加训练数据量：

```python
# 8 种对称变换：原始 + 旋转90° + 旋转180° + 旋转270°
#              + 水平翻转 + 翻转后旋转90° + 翻转后旋转180° + 翻转后旋转270°
```

### 4.4 网络训练

从经验回放缓冲区（容量 50000）中随机采样 batch，进行梯度更新：

- **优化器**：Adam（学习率 0.001，L2 正则化 10⁻⁴）
- **学习率调度**：StepLR，每 20 个迭代衰减为 0.9 倍
- **梯度裁剪**：max_norm = 1.0，防止梯度爆炸
- **每次迭代训练 5 个 epoch**，batch_size = 256

```python
# 策略损失：交叉熵
log_probs = F.log_softmax(policy_logits, dim=1)
policy_loss = -torch.mean(torch.sum(batch_probs * log_probs, dim=1))

# 价值损失：MSE
value_loss = F.mse_loss(value.squeeze(), batch_winners)

# 总损失
loss = policy_loss + value_loss
```

### 4.5 训练配置一览

| 参数 | 值 | 说明 |
|------|-----|------|
| board_size | 13 | 棋盘大小 |
| num_channels | 128 | 卷积通道数 |
| num_res_blocks | 5 | 残差块数量 |
| mcts_simulations | 400 | 每步 MCTS 模拟次数 |
| c_puct | 0.8 | 探索常数 |
| n_iterations | 1000 | 总训练迭代次数 |
| n_games_per_iteration | 20 | 每迭代自我对弈局数 |
| batch_size | 256 | 训练批次大小 |
| buffer_size | 50000 | 经验回放缓冲区容量 |
| learning_rate | 0.001 | 初始学习率 |

### 4.6 断点续训

训练支持从任意 checkpoint 恢复。保存时将 replay_buffer 中的 NumPy 数组转为 Python 原生列表，确保兼容 PyTorch 2.6+ 的安全加载机制：

```python
# 保存时转换
buffer_data.append((
    state.tolist(),   # np.ndarray → list
    prob.tolist(),
    float(winner)     # np.float → float
))

# 加载时恢复
self.replay_buffer.append((
    np.array(state),  # list → np.ndarray
    np.array(prob),
    winner
))
```

---

## 五、Web 服务端 —— `hello.py`

### 5.1 架构设计

后端使用 **Flask + Gunicorn** 构建，采用单 Worker 多线程模式（gthread），避免模型在多进程间重复加载。

关键设计决策：
- **延迟加载模型**：在首次请求时才初始化模型和设备，避免 Gunicorn fork 后 MPS 设备初始化问题
- **全局共享模型**：所有会话共用一个 `PolicyValueNet` 实例，减少内存占用
- **独立 MCTS 实例**：每个会话拥有独立的 MCTS 搜索树，保证状态隔离

### 5.2 会话管理

使用 `OrderedDict` 实现 LRU 风格的会话存储：

```python
class GameSession:
    def __init__(self, session_id, n_simulations=400, game_mode='human_vs_ai', ai2_simulations=400):
        self.game = GomokuGame(BOARD_SIZE, N_IN_ROW, MAX_PIECES, MAX_MOVES)
        self.mcts1 = MCTS(...)  # AI1（黑棋）
        self.mcts2 = MCTS(...)  # AI2（白棋，仅 AI vs AI 模式）
        self.move_lock = Lock() # 线程安全锁
```

- 会话超时时间：30 分钟
- 最大并发会话数：1000
- 定期自动清理过期会话

### 5.3 API 端点

| 端点 | 方法 | 功能 |
|------|------|------|
| `/` | GET | 返回前端页面 |
| `/new_game` | POST | 创建新游戏会话 |
| `/make_move` | POST | 玩家落子 + AI 回应 |
| `/ai_vs_ai_step` | POST | AI vs AI 单步推进 |

#### `/make_move` 流程

```
玩家落子 → 合法性检查 → 更新棋盘 → 同步所有MCTS树
    → 检查玩家是否获胜 → AI思考(MCTS搜索)
    → AI落子 → 检查AI是否获胜 → 返回完整状态
```

#### AI 落子安全机制

`get_ai_move()` 中实现了多层防护：

1. MCTS 搜索后过滤非法动作（防止树复用产生的脏数据）
2. 重新归一化概率后选择最优动作
3. 若所有 MCTS 动作均非法，从合法动作中随机选取
4. 若 `make_move` 仍然失败，执行最终兜底策略

### 5.4 设备适配

```python
def _get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')        # NVIDIA GPU
    elif torch.backends.mps.is_available():
        device = torch.device('mps')          # Apple Silicon
    else:
        device = torch.device('cpu')          # CPU 回退
```

macOS 上启动 Gunicorn 时设置 `OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES` 环境变量，确保 fork 后的 Worker 进程能正常使用 MPS 加速。

---

## 六、前端界面 —— `templates/index.html`

### 6.1 概览

前端是一个单文件 HTML 应用，包含 CSS 样式和 JavaScript 逻辑，实现了完整的棋盘交互界面。

### 6.2 主要功能

- **Canvas 棋盘渲染**：使用 HTML5 Canvas 绘制 13×13 棋盘，支持触摸和鼠标操作
- **游戏模式切换**：支持 人类 vs AI 和 AI vs AI 两种模式
- **AI 难度调节**：通过滑块调整 MCTS 模拟次数（50-1600 次），模拟次数越多 AI 越强
- **执棋选择**：玩家可选择执黑（先手）或执白（后手）
- **实时状态显示**：展示 AI 胜率评估、回合数、AI 思考状态
- **获胜动画**：五子连珠时高亮显示获胜棋子连线
- **响应式设计**：适配桌面和移动端屏幕

### 6.3 前后端通信

前端通过 `fetch` API 与后端通信，每次请求携带 `client_session_id`（基于 UUID 生成），确保多用户状态隔离：

```javascript
// 玩家落子
fetch('/make_move', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
        client_session_id: sessionId,
        x: row,
        y: col
    })
})
```

每次 AI 回应后，前端都会用服务端返回的 `board_state` 完整同步棋盘状态，确保前后端一致性。

---

## 七、AlphaZero 算法核心思想总结

AlphaZero 的核心创新在于将深度神经网络与 MCTS 完美结合，形成一个自我增强的闭环：

```
     ┌─────────────────────────────────────────┐
     │                                         │
     ▼                                         │
  神经网络          指导搜索                     │
 (策略+价值)  ───────────→  MCTS 搜索            │
     ▲                        │                │
     │                        │ 生成训练数据      │
     │                        ▼                │
     └──────── 训练 ──────── 自我对弈数据 ────────┘
```

1. **神经网络**提供落子先验概率和局面评估，指导 MCTS 搜索方向
2. **MCTS** 通过大量模拟得到比神经网络更精确的策略，作为训练目标
3. **自我对弈**不断产生新的训练数据，网络持续进化
4. 整个过程不需要任何人类棋谱，完全从零开始自我学习

这种"搜索增强学习"的范式，使得 AI 能够在不依赖人类知识的情况下，发现超越人类的策略。

---

## 八、如何运行

### 环境准备

```bash
cd GomokuGame
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt  # flask, torch, numpy, gunicorn
```

### 训练模型

```bash
python train.py --iterations 1000 --games_per_iter 20 --mcts_sims 400
```

训练支持断点续训，默认从 `latest_checkpoint.pth` 恢复。

### 启动 Web 服务

```bash
python hello.py --port 8080 --threads 4
```

浏览器访问 `http://localhost:8080` 即可开始对弈。
