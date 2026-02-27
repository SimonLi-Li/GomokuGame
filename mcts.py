# mcts.py - AlphaZero的MCTS搜索算法（修复数值溢出问题）
# 作用：利用神经网络指导搜索，找到最优落子位置

import numpy as np
import math
from game import GomokuGame

class MCTSNode:
    """
    MCTS搜索树节点
    存储：访问次数N、总价值W、先验概率P、子节点
    """
    
    def __init__(self, prior_prob, parent=None):
        """
        prior_prob: 先验概率（来自神经网络的策略输出）
        parent: 父节点
        """
        self.parent = parent
        self.children = {}  # 字典：动作 -> 子节点
        
        self.N = 0          # 访问次数
        self.W = 0          # 总价值（累积奖励）
        self.Q = 0          # 平均价值 Q = W/N
        self.P = prior_prob # 先验概率
    
    def select_child(self, c_puct):
        """
        选择UCB值最大的子节点
        UCB公式：Q + c_puct * P * sqrt(父节点N) / (1 + 子节点N)
        - Q：利用（exploitation）- 选择价值高的
        - P*sqrt...：探索（exploration）- 选择访问少的
        """
        best_value = -float('inf')
        best_action = None
        best_child = None
        
        for action, child in self.children.items():
            # UCB计算
            u_value = child.Q + c_puct * child.P * math.sqrt(self.N) / (1 + child.N)
            
            if u_value > best_value:
                best_value = u_value
                best_action = action
                best_child = child
        
        return best_action, best_child
    
    def expand(self, action_priors):
        """
        扩展节点：为所有合法动作创建子节点
        action_priors: 字典 {动作: 先验概率}
        """
        for action, prior in action_priors.items():
            if action not in self.children:
                self.children[action] = MCTSNode(prior, parent=self)
    
    def update(self, value):
        """
        反向传播更新节点统计信息
        value: 叶子节点的价值评估（来自神经网络或对局结果）
        """
        self.N += 1
        self.W += value
        self.Q = self.W / self.N
    
    def is_leaf(self):
        """判断是否为叶子节点（未扩展）"""
        return len(self.children) == 0
    
    def is_root(self):
        """判断是否为根节点"""
        return self.parent is None


class MCTS:
    """
    蒙特卡洛树搜索引擎
    AlphaZero核心：结合神经网络的MCTS
    """
    
    def __init__(self, policy_value_fn, c_puct=5.0, n_simulations=800, temperature=1.0, add_dirichlet_noise=True):
        """
        policy_value_fn: 策略价值函数（神经网络推理接口）
        c_puct: 探索常数（越大越倾向探索）
        n_simulations: 每次决策的模拟次数（越多越强，但越慢）
        temperature: 温度参数（控制策略的随机性，训练初期高，后期降至0）
        add_dirichlet_noise: 是否在根节点加入Dirichlet噪声（训练用True，对战用False）
        """
        self.policy_value_fn = policy_value_fn
        self.c_puct = c_puct
        self.n_simulations = n_simulations
        self.temperature = temperature
        self.add_dirichlet_noise = add_dirichlet_noise
        
        self.root = MCTSNode(prior_prob=1.0, parent=None)
    
    def simulate(self, game: GomokuGame):
        """
        执行一次MCTS模拟（搜索树遍历）
        1. 选择（Selection）：从根节点沿UCB最大路径下降到叶子节点
        2. 扩展（Expansion）：用神经网络评估叶子节点，创建子节点
        3. 反向传播（Backpropagation）：更新路径上所有节点的统计信息
        """
        node = self.root
        game_copy = game.copy()
        
        # 阶段1：选择 - 沿树下降到叶子节点
        while not node.is_leaf():
            action, node = node.select_child(self.c_puct)
            game_copy.make_move(action[0], action[1])
        
        # 检查游戏是否结束
        if game_copy.game_over:
            # 终局状态：使用真实胜负作为价值
            if game_copy.winner == 0:
                value = 0  # 平局
            else:
                # 胜者为1，负者为-1（需要从当前玩家视角）
                value = 1 if game_copy.winner == game_copy.current_player else -1
        else:
            # 阶段2：扩展 - 使用神经网络评估叶子节点
            state = game_copy.get_state_for_network()
            policy_probs, value = self.policy_value_fn(state)
            
            # 过滤非法动作，重新归一化概率
            legal_moves = game_copy.get_legal_moves()

            # 如果没有合法动作（棋盘已满或游戏实际已结束），直接返回
            if not legal_moves:
                value = 0  # 平局
                # 反向传播
                while node is not None:
                    node.update(-value)
                    value = -value
                    node = node.parent
                return

            action_priors = {}
            total_prob = 0
            
            for move in legal_moves:
                action_idx = move[0] * game_copy.board_size + move[1]
                action_priors[move] = policy_probs[action_idx]
                total_prob += policy_probs[action_idx]
            
            # 归一化（确保概率和为1）
            if total_prob > 1e-10:
                for move in action_priors:
                    action_priors[move] /= total_prob
            else:
                # 如果所有合法动作概率为0（网络输出异常），使用均匀分布
                uniform_prob = 1.0 / len(legal_moves)
                for move in action_priors:
                    action_priors[move] = uniform_prob
            
            # 添加Dirichlet噪声（增加探索性，仅在根节点）
            if self.add_dirichlet_noise and node == self.root:
                noise = np.random.dirichlet([0.3] * len(action_priors))
                for i, move in enumerate(action_priors):
                    action_priors[move] = 0.75 * action_priors[move] + 0.25 * noise[i]
            
            node.expand(action_priors)
        
        # 阶段3：反向传播 - 更新路径上所有节点
        while node is not None:
            # 从对手视角反转价值
            node.update(-value)
            value = -value
            node = node.parent
    
    def get_action_probs(self, game: GomokuGame, temperature=None):
        """
        执行n_simulations次模拟后，返回每个动作的访问概率
        temperature: 温度参数
            - T=1: 按访问次数比例选择（训练时用）
            - T→0: 选择访问次数最多的（对弈时用）
        """
        if temperature is None:
            temperature = self.temperature
        
        # 执行n_simulations次MCTS模拟
        for _ in range(self.n_simulations):
            self.simulate(game)
        
        # 统计根节点的子节点访问次数
        action_visits = [(action, child.N) for action, child in self.root.children.items()]
        
        if not action_visits:
            # 如果没有子节点（理论上不应该发生），返回随机合法动作
            legal_moves = game.get_legal_moves()
            if legal_moves:
                return legal_moves, np.ones(len(legal_moves)) / len(legal_moves)
            else:
                return [], np.array([])
        
        actions, visits = zip(*action_visits)
        visits = np.array(visits, dtype=np.float64)  # 使用float64提高精度
        
        # 根据温度参数计算概率分布（使用数值稳定的方法）
        if temperature < 1e-3:
            # 温度接近0：选择访问最多的（贪心）
            probs = np.zeros(len(visits))
            probs[np.argmax(visits)] = 1.0
        else:
            # 温度>0：按访问次数的温度缩放比例
            # 使用log-space计算避免数值溢出
            # log(prob) = (1/T) * log(visits) - log(sum(visits^(1/T)))
            
            # 添加小常数避免log(0)
            log_visits = np.log(visits + 1e-10)
            scaled_log_visits = log_visits / temperature
            
            # 减去最大值避免exp溢出
            max_log = np.max(scaled_log_visits)
            scaled_log_visits -= max_log
            
            # 计算指数
            visits_temp = np.exp(scaled_log_visits)
            
            # 归一化
            sum_visits = np.sum(visits_temp)
            if sum_visits > 1e-10:
                probs = visits_temp / sum_visits
            else:
                # 如果还是出现问题，使用均匀分布
                print(f"⚠️  概率计算异常，使用均匀分布")
                probs = np.ones(len(visits)) / len(visits)
        
        # 最终检查：确保概率有效
        if np.any(np.isnan(probs)) or np.any(np.isinf(probs)):
            print(f"⚠️  检测到无效概率（NaN/Inf），使用均匀分布")
            print(f"   visits: {visits}")
            print(f"   temperature: {temperature}")
            probs = np.ones(len(visits)) / len(visits)
        
        # 确保概率和为1（防止浮点误差）
        probs = probs / np.sum(probs)
        
        return actions, probs
    
    def update_with_move(self, last_move):
        """
        复用搜索树：将落子后的子节点提升为新的根节点
        作用：避免重复搜索已探索的部分
        """
        if last_move in self.root.children:
            self.root = self.root.children[last_move]
            self.root.parent = None
        else:
            # 如果该动作未被搜索过，创建新树
            self.root = MCTSNode(prior_prob=1.0, parent=None)
    
    def reset(self):
        """重置搜索树"""
        self.root = MCTSNode(prior_prob=1.0, parent=None)
