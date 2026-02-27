# train.py - AlphaZero训练脚本（13x13五子棋版本）
# 修复：Policy Loss计算 + 模型保存格式（兼容PyTorch 2.6+）+ 支持指定checkpoint继续训练

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import time
import argparse
from collections import deque
import pickle
from tqdm import tqdm

from game import GomokuGame
from model import PolicyValueNet
from mcts import MCTS


class AlphaZeroTrainer:
    """
    AlphaZero训练器（13x13五子棋版本）
    """
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('mps' if torch.backends.mps.is_available() 
                                   else 'cuda' if torch.cuda.is_available() 
                                   else 'cpu')
        
        print(f"🖥️  使用设备: {self.device}")
        
        # 初始化神经网络
        self.model = PolicyValueNet(
            board_size=config['board_size'],
            num_channels=config['num_channels'],
            num_res_blocks=config['num_res_blocks']
        ).to(self.device)
        
        # 优化器
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['l2_const']
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=config['lr_decay_steps'],
            gamma=config['lr_decay_rate']
        )
        
        # 经验回放缓冲区
        self.replay_buffer = deque(maxlen=config['buffer_size'])
        
        # 训练统计
        self.train_stats = {
            'iteration': 0,
            'policy_loss': [],
            'value_loss': [],
            'total_loss': [],
            'entropy': [],
            'win_rate': []
        }
        
        # 创建checkpoint目录
        os.makedirs(config['checkpoint_dir'], exist_ok=True)
    
    def policy_value_fn(self, state):
        """策略价值函数（用于MCTS）"""
        self.model.eval()
        return self.model.predict(state, self.device)
    
    def self_play_game(self, mcts_simulations, temperature=1.0):
        """
        自我对弈一局游戏
        返回：(states, mcts_probs, winners) 列表
        """
        game = GomokuGame(
            board_size=self.config['board_size'],
            n_in_row=self.config['n_in_row'],
            max_pieces_per_player=self.config['max_pieces'],
            max_moves=self.config['max_game_moves']
        )
        
        mcts = MCTS(
            policy_value_fn=self.policy_value_fn,
            c_puct=self.config['c_puct'],
            n_simulations=mcts_simulations,
            temperature=temperature,
            add_dirichlet_noise=True
        )
        
        states = []
        mcts_probs = []
        current_players = []
        
        while not game.game_over:
            # 获取当前状态
            state = game.get_state_for_network()
            
            # MCTS搜索
            actions, probs = mcts.get_action_probs(game, temperature=temperature)
            
            # 记录状态和概率分布
            states.append(state)
            
            # 将动作概率转换为棋盘概率分布
            prob_map = np.zeros(self.config['board_size'] ** 2)
            for action, prob in zip(actions, probs):
                idx = action[0] * self.config['board_size'] + action[1]
                prob_map[idx] = prob
            mcts_probs.append(prob_map)
            current_players.append(game.current_player)
            
            # 选择动作（带探索）
            action_idx = np.random.choice(len(actions), p=probs)
            action = actions[action_idx]
            
            # 执行动作
            game.make_move(action[0], action[1])
            mcts.update_with_move(action)
        
        # 游戏结束，计算每个位置的收益
        winners = np.zeros(len(current_players))
        if game.winner != 0:  # 有胜者
            for i, player in enumerate(current_players):
                winners[i] = 1.0 if player == game.winner else -1.0
        # 平局时winners保持全0
        
        return states, mcts_probs, winners
    
    def collect_self_play_data(self, n_games):
        """
        收集自我对弈数据
        """
        print(f"\n📊 开始收集 {n_games} 局自我对弈数据...")
        
        # 动态调整温度参数（前期探索，后期利用）
        iteration = self.train_stats['iteration']
        if iteration < 10:
            temp = 1.0
        elif iteration < 30:
            temp = 0.8
        else:
            temp = 0.5
        
        print(f"🌡️  当前温度参数: {temp}")
        
        for i in tqdm(range(n_games), desc="自我对弈"):
            states, mcts_probs, winners = self.self_play_game(
                mcts_simulations=self.config['mcts_simulations'],
                temperature=temp
            )
            
            # 数据增强：对于3x3棋盘，可以做旋转和翻转
            augmented_data = self.augment_data(states, mcts_probs, winners)
            self.replay_buffer.extend(augmented_data)
        
        print(f"✅ 数据收集完成，缓冲区大小: {len(self.replay_buffer)}")
    
    def augment_data(self, states, mcts_probs, winners):
        """
        数据增强：旋转和翻转
        对于正方形棋盘，可以生成8种对称形式
        """
        augmented = []
        board_size = self.config['board_size']
        
        for state, prob, winner in zip(states, mcts_probs, winners):
            prob_2d = prob.reshape(board_size, board_size)
            
            # 原始数据
            augmented.append((state, prob, winner))
            
            # 旋转90度
            state_rot90 = np.array([np.rot90(s) for s in state])
            prob_rot90 = np.rot90(prob_2d).flatten()
            augmented.append((state_rot90, prob_rot90, winner))
            
            # 旋转180度
            state_rot180 = np.array([np.rot90(s, 2) for s in state])
            prob_rot180 = np.rot90(prob_2d, 2).flatten()
            augmented.append((state_rot180, prob_rot180, winner))
            
            # 旋转270度
            state_rot270 = np.array([np.rot90(s, 3) for s in state])
            prob_rot270 = np.rot90(prob_2d, 3).flatten()
            augmented.append((state_rot270, prob_rot270, winner))
            
            # 水平翻转
            state_flip = np.array([np.fliplr(s) for s in state])
            prob_flip = np.fliplr(prob_2d).flatten()
            augmented.append((state_flip, prob_flip, winner))
            
            # 水平翻转 + 旋转90度
            state_flip_rot90 = np.array([np.rot90(np.fliplr(s)) for s in state])
            prob_flip_rot90 = np.rot90(np.fliplr(prob_2d)).flatten()
            augmented.append((state_flip_rot90, prob_flip_rot90, winner))
            
            # 水平翻转 + 旋转180度
            state_flip_rot180 = np.array([np.rot90(np.fliplr(s), 2) for s in state])
            prob_flip_rot180 = np.rot90(np.fliplr(prob_2d), 2).flatten()
            augmented.append((state_flip_rot180, prob_flip_rot180, winner))
            
            # 水平翻转 + 旋转270度
            state_flip_rot270 = np.array([np.rot90(np.fliplr(s), 3) for s in state])
            prob_flip_rot270 = np.rot90(np.fliplr(prob_2d), 3).flatten()
            augmented.append((state_flip_rot270, prob_flip_rot270, winner))
        
        return augmented
    
    def train_network(self):
        """
        训练神经网络（修复损失计算）
        """
        if len(self.replay_buffer) < self.config['batch_size']:
            print("⚠️  缓冲区数据不足，跳过训练")
            return
        
        print(f"\n🎯 开始训练神经网络...")
        self.model.train()
        
        total_policy_loss = 0
        total_value_loss = 0
        total_loss = 0
        total_entropy = 0
        n_batches = 0
        
        # 训练多个epoch
        for epoch in range(self.config['train_epochs']):
            # 随机采样batch
            indices = np.random.choice(
                len(self.replay_buffer),
                size=self.config['batch_size'],
                replace=False
            )
            
            batch_states = []
            batch_probs = []
            batch_winners = []
            
            for idx in indices:
                state, prob, winner = self.replay_buffer[idx]
                batch_states.append(state)
                batch_probs.append(prob)
                batch_winners.append(winner)
            
            # 转换为tensor
            batch_states = torch.FloatTensor(np.array(batch_states)).to(self.device)
            batch_probs = torch.FloatTensor(np.array(batch_probs)).to(self.device)
            batch_winners = torch.FloatTensor(np.array(batch_winners)).to(self.device)
            
            # 前向传播
            policy_logits, value = self.model(batch_states)
            
            # ===== 修复：使用正确的交叉熵损失 =====
            # Policy Loss: 使用交叉熵（KL散度）
            log_probs = F.log_softmax(policy_logits, dim=1)
            policy_loss = -torch.mean(torch.sum(batch_probs * log_probs, dim=1))
            
            # Value Loss: MSE
            value_loss = F.mse_loss(value.squeeze(), batch_winners)
            
            # 计算策略熵（衡量探索程度）
            probs = F.softmax(policy_logits, dim=1)
            entropy = -torch.mean(torch.sum(probs * log_probs, dim=1))
            
            # Total Loss
            loss = policy_loss + value_loss
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪（防止梯度爆炸）
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_loss += loss.item()
            total_entropy += entropy.item()
            n_batches += 1
        
        # 更新学习率
        self.scheduler.step()
        
        # 记录统计信息
        avg_policy_loss = total_policy_loss / n_batches
        avg_value_loss = total_value_loss / n_batches
        avg_total_loss = total_loss / n_batches
        avg_entropy = total_entropy / n_batches
        
        self.train_stats['policy_loss'].append(avg_policy_loss)
        self.train_stats['value_loss'].append(avg_value_loss)
        self.train_stats['total_loss'].append(avg_total_loss)
        self.train_stats['entropy'].append(avg_entropy)
        
        print(f"✅ 训练完成")
        print(f"   📊 Policy Loss: {avg_policy_loss:.4f}")
        print(f"   📊 Value Loss:  {avg_value_loss:.4f}")
        print(f"   📊 Total Loss:  {avg_total_loss:.4f}")
        print(f"   📊 Entropy:     {avg_entropy:.4f}")
    
    def save_checkpoint(self, filename='checkpoint.pth'):
        """
        保存模型和训练状态（兼容PyTorch 2.6+）
        只保存Python原生类型，避免numpy对象序列化问题
        """
        # 将replay_buffer转换为纯Python列表（避免numpy对象）
        buffer_data = []
        for state, prob, winner in list(self.replay_buffer):
            buffer_data.append((
                state.tolist() if isinstance(state, np.ndarray) else state,
                prob.tolist() if isinstance(prob, np.ndarray) else prob,
                float(winner)  # 确保winner是Python float
            ))
        
        checkpoint = {
            'iteration': int(self.train_stats['iteration']),
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_stats': {
                'iteration': int(self.train_stats['iteration']),
                'policy_loss': [float(x) for x in self.train_stats['policy_loss']],
                'value_loss': [float(x) for x in self.train_stats['value_loss']],
                'total_loss': [float(x) for x in self.train_stats['total_loss']],
                'entropy': [float(x) for x in self.train_stats['entropy']],
                'win_rate': [float(x) for x in self.train_stats['win_rate']]
            },
            'replay_buffer': buffer_data,
            'config': self.config  # 保存配置便于后续加载
        }
        
        path = os.path.join(self.config['checkpoint_dir'], filename)
        torch.save(checkpoint, path)
        print(f"💾 模型已保存到: {path}")
    
    def load_checkpoint(self, filename='checkpoint.pth'):
        """
        加载模型和训练状态（兼容PyTorch 2.6+）
        """
        path = os.path.join(self.config['checkpoint_dir'], filename)
        
        if not os.path.exists(path):
            print(f"⚠️  未找到checkpoint文件: {path}")
            return False
        
        try:
            # 允许加载包含numpy对象的旧版本checkpoint
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.train_stats = checkpoint['train_stats']
            
            # 恢复replay_buffer（将列表转回numpy数组）
            buffer_data = checkpoint['replay_buffer']
            self.replay_buffer.clear()
            for state, prob, winner in buffer_data:
                self.replay_buffer.append((
                    np.array(state),
                    np.array(prob),
                    winner
                ))
            
            print(f"✅ 已加载checkpoint: {path}")
            print(f"📊 当前迭代: {self.train_stats['iteration']}")
            print(f"📈 缓冲区恢复: {len(self.replay_buffer)} 条数据")
            
            # 显示历史训练信息
            if len(self.train_stats['policy_loss']) > 0:
                print(f"📉 最近Policy Loss: {self.train_stats['policy_loss'][-1]:.4f}")
                print(f"📉 最近Value Loss:  {self.train_stats['value_loss'][-1]:.4f}")
            
            return True
        except Exception as e:
            print(f"❌ 加载checkpoint失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def train(self):
        """主训练循环"""
        print("\n" + "="*60)
        print("🚀 AlphaZero训练开始（13x13五子棋）")
        print("="*60)
        
        start_time = time.time()
        start_iteration = self.train_stats['iteration']
        
        # 显示训练计划
        print(f"\n📅 训练计划:")
        print(f"   起始迭代: {start_iteration}")
        print(f"   目标迭代: {self.config['n_iterations']}")
        print(f"   剩余迭代: {self.config['n_iterations'] - start_iteration}")
        
        for iteration in range(start_iteration, self.config['n_iterations']):
            self.train_stats['iteration'] = iteration + 1
            
            print(f"\n{'='*60}")
            print(f"📍 迭代 {iteration + 1}/{self.config['n_iterations']}")
            print(f"{'='*60}")
            
            # 1. 收集自我对弈数据
            self.collect_self_play_data(self.config['n_games_per_iteration'])
            
            # 2. 训练网络
            self.train_network()
            
            # 3. 定期保存模型
            if (iteration + 1) % self.config['save_interval'] == 0:
                self.save_checkpoint(f'checkpoint_iter_{iteration + 1}.pth')
            
            # 4. 保存最新模型
            self.save_checkpoint('latest_checkpoint.pth')
            
            # 5. 显示进度
            elapsed = time.time() - start_time
            print(f"\n⏱️  已用时间: {elapsed / 60:.1f} 分钟")
            print(f"📈 缓冲区大小: {len(self.replay_buffer)}")
            print(f"📚 当前学习率: {self.scheduler.get_last_lr()[0]:.6f}")
            
            # 6. 显示最近的损失趋势
            if len(self.train_stats['policy_loss']) >= 5:
                recent_policy = self.train_stats['policy_loss'][-5:]
                recent_value = self.train_stats['value_loss'][-5:]
                print(f"📉 最近5次Policy Loss: {[f'{x:.3f}' for x in recent_policy]}")
                print(f"📉 最近5次Value Loss:  {[f'{x:.3f}' for x in recent_value]}")
        
        # 训练完成
        print("\n" + "="*60)
        print("🎉 训练完成！")
        print("="*60)
        
        # 保存最终模型
        self.save_checkpoint('final_model.pth')
        
        # 保存训练统计
        stats_path = os.path.join(self.config['checkpoint_dir'], 'train_stats.pkl')
        with open(stats_path, 'wb') as f:
            pickle.dump(self.train_stats, f)
        print(f"📊 训练统计已保存到: {stats_path}")
        
        # 绘制损失曲线（如果有matplotlib）
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # Policy Loss
            axes[0, 0].plot(self.train_stats['policy_loss'])
            axes[0, 0].set_title('Policy Loss')
            axes[0, 0].set_xlabel('Iteration')
            axes[0, 0].grid(True)
            
            # Value Loss
            axes[0, 1].plot(self.train_stats['value_loss'])
            axes[0, 1].set_title('Value Loss')
            axes[0, 1].set_xlabel('Iteration')
            axes[0, 1].grid(True)
            
            # Total Loss
            axes[1, 0].plot(self.train_stats['total_loss'])
            axes[1, 0].set_title('Total Loss')
            axes[1, 0].set_xlabel('Iteration')
            axes[1, 0].grid(True)
            
            # Entropy
            axes[1, 1].plot(self.train_stats['entropy'])
            axes[1, 1].set_title('Policy Entropy')
            axes[1, 1].set_xlabel('Iteration')
            axes[1, 1].grid(True)
            
            plt.tight_layout()
            plot_path = os.path.join(self.config['checkpoint_dir'], 'training_curves.png')
            plt.savefig(plot_path, dpi=300)
            print(f"📈 训练曲线已保存到: {plot_path}")
        except ImportError:
            print("⚠️  未安装matplotlib，跳过绘图")


def main():
    """主函数"""

    # ===== 新增：命令行参数解析 =====
    parser = argparse.ArgumentParser(description='AlphaZero训练脚本（13x13五子棋）')
    parser.add_argument('--resume', type=str, default='latest_checkpoint.pth',
                        help='要加载的checkpoint文件名（默认：latest_checkpoint.pth）')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints_gomoku_13x13',
                        help='checkpoint保存目录（默认：./checkpoints_gomoku_13x13）')
    parser.add_argument('--iterations', type=int, default=1000,
                        help='总训练迭代次数（默认：1000）')
    parser.add_argument('--games_per_iter', type=int, default=20,
                        help='每次迭代自我对弈局数（默认：10）')
    parser.add_argument('--mcts_sims', type=int, default=400,
                        help='MCTS模拟次数（默认：400）')
    args = parser.parse_args()

    # 训练配置（针对13x13五子棋优化）
    config = {
        # 游戏参数
        'board_size': 13,
        'n_in_row': 5,
        'max_pieces': 0,          # 0表示无限制
        'max_game_moves': 200,    # 最大回合数

        # 网络参数
        'num_channels': 128,      # 卷积通道数
        'num_res_blocks': 5,      # 残差块数量

        # MCTS参数
        'mcts_simulations': args.mcts_sims,  # 每步MCTS模拟次数
        'c_puct': 0.8,            # UCB探索常数

        # 训练参数
        'n_iterations': args.iterations,      # 总迭代次数
        'n_games_per_iteration': args.games_per_iter,  # 每次迭代自我对弈局数
        'batch_size': 256,        # 训练批次大小
        'train_epochs': 5,        # 每次迭代训练轮数
        'buffer_size': 50000,     # 经验回放缓冲区大小

        # 优化器参数
        'learning_rate': 0.001,   # 学习率
        'l2_const': 1e-4,         # L2正则化
        'lr_decay_steps': 20,     # 学习率衰减步数
        'lr_decay_rate': 0.9,     # 学习率衰减率

        # 保存参数
        'checkpoint_dir': args.checkpoint_dir,
        'save_interval': 50       # 每N次迭代保存一次
    }
    
    # 打印配置
    print("\n📋 训练配置:")
    print("-" * 40)
    for key, value in config.items():
        print(f"{key:25s}: {value}")
    print("-" * 40)
    
    # 创建训练器
    trainer = AlphaZeroTrainer(config)
    
    # ===== 修改：加载指定的checkpoint =====
    print(f"\n🔄 尝试加载checkpoint: {args.resume}")
    loaded = trainer.load_checkpoint(args.resume)
    
    if loaded:
        print(f"\n✅ 成功加载checkpoint，将从迭代 {trainer.train_stats['iteration'] + 1} 开始训练")
    else:
        print(f"\n⚠️  未加载checkpoint，将从头开始训练")
        user_input = input("是否继续？(y/n): ")
        if user_input.lower() != 'y':
            print("训练已取消")
            return
    
    # 开始训练
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n\n⚠️  训练被用户中断")
        print("💾 正在保存当前进度...")
        trainer.save_checkpoint('interrupted_checkpoint.pth')
        print("✅ 进度已保存")


if __name__ == '__main__':
    main()
