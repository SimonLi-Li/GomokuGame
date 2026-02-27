# model.py - AlphaZero核心神经网络（五子棋版）
# 作用：使用简化的网络架构预测策略和价值

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """
    残差块（Residual Block）
    结构：输入 -> 卷积1 -> BN -> ReLU -> 卷积2 -> BN -> (+输入) -> ReLU
    """
    
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x):
        residual = x
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        
        return out


class PolicyValueNet(nn.Module):
    """
    AlphaZero策略价值网络（五子棋版）
    输入：棋盘状态 (batch_size, 4, board_size, board_size)
    输出：
        - policy: 每个位置的落子概率 (batch_size, board_size*board_size)
        - value: 当前局面的胜率评估 (batch_size, 1)，范围[-1, 1]
    """

    def __init__(self, board_size=13, num_channels=128, num_res_blocks=5):
        """
        board_size: 棋盘大小（五子棋为13）
        num_channels: 卷积通道数（五子棋用128）
        num_res_blocks: 残差块数量（五子棋用5层）
        """
        super(PolicyValueNet, self).__init__()
        
        self.board_size = board_size
        self.num_channels = num_channels
        
        # 初始卷积层
        self.conv_input = nn.Conv2d(4, num_channels, kernel_size=3, padding=1, bias=False)
        self.bn_input = nn.BatchNorm2d(num_channels)
        
        # 残差塔
        self.res_blocks = nn.ModuleList([
            ResidualBlock(num_channels) for _ in range(num_res_blocks)
        ])
        
        # 策略头
        self.policy_conv = nn.Conv2d(num_channels, 16, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(16)
        self.policy_fc = nn.Linear(16 * board_size * board_size, board_size * board_size)
        
        # 价值头
        self.value_conv = nn.Conv2d(num_channels, 16, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(16)
        self.value_fc1 = nn.Linear(16 * board_size * board_size, 64)
        self.value_fc2 = nn.Linear(64, 1)
    
    def forward(self, x):
        """前向传播"""
        # 初始卷积
        x = F.relu(self.bn_input(self.conv_input(x)))
        
        # 残差块堆叠
        for res_block in self.res_blocks:
            x = res_block(x)
        
        # 策略头
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(-1, 16 * self.board_size * self.board_size)
        policy_logits = self.policy_fc(policy)
        
        # 价值头
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(-1, 16 * self.board_size * self.board_size)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))
        
        return policy_logits, value
    
    def predict(self, state, device):
        """
        单次预测（推理模式）
        state: numpy数组 (4, board_size, board_size)
        返回：(policy概率分布, value标量)
        """
        self.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            
            policy_logits, value = self.forward(state_tensor)
            
            policy_probs = F.softmax(policy_logits, dim=1).cpu().numpy()[0]
            value_scalar = value.cpu().numpy()[0][0]
            
        return policy_probs, value_scalar


class AlphaZeroLoss(nn.Module):
    """
    AlphaZero损失函数（修复版）
    损失 = (价值预测 - 真实胜负)² - π·log(p) + c·||θ||²
    """
    
    def __init__(self, value_loss_weight=1.0, l2_weight=1e-4):
        super(AlphaZeroLoss, self).__init__()
        self.value_loss_weight = value_loss_weight
        self.l2_weight = l2_weight
    
    def forward(self, policy_logits, value, target_policy, target_value, model):
        """
        计算总损失
        policy_logits: 网络输出的策略logits (batch_size, board_size*board_size)
        value: 网络输出的价值 (batch_size, 1)
        target_policy: MCTS搜索得到的目标策略分布 (batch_size, board_size*board_size)
        target_value: 真实对局结果 (batch_size, 1) 或 (batch_size,)
        """
        # 价值损失（均方误差）- 修复形状不匹配问题
        value_loss = F.mse_loss(value.view(-1), target_value.view(-1))
        
        # 策略损失（交叉熵）
        policy_loss = -torch.mean(torch.sum(target_policy * F.log_softmax(policy_logits, dim=1), dim=1))
        
        # L2正则化
        l2_reg = 0
        for param in model.parameters():
            l2_reg += torch.sum(param ** 2)
        
        total_loss = self.value_loss_weight * value_loss + policy_loss + self.l2_weight * l2_reg
        
        return total_loss, value_loss, policy_loss
