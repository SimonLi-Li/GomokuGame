# game.py - 13x13五子棋
# 标准五子棋规则：先连成5子获胜

import numpy as np
from collections import deque

class GomokuGame:
    """
    13x13五子棋
    - 棋盘大小：13x13
    - 胜利条件：先连成五子获胜
    - 状态表示：0=空位, 1=黑子(先手), -1=白子(后手)
    """

    def __init__(self, board_size=13, n_in_row=5, max_pieces_per_player=0, max_moves=200):
        """
        初始化棋盘
        board_size: 棋盘边长（默认13）
        n_in_row: 连续多少子获胜（默认5）
        max_pieces_per_player: 每方最多棋子数（0表示无限制）
        max_moves: 最大回合数，防止无限循环（默认200）
        """
        self.board_size = board_size
        self.n_in_row = n_in_row
        self.max_pieces_per_player = max_pieces_per_player
        self.max_moves = max_moves
        self.reset()
    
    def reset(self):
        """重置棋盘到初始状态"""
        self.board = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        self.current_player = 1  # 1=黑子先手
        self.last_move = None    # 记录最后一步落子位置
        self.winner = None       # 获胜方
        self.game_over = False
        self.move_count = 0      # 回合计数器
        
        # 使用队列记录每方的棋子位置（仅在有限制时使用）
        if self.max_pieces_per_player > 0:
            self.black_pieces = deque(maxlen=self.max_pieces_per_player)  # 黑子位置队列
            self.white_pieces = deque(maxlen=self.max_pieces_per_player)  # 白子位置队列
        else:
            self.black_pieces = deque()  # 无限制
            self.white_pieces = deque()  # 无限制
        
        self.move_history = []   # 完整历史记录
        self.removed_piece = None  # 记录本回合移除的棋子位置
    
    def get_legal_moves(self):
        """
        获取所有合法落子位置（只能下空位）
        """
        return list(zip(*np.where(self.board == 0)))
    
    def get_player_pieces(self, player):
        """获取指定玩家的棋子队列"""
        if player == 1:
            return self.black_pieces
        else:
            return self.white_pieces
    
    def make_move(self, x, y):
        """
        在(x, y)位置落子
        自动处理棋子限制：如果超过3个棋子，移除最早的
        返回：是否落子成功
        """
        if self.game_over:
            return False
        
        if x < 0 or x >= self.board_size or y < 0 or y >= self.board_size:
            return False  # 越界
        
        if self.board[x, y] != 0:
            return False  # 该位置已有棋子
        
        self.removed_piece = None  # 重置移除记录
        
        # 获取当前玩家的棋子队列
        pieces_queue = self.get_player_pieces(self.current_player)

        # 如果设置了棋子数限制且已达上限，移除最早的棋子
        if self.max_pieces_per_player > 0 and len(pieces_queue) >= self.max_pieces_per_player:
            old_x, old_y = pieces_queue[0]  # 获取最早的棋子位置（队列头部）
            self.board[old_x, old_y] = 0     # 从棋盘移除
            self.removed_piece = (old_x, old_y)  # 记录移除位置
            # deque会自动移除最早的元素（因为设置了maxlen）
        
        # 下新棋
        self.board[x, y] = self.current_player
        pieces_queue.append((x, y))  # 添加到队列尾部
        
        self.last_move = (x, y)
        self.move_history.append({
            'player': self.current_player,
            'move': (x, y),
            'removed': self.removed_piece,
            'move_count': self.move_count
        })
        self.move_count += 1
        
        # 检查游戏是否结束
        if self._check_winner(x, y):
            self.winner = self.current_player
            self.game_over = True
        elif self.move_count >= self.max_moves:
            # 达到最大回合数，判定为平局
            self.winner = 0
            self.game_over = True
        elif len(self.get_legal_moves()) == 0:
            # 棋盘已满，判定为平局
            self.winner = 0
            self.game_over = True
        
        # 切换玩家
        self.current_player = -self.current_player
        return True
    
    def _check_winner(self, x, y):
        """
        检查在(x,y)落子后是否获胜
        检查四个方向：横向、纵向、主对角线、副对角线
        """
        player = self.board[x, y]
        directions = [
            [(0, 1), (0, -1)],   # 横向
            [(1, 0), (-1, 0)],   # 纵向
            [(1, 1), (-1, -1)],  # 主对角线 \
            [(1, -1), (-1, 1)]   # 副对角线 /
        ]
        
        for direction in directions:
            count = 1
            winning_line = [(x, y)]
            
            for dx, dy in direction:
                nx, ny = x + dx, y + dy
                while (0 <= nx < self.board_size and 
                       0 <= ny < self.board_size and 
                       self.board[nx, ny] == player):
                    count += 1
                    winning_line.append((nx, ny))
                    nx += dx
                    ny += dy
            
            if count >= self.n_in_row:
                self.winning_line = winning_line
                return True
        
        return False
    
    def get_state_for_network(self):
        """
        将当前棋盘状态转换为神经网络输入格式
        返回：形状为 (4, board_size, board_size) 的numpy数组
        """
        state = np.zeros((4, self.board_size, self.board_size), dtype=np.float32)
        
        # 通道0: 当前玩家的棋子
        state[0][self.board == self.current_player] = 1.0
        
        # 通道1: 对手的棋子
        state[1][self.board == -self.current_player] = 1.0
        
        # 通道2: 最后落子位置
        if self.last_move is not None:
            state[2][self.last_move] = 1.0
        
        # 通道3: 当前玩家标记
        if self.current_player == 1:
            state[3][:, :] = 1.0
        
        return state
    
    def copy(self):
        """深拷贝当前游戏状态"""
        new_game = GomokuGame(self.board_size, self.n_in_row, 
                             self.max_pieces_per_player, self.max_moves)
        new_game.board = self.board.copy()
        new_game.current_player = self.current_player
        new_game.last_move = self.last_move
        new_game.winner = self.winner
        new_game.game_over = self.game_over
        new_game.move_count = self.move_count
        new_game.black_pieces = self.black_pieces.copy()
        new_game.white_pieces = self.white_pieces.copy()
        new_game.move_history = self.move_history.copy()
        new_game.removed_piece = self.removed_piece
        return new_game
    
    def __str__(self):
        """打印棋盘（用于调试）"""
        symbols = {0: '·', 1: '●', -1: '○'}
        result = "  " + " ".join(str(i) for i in range(self.board_size)) + "\n"
        for i in range(self.board_size):
            result += f"{i} " + " ".join(symbols[cell] for cell in self.board[i]) + "\n"
        result += f"\n黑子位置: {list(self.black_pieces)}"
        result += f"\n白子位置: {list(self.white_pieces)}"
        result += f"\n回合数: {self.move_count}"
        return result
