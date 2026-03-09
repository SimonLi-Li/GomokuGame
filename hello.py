# app.py - Flask Web后端（13x13五子棋 - 支持AI对战模式 - 状态同步优化）

from flask import Flask, render_template, request, jsonify
import torch
import numpy as np
import os
import sys
import uuid
from threading import Lock
from datetime import datetime, timedelta
from collections import OrderedDict

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from game import GomokuGame
from model import PolicyValueNet
from mcts import MCTS

app = Flask(__name__)

# 全局配置
BOARD_SIZE = 13
N_IN_ROW = 5
MAX_PIECES = 0  # 0表示无限制
MAX_MOVES = 200
MODEL_FILE = 'checkpoints_gomoku_13x13/final_model.pth'
SESSION_TIMEOUT_MINUTES = 30
MAX_SESSIONS = 1000
MCTS_DEFAULT_SIMULATIONS = 400

# 设备配置（延迟到实际使用时确定，避免模块导入时初始化MPS）
device = None


def _get_device():
    global device
    if device is None:
        forced = os.environ.get('PYTORCH_DEVICE')
        if forced:
            device = torch.device(forced)
        elif torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    return device

# 全局模型
global_model = None

# 会话存储
game_sessions = OrderedDict()
sessions_lock = Lock()


class GameSession:
    """单个用户的游戏会话"""
    def __init__(self, session_id, n_simulations=400, game_mode='human_vs_ai', ai2_simulations=400):
        self.session_id = session_id
        self.game = GomokuGame(BOARD_SIZE, N_IN_ROW, MAX_PIECES, MAX_MOVES)
        self.game_mode = game_mode
        
        # AI1 (黑棋或人类对手)
        self.mcts1 = MCTS(
            policy_value_fn=policy_value_fn,
            c_puct=0.8,
            n_simulations=n_simulations,
            temperature=1e-3,
            add_dirichlet_noise=False
        )
        
        # AI2 (白棋，仅在AI vs AI模式下使用)
        if game_mode == 'ai_vs_ai':
            self.mcts2 = MCTS(
                policy_value_fn=policy_value_fn,
                c_puct=0.8,
                n_simulations=ai2_simulations,
                temperature=1e-3,
                add_dirichlet_noise=False
            )
        else:
            self.mcts2 = None
        
        self.last_activity = datetime.now()
        self.player_color = 'black'
        self.move_lock = Lock()
    
    def update_activity(self):
        """更新最后活跃时间"""
        self.last_activity = datetime.now()
    
    def is_expired(self):
        """检查会话是否超时"""
        return datetime.now() - self.last_activity > timedelta(minutes=SESSION_TIMEOUT_MINUTES)


def convert_to_python_types(obj):
    """递归转换numpy类型为Python原生类型"""
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_python_types(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_python_types(item) for item in obj]
    else:
        return obj


def load_model():
    """加载训练好的模型（全局共享）"""
    global global_model

    dev = _get_device()
    base_path = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_path, MODEL_FILE)

    print(f"🔍 正在检查模型路径: {model_path}")

    global_model = PolicyValueNet(
        board_size=BOARD_SIZE,
        num_channels=128,
        num_res_blocks=5
    ).to(dev)

    if os.path.exists(model_path):
        try:
            checkpoint = torch.load(model_path, map_location=dev, weights_only=False)

            if 'model_state_dict' in checkpoint:
                global_model.load_state_dict(checkpoint['model_state_dict'])
            else:
                global_model.load_state_dict(checkpoint)

            global_model.eval()
            print(f"✅ 模型加载成功！")

            if 'episode' in checkpoint:
                print(f"📊 训练轮次: {checkpoint['episode']}")
            if 'win_rate' in checkpoint:
                print(f"🎯 最终胜率: {checkpoint['win_rate']:.2%}")

        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            print("⚠️  将使用随机初始化的模型运行")
    else:
        print(f"⚠️  未找到模型文件 {MODEL_FILE}，将使用随机化模型")

    return global_model


def _ensure_model_loaded():
    """延迟加载模型，确保在gunicorn worker fork之后才初始化MPS"""
    global global_model
    if global_model is None:
        load_model()
        dev = _get_device()
        print("="*60)
        print(f"💻 计算设备: {dev}")
        print(f"🎮 棋盘大小: {BOARD_SIZE}x{BOARD_SIZE}")
        print(f"🎯 获胜条件: {N_IN_ROW}子连线")
        print(f"👥 支持多用户并发游戏（完全隔离）")
        print(f"🤖 支持AI vs AI对战模式")
        print(f"⏱️  会话超时: {SESSION_TIMEOUT_MINUTES}分钟")
        print(f"📊 最大会话数: {MAX_SESSIONS}")
        print("="*60)


def policy_value_fn(state):
    """策略价值函数 - 桥接MCTS与PyTorch模型"""
    _ensure_model_loaded()
    global_model.eval()
    return global_model.predict(state, _get_device())


def get_game_session(session_id):
    """获取指定ID的游戏会话"""
    if not session_id:
        return None
    
    with sessions_lock:
        if session_id not in game_sessions:
            return None
        
        game_sessions.move_to_end(session_id)
        game_session = game_sessions[session_id]
        game_session.update_activity()
        return game_session


def cleanup_expired_sessions():
    """清理过期的游戏会话"""
    with sessions_lock:
        expired_ids = [
            sid for sid, gs in list(game_sessions.items()) 
            if gs.is_expired()
        ]
        
        for sid in expired_ids:
            del game_sessions[sid]
        
        while len(game_sessions) > MAX_SESSIONS:
            game_sessions.popitem(last=False)
        
        if expired_ids:
            print(f"🧹 清理了 {len(expired_ids)} 个过期会话，当前活跃会话: {len(game_sessions)}")


def get_board_state(game):
    """获取当前棋盘的完整状态"""
    return convert_to_python_types(game.board.tolist())


def parse_simulations(value, default=MCTS_DEFAULT_SIMULATIONS):
    """解析MCTS模拟次数，非法输入回退到默认值"""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


@app.route('/')
def index():
    """主页"""
    return render_template('index.html')


@app.route('/new_game', methods=['POST'])
def new_game():
    """开始新游戏"""
    data = request.json
    
    client_session_id = data.get('client_session_id')
    if not client_session_id:
        return jsonify({'success': False, 'message': '缺少会话标识'})
    
    player_color = data.get('player_color', 'black')
    n_simulations = parse_simulations(data.get('n_simulations', MCTS_DEFAULT_SIMULATIONS))
    game_mode = data.get('game_mode', 'human_vs_ai')
    ai2_simulations = parse_simulations(data.get('ai2_simulations', MCTS_DEFAULT_SIMULATIONS))
    
    # 创建新的游戏会话
    game_session = GameSession(
        client_session_id, 
        n_simulations, 
        game_mode, 
        ai2_simulations
    )
    game_session.player_color = player_color
    
    with sessions_lock:
        game_sessions[client_session_id] = game_session
        game_sessions.move_to_end(client_session_id)
    
    print(
        f"🎮 新游戏 | 会话: {client_session_id[:8]}... | 模式: {game_mode} "
        f"| AI1模拟: {n_simulations} | AI2模拟: {ai2_simulations} | 总会话数: {len(game_sessions)}"
    )
    
    response = {
        'success': True,
        'message': '游戏开始',
        'player_color': player_color,
        'max_pieces': MAX_PIECES,
        'game_mode': game_mode,
        'board_state': get_board_state(game_session.game)
    }
    
    # AI vs AI模式：让黑棋先走
    if game_mode == 'ai_vs_ai':
        ai_move = get_ai_move(game_session, use_mcts2=False)
        response['ai_move'] = convert_to_python_types(ai_move)
        response['board_state'] = get_board_state(game_session.game)
    # 人类 vs AI模式：如果玩家选择白棋，AI执黑先手
    elif player_color == 'white':
        ai_move = get_ai_move(game_session, use_mcts2=False)
        response['ai_move'] = convert_to_python_types(ai_move)
        response['board_state'] = get_board_state(game_session.game)
    
    cleanup_expired_sessions()
    
    return jsonify(response)


@app.route('/make_move', methods=['POST'])
def make_move():
    """处理玩家落子（仅人类vs AI模式）"""
    data = request.json
    
    client_session_id = data.get('client_session_id')
    if not client_session_id:
        return jsonify({'success': False, 'message': '缺少会话标识'})
    
    game_session = get_game_session(client_session_id)
    
    if game_session is None:
        return jsonify({'success': False, 'message': '游戏会话不存在，请重新开始'})
    
    with game_session.move_lock:
        x = int(data.get('x'))
        y = int(data.get('y'))
        
        # 1. 玩家落子
        if not game_session.game.make_move(x, y):
            return jsonify({'success': False, 'message': '非法落子'})
        
        player_removed = game_session.game.removed_piece
        # 同步更新所有MCTS树
        game_session.mcts1.update_with_move((x, y))
        if game_session.mcts2:
            game_session.mcts2.update_with_move((x, y))
        
        # 2. 检查玩家是否获胜
        if game_session.game.game_over:
            return jsonify(convert_to_python_types({
                'success': True,
                'game_over': True,
                'winner': game_session.game.winner,
                'winning_line': game_session.game.winning_line if hasattr(game_session.game, 'winning_line') else [],
                'player_removed': player_removed,
                'move_count': game_session.game.move_count,
                'board_state': get_board_state(game_session.game)
            }))
        
        # 3. AI 思考并落子
        ai_move = get_ai_move(game_session, use_mcts2=False)
        
        # 4. 检查AI落子后是否结束
        game_over = game_session.game.game_over
        winner = game_session.game.winner if game_over else 0
        
        return jsonify(convert_to_python_types({
            'success': True,
            'ai_move': ai_move,
            'game_over': game_over,
            'winner': winner,
            'winning_line': game_session.game.winning_line if hasattr(game_session.game, 'winning_line') else [],
            'player_removed': player_removed,
            'move_count': game_session.game.move_count,
            'board_state': get_board_state(game_session.game)
        }))


@app.route('/ai_vs_ai_step', methods=['POST'])
def ai_vs_ai_step():
    """执行AI vs AI的单步对局"""
    data = request.json
    
    client_session_id = data.get('client_session_id')
    if not client_session_id:
        return jsonify({'success': False, 'message': '缺少会话标识'})
    
    game_session = get_game_session(client_session_id)
    
    if game_session is None:
        return jsonify({'success': False, 'message': '游戏会话不存在，请重新开始'})
    
    if game_session.game_mode != 'ai_vs_ai':
        return jsonify({'success': False, 'message': '当前不是AI对战模式'})
    
    with game_session.move_lock:
        # 判断当前是哪个AI的回合
        current_player = game_session.game.current_player
        use_mcts2 = (current_player == -1)  # 白棋使用mcts2
        
        # AI落子
        ai_move = get_ai_move(game_session, use_mcts2=use_mcts2)
        
        # 检查游戏是否结束
        game_over = game_session.game.game_over
        winner = game_session.game.winner if game_over else 0
        
        return jsonify(convert_to_python_types({
            'success': True,
            'ai_move': ai_move,
            'current_player': current_player,
            'game_over': game_over,
            'winner': winner,
            'winning_line': game_session.game.winning_line if hasattr(game_session.game, 'winning_line') else [],
            'move_count': game_session.game.move_count,
            'board_state': get_board_state(game_session.game)
        }))


def get_ai_move(game_session, use_mcts2=False):
    """执行AI思考（针对特定会话）"""
    # 选择使用哪个MCTS
    mcts = game_session.mcts2 if use_mcts2 and game_session.mcts2 else game_session.mcts1

    actions, probs = mcts.get_action_probs(game_session.game, temperature=1e-3)

    # 过滤掉非法动作（已被占据的位置），防止MCTS树复用导致的脏数据
    legal_moves = set(game_session.game.get_legal_moves())
    legal_mask = np.array([a in legal_moves for a in actions])

    if legal_mask.any():
        # 将非法动作概率置零并重新归一化
        filtered_probs = probs * legal_mask
        prob_sum = filtered_probs.sum()
        if prob_sum > 1e-10:
            filtered_probs /= prob_sum
        else:
            filtered_probs = legal_mask.astype(float) / legal_mask.sum()
        best_action_idx = np.argmax(filtered_probs)
    else:
        # 所有MCTS动作都非法，从合法动作中随机选一个
        legal_list = list(legal_moves)
        if legal_list:
            best_action_idx = None
            best_action = legal_list[0]
        else:
            return None

    if best_action_idx is not None:
        best_action = actions[best_action_idx]

    state = game_session.game.get_state_for_network()
    _, value_eval = policy_value_fn(state)

    win_rate = (value_eval + 1.0) / 2.0

    success = game_session.game.make_move(best_action[0], best_action[1])
    if not success:
        # 兜底：如果仍然失败，从合法动作中选取
        legal_list = list(game_session.game.get_legal_moves())
        if not legal_list:
            return None
        best_action = legal_list[0]
        game_session.game.make_move(best_action[0], best_action[1])

    ai_removed = game_session.game.removed_piece

    # 同步更新所有MCTS树，防止树复用时出现脏数据
    game_session.mcts1.update_with_move(best_action)
    if game_session.mcts2:
        game_session.mcts2.update_with_move(best_action)

    return {
        'x': int(best_action[0]),
        'y': int(best_action[1]),
        'win_rate': float(win_rate),
        'removed_piece': ai_removed,
        'is_ai2': use_mcts2
    }




if __name__ == '__main__':
    import argparse
    import subprocess
    import platform

    parser = argparse.ArgumentParser(description='13x13五子棋服务器（Gunicorn）')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='监听地址')
    parser.add_argument('--port', type=int, default=8080, help='监听端口')
    parser.add_argument('--threads', type=int, default=4, help='线程数')
    parser.add_argument('--timeout', type=int, default=120, help='请求超时时间（秒）')
    args = parser.parse_args()

    env = os.environ.copy()

    # macOS上MPS (Metal) 不支持在fork的子进程中使用，会导致SIGABRT崩溃。
    # gunicorn的prefork worker模型会触发此问题，即使设置了
    # OBJC_DISABLE_INITIALIZE_FORK_SAFETY也无法解决。
    # 解决方案：在gunicorn worker中强制使用CPU推理。
    if platform.system() == 'Darwin' and torch.backends.mps.is_available():
        env['PYTORCH_DEVICE'] = 'cpu'
        print("⚠️  macOS检测到MPS，但gunicorn worker中不兼容，已切换为CPU推理")

    cmd = [
        sys.executable, '-m', 'gunicorn',
        '--worker-class', 'gthread',
        '--workers', '1',
        '--threads', str(args.threads),
        '-b', f'{args.host}:{args.port}',
        '--timeout', str(args.timeout),
        'hello:app'
    ]
    print(f"🚀 使用Gunicorn启动服务: http://{args.host}:{args.port} (threads={args.threads})")
    subprocess.run(cmd, env=env)
