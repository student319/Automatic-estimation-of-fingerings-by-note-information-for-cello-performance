import numpy as np
#import matplotlib.pyplot as plt
import time


def midi_accuracy(pred_midi, true_midi):
    correct_predictions = np.sum(np.abs(pred_midi - true_midi) <= 0)
    total_predictions = len(true_midi)
    accuracy = correct_predictions / total_predictions
    return accuracy


def distance_calculation(positions):
    n_values = [2, 3, 5, 7, 8, 10, 12, 14, 15, 17, 19, 20]
    position_lengths = {n: 33 * (1 / 1.059) ** n for n in n_values}

    # 移動距離と移動回数を初期化
    total_movement_distance = 0
    movement_count = 0
    previous_position_length = None

    for position in positions:
        # 現在のポジションの弦の長さを取得
        if position in position_lengths:
            current_position_length = position_lengths[position]
            
            # 初回以降に移動距離と移動回数をカウント
            if previous_position_length is not None:
                movement_distance = abs(previous_position_length - current_position_length)
                if movement_distance > 0:  # 移動があった場合のみカウント
                    total_movement_distance += movement_distance
                    movement_count += 1
            
            # 現在のポジション長を次の比較用に更新
            previous_position_length = current_position_length

    return total_movement_distance, movement_count


def speed_calculation(position, start):
    # 小節の境界インデックスを見つける
    bar_starts = np.where(np.diff(start) < 0)[0] + 1
    bar_starts = np.concatenate(([0], bar_starts, [len(start)]))

    # 小節ごとの移動距離を計算
    bar_distances = []
    for i in range(len(bar_starts) - 1):
        start_idx = bar_starts[i]
        end_idx = bar_starts[i + 1]
        bar_positions = position[start_idx:end_idx]
        bar_distance = np.sum(np.abs(np.diff(bar_positions)))  # ポジション移動距離の合計
        bar_distances.append(bar_distance) 
    
    return bar_distances

def time_calculation():
    end = time.time()
    return end

def difficulty_check():
    pass