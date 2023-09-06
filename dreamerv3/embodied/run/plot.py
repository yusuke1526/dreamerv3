from io import BytesIO

import numpy as np
import matplotlib.pyplot as plt


def plot_action_on_video(video, action):
  '''動画に行動の実数値のグラフをプロットする
  args:
    video: numpy array
      (sequence_length, w, h, c)
    action: numpy array
      (sequence_length, action_dim)
  '''
  sequence_length, w, h, c = video.shape
  action_dim = action.shape[-1]

  # グラフをプロットする部分のサイズを設定
  graph_width = w // 4
  graph_height = h // 4

  # 動画フレームと行動のグラフを合成する新しい動画データを作成
  video_with_graph = np.copy(video)

  # 各バッチ内の各フレームについて処理
  for frame_idx in range(sequence_length):
    # 動画フレームと対応する行動データを取得
    frame = video[frame_idx]
    action_data = action[frame_idx]

    # 動画フレームの右下にグラフをプロットする領域を切り出し
    graph_area = frame[-graph_height:, -graph_width:]

    # 行動のグラフをプロット
    fig, ax = plt.subplots(figsize=(graph_width / 100, graph_height / 100), dpi=100)  # グラフのサイズを指定
    ax.bar([0, 1], action_data, color='white')
    ax.set_ylim(-1.0, 1.0)
    ax.axis('off')  # 軸を非表示にする

    # プロットしたグラフをBytesIOに保存
    with BytesIO() as data:
      plt.savefig(data, format='png', transparent=True)
      plt.close()

      # BytesIOから画像を読み込んで画像にグラフを配置
      data.seek(0)
      graph_image = (plt.imread(data) * 255).astype(int)
      video_with_graph[frame_idx, -graph_height:, -graph_width:] = np.where(graph_image[:, :, 3:4] == 0, video_with_graph[frame_idx, -graph_height:, -graph_width:], graph_image[:, :, :3])

  # 合成した動画データを返す
  return video_with_graph