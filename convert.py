import os
import sys
import numpy as np
from PIL import Image

# 画像ファイルが格納されているディレクトリ
image_directory = sys.argv[1]

# 画像ファイルのリストを取得
image_files = [f for f in os.listdir(image_directory) if f.endswith(".jpg")]

# 画像データを格納するリスト
image_data_list = []

# 画像ファイルを読み込んでリストに追加
for image_file in image_files:
    image_path = os.path.join(image_directory, image_file)
    image = Image.open(image_path)
    image_data = np.array(image)
    image_data_list.append(image_data)

# 画像データのリストをNumPy配列に変換
image_data_array = np.array(image_data_list)

# Pathを設定
output_directory = 'image_data'  # 保存先のディレクトリ名
output_file = 'image_data.npz'  # 保存するファイル名

# 出力ディレクトリが存在しない場合、自動的に作成
os.makedirs(output_directory, exist_ok=True)

# 画像データをnpzファイルに保存
output_path = os.path.join(output_directory, output_file)
np.savez(output_path, images=image_data_array)

print("The images have been converted and saved to", output_path)