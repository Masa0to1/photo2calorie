import os
import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cnn_model

LABELS = ["マグロ寿司", "サラダ", "麻婆豆腐", "餃子", "牛丼", "ピザ", "ステーキ", "ハンバーガー"]
CALORIES = [588, 118, 648, 196, 652, 1176, 355, 294]

def load_and_preprocess_image(image_path, im_rows, im_cols, im_color):
    # 画像を読み込む
    img = Image.open(image_path)
    img = img.convert("RGB") # 色空間をRGBに
    img = img.resize((im_cols, im_rows)) # サイズ変更
    plt.imshow(img)
    plt.show()
    # データに変換
    x = np.asarray(img)
    x = x.reshape(-1, im_rows, im_cols, im_color)
    x = x / 255
    return x

def predict_image(image_path, model):
    x = load_and_preprocess_image(image_path, im_rows, im_cols, im_color)
    pre = model.predict(x)[0]
    idx = pre.argmax()
    per = int(pre[idx] * 100)
    return idx, per
    
def print_prediction_result(image_path, model):
    idx, per = predict_image(image_path, model)
    # 答えを表示
    print("この写真は、", LABELS[idx], "で、カロリーは", CALORIES[idx], "kcal")
    print("可能性は、", per, "%")

if __name__ == '__main__':
    target_image_dir = "target_image"
    target_image = sys.argv[1]
    target_image_path = os.path.join(target_image_dir, target_image)
    
    im_rows = 32 # 画像の縦ピクセルサイズ
    im_cols = 32 # 画像の横ピクセルサイズ
    im_color = 3 # 画像の色空間
    in_shape = (im_rows, im_cols, im_color)
    nb_classes = 8

    # 保存したCNNモデルを読み込む
    model = cnn_model.get_model(in_shape, nb_classes)
    model.load_weights('./weight/photos-model.hdf5')
    
    print_prediction_result(target_image_path, model)