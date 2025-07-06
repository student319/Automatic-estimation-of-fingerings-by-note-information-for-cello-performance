import csv
import pickle
import numpy as np

# 処理対象のCSVファイルリスト

csv_files = [
    "v2.csv",
    "v3.csv",
    "v4.csv",
    "v5.csv",
    "v6.csv",
    "v7.csv",
    "v9.csv",
    "v10.csv",
    "v11.csv",
    "v12.csv",
    "v13.csv",
    "v14.csv",
    "v15.csv",
    "v17.csv",
    "v18.csv",
    "v19.csv",
    "v20.csv",
    "v21.csv"
]

# dtypeの定義（7フィールド）
# 注意！元のデータは4つ余分な要素を含んでいるため、makedata(train, test)を使えない
dtype = [
    ('pitch', '<i4'),
    ('start', '<f8'),
    ('duration', '<f8'),
    ('string', '<i4'),
    ('position', '<i4'),
    ('finger', '<i4'),
    ('expansion', '<i4')
]

# 全データを格納する辞書
pickle_data = {}

for csv_file in csv_files:
    # データをリストに格納
    data_list = []
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # ヘッダーをスキップ

        for row in reader:
            data_tuple = tuple(int(x) if i != 1 and i != 2 else float(x) for i, x in enumerate(row[:8]))
            data_list.append(data_tuple)


    testing = 1
    if testing == 1:
        #testing
        # 32音符ごとにチャンク分割し、不足を指定の値で埋める
        chunks = []
        for i in range(0, len(data_list), 32):
            chunk = data_list[i:i + 32]
            while len(chunk) < 32:
                chunk.append((35, 3.0, 3.0, 3, 3, 3, 0))  # 最初は54、他は3で埋める
            chunks.append(chunk)
    else:
        #training
        # ホップサイズとチャンクサイズ
        chunk_size = 32
        hop_size = 16
        # チャンクをスライディングウィンドウで分割
        chunks = []
        for i in range(0, len(data_list) - chunk_size + 1, hop_size):
            chunk = data_list[i:i + chunk_size]
            chunks.append(chunk)

        # 最後のチャンクが不足している場合にパディング
        if len(data_list) % hop_size != 0:
            chunk = data_list[-chunk_size:]
            while len(chunk) < chunk_size:
                chunk.append((35, 3.0, 3.0, 3, 3, 3, 3))  # デフォルト値で埋める
            chunks.append(chunk)




    # 各チャンクをNumPyのstructured arrayに変換
    segments_array = np.array(chunks, dtype=dtype)

    # 辞書に格納（ファイル名からキーを生成）
    key = f"\\{csv_file.split('.')[0]}"
    pickle_data[key] = {
        'segments': segments_array,
        'length': len(data_list)
    }

# 辞書をpickle形式で保存
#with open("trainingdataset32n3.pickle", "wb") as f:
with open("testingdataset32n3.pickle", "wb") as f:
    pickle.dump(pickle_data, f)

print("file created")
