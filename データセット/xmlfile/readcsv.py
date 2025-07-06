import csv

# 読み込みたいCSVファイルのパスを指定
file_path = 'out2.csv'

with open(file_path, mode='r', encoding='utf-8') as file:
    reader = csv.reader(file)
    
    # 最初の行（タイトル行）を取得して表示
    header = next(reader)
    print("Header:", header)
    
    # 続く3行分のデータを取得して表示
    for i in range(3):
        row = next(reader, None)  # 次の行が存在するか確認しながら取得
        if row is not None:
            print("Row", i + 1, ":", row)
        else:
            print("Row", i + 1, ": No more data")
