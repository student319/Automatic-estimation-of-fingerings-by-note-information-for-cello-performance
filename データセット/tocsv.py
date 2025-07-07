"""
musicXML形式のファイルを音符に関する必要な情報を抽出した上でcsvファイルに変換するプログラムです
"""

from music21 import converter, note, stream, chord
import pandas as pd

# ファイル名パラメータ
xml_file = "21songs_32edited/Song21.mxl"
csv_file = "21songs_32csv/21.csv"

# MusicXMLファイルを読み込む
score = converter.parse(xml_file)

# 結果を格納するリスト
data = []
part = score.parts[0]  # 最初のパートを指定

# 小節ごとに音符を解析
if(1 == 1):
    for measure in part.getElementsByClass(stream.Measure):  # 小節ごとのループ
        measure_duration = measure.barDuration.quarterLength  # 小節の長さ(四分音符単位)
        
        for element in measure.notes:  # 小節内の音符を取得
            if isinstance(element, note.Note):
                # MIDI音高
                midi = element.pitch.midi
                
                # 音符の開始時刻 (onset: 小節単位で0〜1の範囲に変換)
                onset = element.offset / measure_duration
                
                # 音符の継続時間 (四分音符単位)
                duration = element.quarterLength / measure_duration

                # 空欄のフィールド (string, position, finger)
                string = None
                position = None
                finger = None

                # データをリストに追加
                data.append([midi, onset, duration, string, position, finger])
            elif isinstance(element, chord.Chord):  # 重音の場合
                # 最も音高が高い音を取得
                lowest_note = min(element.notes, key=lambda n: n.pitch.midi)
                midi = lowest_note.pitch.midi
                
                # 音符の開始時刻 (onset: 小節単位で0〜1の範囲に変換)
                onset = element.offset / measure_duration
                
                # 音符の継続時間 (四分音符単位)
                duration = element.quarterLength / measure_duration
                
                # 空欄のフィールド (string, position, finger)
                string = None
                position = None
                finger = None

                # データをリストに追加
                data.append([midi, onset, duration, string, position, finger])

# データをDataFrameに変換
df = pd.DataFrame(data, columns=["MIDI", "Onset", "Duration", "String", "Position", "Finger"])

# CSVファイルとして保存
df.to_csv(csv_file, index=False)

print(f"CSVファイルが保存されました: {csv_file}")
