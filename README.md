# チェロ演奏のための音符情報による運指の自動推定

本リポジトリは、チェロ演奏における最適な運指を楽譜情報から自動推定する深層学習モデルの実装を公開したものである。
本研究は、Cheungらによるヴァイオリン運指推定の先行研究［1］を基に、ヴァイオリンの運指パターンで事前学習を行い、それを転移学習・ファインチューニングすることでチェロへの適用を図った。
本研究成果は、2025年5月に開催された「**電子情報通信学会 イメージ・メディア・クオリティ研究会（IEICE-IMQ）**」にて発表を行った。

## 概要

- 音符のMIDI情報（音高・開始時刻・継続時間）を入力とし、弦・ポジション・指番号のラベルを推定
- 4種類のモデルを実装・比較（本リポジトリでは(a)のみ公開されている）
  - (a) 転移学習モデル
  - (b) 簡約モデル（エンコーダ・デコーダなし）
  - (c) ファインチューニングモデル
  - (d) 深層学習モデル（事前学習なし）

## 推奨環境

- Python 3.10.12
- TensorFlow 2.17.0
- NumPy 1.26.4

## ディレクトリ構成

project-root/  
├── dataset/                  # データセットおよびファイル形式変換コード  
│   ├── tocsv.py              # musicXML → CSV  
│   ├── make_pickle.py        # CSV → pickle  
│   └── data/                 # チェロ用の自作データセット（CSV）  
│  
├── program/                  # ソースコード  
│   ├── main.py               # 実行ファイル  
│   ├── custom_layers.py      # カスタム層の定義  
│   └── dataset_prepare.py    # データ前処理  
│  
└── README.md  

## データセット

- 事前学習用：TNUA Violin Fingering Dataset[2]
- 転移学習・評価用：デュポール　チェロ奏法と21の練習曲[3]に基づき自作したデータセット


## セットアップ・実行

```bash  
git clone https://github.com/student319/Automatic-estimation-of-fingerings-by-note-information-for-cello-performance.git  
cd program
python main.py
```


## 学会発表：  
[IEICE-IMQ 研究会 プログラム（2025年5月）](https://ken.ieice.org/ken/program/index.php?tgs_regid=29051ffd263895bed9d2b9d591ba66c06956421ef30b5393bfa12d1b707d3f7a&tgid=IEICE-IMQ)


## 参考文献

[1] Vincent KM Cheung, Hsuan-Kai Kao, Li Su, et al. Semi-supervised violin finger ing generation using variational autoencoders. In ISMIR, pages 113–120, 2021.

[2] Yi-Hsin Jen, Tsung-Ping Chen, Shih-Wei Sun, and Li Su. Positioning left-hand movement in violin performance: A system and user study of fingering pattern generation. In Proceedings of the 26th International Conference on Intelligent User Interfaces, pages 208–212, 2021.

[3] デュポール. チェロ奏法と 21 の練習曲 運指・運弓 に 関 す る 試 論 と エ チ ュ ー ド ( 原 典 版/solo+bass). 音楽之友社, 2024.
