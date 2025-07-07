# チェロ演奏のための音符情報による運指の自動推定

本リポジトリは、チェロ演奏における最適な運指を楽譜情報から自動推定する深層学習モデルの実装を公開したものです。
ヴァイオリンの運指パターンで事前学習を行い、それを転移学習・ファインチューニングすることでチェロへの適用を図っています。
本研究成果は、2025年5月に開催された「**電子情報通信学会 イメージ・メディア・クオリティ研究会（IEICE-IMQ）**」にて発表を行いました。

## 概要

- 音符のMIDI情報（音高・開始時刻・継続時間）を入力とし、弦・ポジション・指番号のラベルを推定
- 4種類のモデルを実装・比較（本リポジトリでは(a)のみ公開されています）
  - (a) 転移学習モデル
  - (b) 簡約モデル（エンコーダ・デコーダなし）
  - (c) ファインチューニングモデル
  - (d) 深層学習モデル（事前学習なし）

## Requirements

- Python 3.10.12
- TensorFlow 2.17.0
- NumPy 1.26.4

## ディレクトリ構成

project-root/  
│  
├── dataset/ # データセット（TNUA, Celloデータ）  
├── program/ # モデル定義・学習スクリプトなど  
│ ├── main.py  
│ ├── custom_layers.py  
│ └── dataset_prepare.py  
└── README.md  


## セットアップ

```bash  
git clone https://github.com/student319/Automatic-estimation-of-fingerings-by-note-information-for-cello-performance.git  
cd プログラム
```

## 実行方法  

```bash
python main.py
```

## データセット

- 事前学習用：TNUA Violin Fingering Dataset
- 転移学習・評価用：デュポール「チェロ奏法と21の練習曲」に基づき自作したデータセット


## 学会発表：  
[IEICE-IMQ 研究会 プログラム（2025年5月）](https://ken.ieice.org/ken/program/index.php?tgs_regid=29051ffd263895bed9d2b9d591ba66c06956421ef30b5393bfa12d1b707d3f7a&tgid=IEICE-IMQ)


## 引用・参考文献

Cheung et al. “Semi-supervised violin fingering generation using variational autoencoders,” ISMIR, 2021.

TensorFlow: Transfer Learning and Fine-Tuning

Sheet Music Scanner: https://sheetmusicscanner.com

MuseScore: https://musescore.org/ja

デュポール「チェロ奏法と21の練習曲」音楽之友社, 2024.
