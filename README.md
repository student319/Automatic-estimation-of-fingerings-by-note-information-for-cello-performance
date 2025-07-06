# Automatic Estimation of Fingering by Note Information for Cello Performance

本リポジトリは、チェロ演奏における最適な運指を楽譜情報から自動推定する深層学習モデルの実装を公開したものです。  
ヴァイオリンの運指パターンで事前学習を行い、それを転移学習・ファインチューニングすることでチェロへの適用を図っています。

## 概要

- 音符のMIDI情報（音高・開始時刻・継続時間）を入力とし、弦・ポジション・指番号のラベルを推定
- 4種類のモデルを実装・比較
  - (a) 転移学習モデル
  - (b) 簡約モデル（エンコーダ・デコーダなし）
  - (c) ファインチューニングモデル
  - (d) 深層学習モデル（事前学習なし）

## 使用技術

- Python 3.10.12
- TensorFlow 2.17.0
- NumPy 1.26.4

## ディレクトリ構成

project-root/  
│  
├── data/ # データセット（TNUA, Celloデータ）  
├── models/ # 学習済みモデルの保存先  
├── notebooks/ # 実験用ノートブック  
├── src/ # モデル定義・学習スクリプトなど  
│ ├── model.py  
│ ├── train.py  
│ └── utils.py  
├── results/ # 実験結果・グラフ・出力など  
├── main.py # 実行ファイル  
├── requirements.txt # 必要ライブラリ  
└── README.md  


## セットアップ

```bash  
git clone https://github.com/your-username/cello-fingering-estimation.git  
cd cello-fingering-estimation  
pip install -r requirements.txt


## 実行方法  

python main.py


## データセット

- 事前学習用：TNUA Violin Fingering Dataset
- 転移学習・評価用：デュポール「チェロ奏法と21の練習曲」に基づき自作したデータセット


## 引用・参考文献

Cheung et al. “Semi-supervised violin fingering generation using variational autoencoders,” ISMIR, 2021.

TensorFlow: Transfer Learning and Fine-Tuning

Sheet Music Scanner: https://sheetmusicscanner.com

MuseScore: https://musescore.org/ja

デュポール「チェロ奏法と21の練習曲」音楽之友社, 2024.
