# YOLO Detection with Docker

## Overview
Docker環境上でYOLOを用いた物体検出を実装し、推論速度（FPS）と検出精度のバランスを検証した。

## System Architecture
Docker
Python
OpenCV
ONNX Runtime
YOLO

## Setup
docker build -t yolo-test .

## Run
docker run --rm -v ${PWD}:/app yolo-test

##Processing Flow
・画像入力
・YOLO推論
・後処理（NMS・閾値処理）
・バウンディングボックス描画
・FPS計測

##Results
・推論速度：12.5 FPS（モデル：YOLOv8-n）
・検出対象：車両など
・検出結果画像を出力

##Improvements
・FPS改善
　・モデル変更（m → s）によりFPS向上
　・解像度変更による処理速度改善
・誤検知対応
　・CONF_TH調整により低確度検出を削減
　・NMS調整で重複検出を削減
##Analysis
　・誤検知の原因
　　・類似物体の誤認識（例：車両の誤分類）
　　・環境要因（光・距離）
　・対応方針
　　・推論パラメータ調整で改善可能な範囲を確認
　　・根本対応は学習データ改善が必要
##Features
　・Dockerによる再現可能な環境
　・FPS計測機能
　・誤検知抽出機能（good / bad分類）
##Future Work
　・TensorRTによる高速化
　・カメラ入力対応
　・ログ出力・常駐化（systemd）