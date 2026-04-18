# YOLO Detection with Docker

## Overview
YOLOを使った物体検出（Docker環境）

## Setup
docker build -t yolo-test .

## Run
docker run --rm -v ${PWD}:/app yolo-test

## Output
検出画像とFPSを出力

## Features
・誤検知抽出
・FPS計測