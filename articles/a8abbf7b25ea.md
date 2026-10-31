---
title: "PythonとQiskitで実装する量子ニューラルネットワーク：量子回路を用いた分類タスクの構築"
emoji: "📝"
type: "tech"
topics: ["ai", "llm", "量子コンピュータ"]
published: true
---

## 導入：量子回路をレイヤーとして扱う考え方

古典的なニューラルネットワークにおいて、各層は入力データに対して非線形な変換を行い、特徴を抽出します。AI×量子技術の文脈において、量子ニューラルネットワーク（QNN）は、量子回路自体を学習可能な「レイヤー」として機能させるアプローチです。

具体的には、古典的なデータを量子状態へとエンコード（量子埋め込み）し、パラメータ化された量子回路（Parameterized Quantum Circuit: PQC）に作用させます。この回路の回転角などのパラメータを、古典的な最適化アルゴリズム（AdamやCOBYLAなど）を用いて更新していくことで、特定のタスクに適した回路構成を学習させます。

本記事では、Qiskitを用いて、量子回路をニューラルネットワークの層として構成し、2次元データの分類タスクを実装するプロセスを解説します。

## 量子エンコーディングとパラメータ化回路の設計

QNNの構成要素は、大きく分けて「データエンコーディング」と「パラメータ化回路（Ansatz）」の2つです。

### 1. データエンコーディング（Feature Map）
古典的な数値（$x, y$）を、量子ビットの位相や振幅に変換するプロセスです。ここでは、回転ゲート（$R_x, R_y, R_z$）を用いて、入力値を量子状態のパラメータとして埋め込む手法をとります。

### 2. パラザメータ化回路（Ansatz）
学習の対象となる重みに相当するのが、この回路の回転角です。入力データとは独立した「学習可能なパラメータ」を含んだ回路を用意します。

## 実装：QiskitによるQNNの構築

以下に、2クラス分類タスクを行うための最小構成のコードを示します。

```python
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit_machine_learning.algorithms import NeuralNetworkClassifier
from qiskit_machine_learning.circuit.library import ZFeatureMap, ZZFeatureMap
from qiskit_algorithms.optimizers import COBYLA
from qiskit_algorithms import NumPyMinimumEigensolver
from qiskit.primitives import Sampler
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

# 1. データセットの準備（月型データセットを使用）
X, y = make_moons(n_samples=50, noise=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. 量子回路の設計
num_qubits = 2
feature_map = ZZFeatureMap(feature_dimension=2, reps=2)

# Ansatz（学習可能なパラメータを持つ回路）の作成
ansatz = QuantumCircuit(num_qubits)
for i in range(num_qubits):
    ansatz.rx(Parameter(f'theta_{i}'), i)
    ansatz.ry(Parameter(f'phi_{i}'), i)

# 3. プリミティブ（Sampler）の設定
sampler = Sampler()

# 4. NeuralNetworkClassifierの設定
# 量子回路の出力を期待値として取り出し、分類器として構成
qnn = NeuralNetworkClassifier(
    feature_map=feature_map,
    ansatz=ansatz,
    sampler=sampler
)

# 5. 最適化アルゴリズムの定義
optimizer = COBYLA(maxiter=100)

# 6. 学習の実行
qnn.fit(X_train, y_train)

# 7. 精度検証
score = qnn.score(X_test, y_test)
print(f"Test Accuracy: {score}")
```

## コードの解説と動作のポイント

### Feature Mapの役割
`ZZFeatureMap`を使用しています。これは、入力データの高次な相関を量子状態のエンタングルメントとして表現するための回路です。`reps=2`とすることで、データの非線形な関係をより複雑な量子状態として埋め込みます。

### 期待値によるラベル出力
`NeuralNetworkClassifier`は、回路の実行結果（サンプリングされたビット列の統計量）から、特定の観測量（Z基底の期待値など）を計算し、それをクラスの確率として扱います。

### 最適化プロセス
ここでは古典的な最適化器として`COBYLA`を使用しています。これは勾配を必要としない（Gradient-free）アルゴリズムであり、量子回路のノイズや、量子回路特破の「Barren Plateaus（勾配消失問題）」の影響を受けにくい特性があります。

## 実装上の注意点：Barren Plateausへの対策

QNNを大規模化しようとすると、パラメータの勾配が指数関数的に消失する「Barren Plateaus」という現象に直面します。これは、回路の深さや量子ビット数が増えるほど、回路の出力が平均的な値に収束してしまい、学習が進まなくなる問題です。

回避策としては、以下のようなアプローチが有効です。
- **回路の浅い初期化**: 最初はパラメータの変動範囲を小さく設定する。
- **局所的なコスト関数の使用**: 全ての量子ビットに依存する測定ではなく、近接する量子ビットのみの期待値を計算する。
- **適切なAnsatzの選択**: データの構造に合わせた、Entanglement（絡み合い）が強すぎない回路構成にする。

## まとめ

本記事では、Qiskitを用いて量子回路をレイヤーとして扱う、量子ニューラルネットワークの基礎的な実装方法を解説しました。

量子回路を単なる計算機としてではなく、特徴抽出器（Feature Map）と学習可能な重み（Ansatz）の組み合わせとして捉えることで、古典的な機械学習の枠組みに量子的な計算能力を組み込むことが可能です。

より高度な実装や、量子・古典ハイブリッドアルゴリズムの理論については、以下のリソースも参照してください。

https://www.jqca.org/vibe-coding?utm_source=zenn&utm_medium=social&utm_campaign=soloos_post
