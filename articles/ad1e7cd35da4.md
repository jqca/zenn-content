---
title: "PythonとQiskitで実装する量子カーネル法：量子特徴写像による高次元空間へのマッピング"
emoji: "📝"
type: "tech"
topics: ["ai", "llm", "量子コンピュータ"]
published: true
---

## 導入
量子コンピュータの計算資源を、古典的な機械学習のアルゴリズムに組み込む手法として、量子カーネル法（Quantum Kernel Methods）が注目されています。

量子コンピュータは、量子ビットの重ね合わせや量子もつれを利用して、古典的なコンピュータでは扱うことが困難な高次元のヒルベルト空間にデータをマッピングする能力を持っています。この「量子特徴写像」を、カーネル法（Support Vector Machineなど）のカーネル関数として利用することで、古典的な手法では分離困難なデータに対しても、量子的な特徴空間を用いて分類タエスクを解くことが可能になります。

本記事では、Qiskitを用いて、量子カーネル法をどのように実装し、どのようにデータの非線形な境界を学習させるかを、具体的なコードとともに解説します。

## 量子カーネル法の仕組み
カーネル法におけるカーネル関数 $K(x, y)$ は、2つのデータ点 $x$ と $y$ の類似度を、高次元の特徴空間における内積として定義します。

$$K(x, y) = \langle \phi(x) | \phi(y) \rangle$$

量子カーネル法では、量子回路（量子特徴写像） $\phi$ を用いて、古典データ $x$ を量子状態 $|\phi(x)\rangle$ に変換します。このとき、量子回路の出力を測定し、その重なり具合（内積）を計算することで、量子カーネル $K(x, y) = |\langle \phi(x) | \phi(y) \rangle|^2$ を推定します。

この手法の鍵は、パラメータ化された量子回路（PQC）を用いて、入力データに依存した回転ゲートの角度などを制御する点にあります。

## 実装：量子カーネルによる分類タスク
ここでは、Qiskitを用いて、2次元の分類問題を解くための量子カーネルを実装します。

### 1. 環境準備
必要なライブラリをインライブラリをインポートします。

```python
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_algorithms.utils import algorithm_globals
from qiskit_ibm_runtime import Sampler
from sklearn.datasets import make_moons
from sklearn.svm import SVC
```

### 2. 量子特徴写像（Feature Map）の定義
データを量子状態に変換するための回路を定義します。ここでは、入力データを $R_y$ ゲートの回転角にマッピングする単純な回路を構築します。

```python
def create_feature_map(num_qubits):
    feature_map = QuantumCircuit(num_qubits)
    for i in range(num_qubits):
        # 各量子ビットにデータをエンコードするためのパラメータ
        param = Parameter(f'x_{i}')
        feature_map.ry(param, i)
    return feature_map

num_qubits = 2
feature_map = create_feature_map(num_qubits)
print(feature_map.decompose().draw())
```

### 3. 量子カーネルの構築と学習
Scikit-learnの `SVC`（Support Vector Classifier）に、Qiskitの `FidelityQuantumKernel` を組み込みます。

```python
# 1. データセットの生成（半月状のデータ）
X, y = make_moons(n_samples=40, noise=0.1, random_state=42)

# 2. 量子カーネルの準備
# 測定を行うためのSamplerを定義
sampler = Sampler()
kernel = FidelityQuantumKernel(feature_map=feature_map, sampler=sampler)

# 3. SVMモデルの構築
# 量子カーネルをカーネル行列として使用
qsvc = SVC(kernel=kernel.evaluate)

# 4. 学習の実行
qsvc.fit(X, y)

# 5. 予測
predictions = qsvc.predict(X)
```

### 4. 結果の可視化
学習結果をプロットして、決定境界がどのように形成されたかを確認します。

```python
def plot_decision_boundary(X, y, model, feature_map):
    h = .2  # メッシュグリッドのステップサイズ
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # グリッド上の点に対して予測を行う
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, cmap=plt.cm.RdBu, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.RdBu)
    plt.title("Quantum Kernel SVM Decision Boundary")
    plt.show()

plot_decision_boundary(X, y, qsvc, feature_map)
```

## 実装のポイント
この実装における重要な点は、`FidelityQuantumKernel` の役割です。このクラスは、与えられたデータ点ペアに対して量子回路を実行し、その出力（量子状態の忠実度、つまり内積の二乗）を計算してカーネル行列 $K$ を生成します。

古典的な SVM は、この $K$ を受け取るだけで、あとは従来のアルゴティングをそのまま利用できるため、量子アルゴリズムを機械学習ワークフローに統合する際の強力なインターフェースとなります。

## 考察と今後の課題
本実装では $R_y$ ゲートのみを用いた単純なエンコーディングを行いましたが、より複雑なデータ構造（非線形性が強いデータ）に対しては、エンタングルメント（CNOTゲートなど）を含む、より深い量子回路を用いた特徴写像が必要です。

しかし、回路が深くなると、量子ノイズの影響（量子ビットのデコヒーレンスやゲートエラー）により、カーネル値の推定精度が低下します。実機での運用を見据える場合は、量子エラー抑制技術の適用や、ノイズ耐性の高い特徴写像の設計が不可欠なステップとなります。

AI×量子の可能性を広げるため、より高度な量子カーネルの設計に挑戦していきましょう。

詳細な実装テクニックや、量子計算の基礎については、以下のリソースも参考にしてください。
https://www.jqca.org/vibe-coding?utm_source=zenn&utm_medium=social&utm_campaign=soloos_post
https://www.qai-zen.com/?utm_source=zenn&utm_medium=social&utm_campaign=soloos_post
