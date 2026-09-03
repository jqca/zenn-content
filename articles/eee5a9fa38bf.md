---
title: "PythonとQiskitで実装する量子振幅推定：期待値の算出から確率の推定まで"
emoji: "📝"
type: "tech"
topics: ["ai", "llm", "量子コンピュータ"]
published: true
---

量子コンピュータのアルゴリズムの中でも、特に実用的な応用範囲が広いものの一つに「量子振幅推定（Quantum Amplitude Estimation: QAE）」があります。これは、量子回路の測定結果から得られる特定のビット列（例えば「1」）の出現確率を、量子的な重ね合わせ状態を利用して効率的に推定する手法です。

量子振幅推定の基本は、ある状態の確率振幅を、量子位相推定（QPE）の仕組みを用いて抽出することにあります。本記事では、Qiskitを用いて、特定の量子状態の出現確率を推定する基本的な回路の実装手順を解説します。

## 量子振幅推定の原理

量子振幅推定の目的は、量子回路 $A$ を適用した後の状態 $|\psi\rangle = A|0\rangle$ において、特定のターゲット状態 $|1\rangle$ が含まれる確率 $a$ を推定することです。

具体的には、以下のプロセスを辿ります。

1. 状態の準備：回路 $A$ によって、ターゲット状態への確率振幅 $a$ を持つ状態を作成する。
2. 演算子の反転：$A$ の結果に対して、ターゲット状態のビットを反転させる演算（Oracle）を適用する。
3. Grover演算の反転：拡散演算子（Diffuser）を適用し、振幅を増幅させる。
4. 位相の抽出：この一連の反転プロセスを繰り返すことで、回転角 $\theta$ （ここで $a = \sin^2(\theta)$）を量子位相推定によって計算する。

最終的に、得られた $\theta$ から元の確率 $a$ を逆算します。

## Qiskitによる実装手順

以下に、単純な確率振幅を推定するための実装例を示します。ここでは、あらかじめ定義された確率を持つ状態をシミュレーションし、その値を推定するプロセスを記述します。

```python
import numpy as np
from qiskit import QuantumCircuit
from qiskit_algorithms import AmplitudeEstimation, EstimationProblem
from qiskit_aer import AerSimulator
from qiskit.primitives import Sampler

# 1. ターゲットとなる確率の設定
# ここでは、推定したい「正解」の確率を 0.3 と設定します
target_probability = 0.3

# 2. 状態準備回路の作成
# 1量子ビットの回路において、確率 target_probability を持つ状態を作成する
qc = QuantumCircuit(1)
theta = np.arcsin(np.sqrt(target_probability))
qc.ry(2 * theta, 0)

# 3. EstimationProblem の定義
# 準備した回路 qc を用いて、どの状態（今回は |1>）の確率を推定するかを指定します
problem = EstimationProblem(
    state_preparation=qc,
    observable=None # デフォルトで Z 演算子の期待値（ビット1の確率）を対象とする
)

# 4. 量子振幅推定アルゴリズムの構成
# シミュレータ（Sampler）とアルゴリズムのインスタンスを作成
sampler = Sampler()
ae = AmplitudeEstimation(sampler=sampler)

# 5. 実行と結果の取得
result = ae.estimate(problem)

# 6. 結果の出力
estimated_probability = result.estimation
print(f"設定した確率: {target_probability}")
print(f"推定された確率: {estimated_probability}")
print(f"誤差（差分）: {abs(target_probability - estimated_probability)}")
```

## コードの解説

### 状態準備回路（State Preparation）
`qc.ry(2 * theta, 0)` の部分が重要です。回転ゲート $R_y$ を用いることで、基底状態 $|0\rangle$ から、振幅が $\cos(\theta)$ と $\sin(\theta)$ になる状態へと遷移させています。この $\sin^2(\theta)$ が、私たちが推定したい確率 $a$ となります。

### EstimationProblem の役割
`EstimationProblem` は、推定したい「状態の準備（state_preparation）」と、その「観測対象（observable）」を紐付ける役割を持ちます。Qiskitのアルゴリズムでは、これを用いることで、どのビットのどの状態に注目して振幅を計算すべきかを明示的に定義できます。

### アルゴリズムの実行
`AmplitudeEstimation` クラスは、内部的に量子位相推定（QPE）の構造を利用しています。`Sampler`（基本的には `Sampler` または `Sampler` を拡張したシミュレータ）を渡すことで、量子回路の測定結果から振幅の推定値を計算します。

## 実行結果の考察

上記のコードを `AerSimulator` 等で実行すると、設定した `target_probability` (0.3) に近い値が出力されます。

ただし、注意点があります。量子振幅推定の精度は、反転操作（Grover iteration）を何回繰り返すか、つまり「補助量子ビット（ancilla qubits）の数」に依存します。補助量子ビットを増やすほど、より細かな角度 $\theta$ を分解できるため、推定精度は向上しますが、回路の深さと量子リソースの消費量が増大します。

## まとめ

量子振幅推定は、量子計算のメリットを直接的に確率の推定という形で得られる強力な手法です。実装のポイントは、以下の3点に集約されます。

1. ターゲットとなる確率 $a$ を持つ回路 $A$ を作成する。
2. $A$ に対応する回転角 $\theta$ を抽出するための枠組み（EstimationProblem）を用意する。
3. 補助量子ビットの数に応じた精度の限界を理解して、回路規模を設計する。

この技術は、金融におけるオプション価格の計算や、量子化学における分子のエネルギー計算など、今後のAI×量子分野における具体的な計算アルゴリズムの基盤となります。

さらに詳しく量子アルゴリズムの実装について学びたい方は、以下のリソースも参考にしてください。

https://jqca.org/vibe-coding
https://www.qai-zen.com/
