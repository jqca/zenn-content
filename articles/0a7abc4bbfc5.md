---
title: "PythonとQiskitで実装する量子ウォード・モデル：量子ビットの状態遷移と期待値の計算"
emoji: "📝"
type: "tech"
topics: ["ai", "llm", "量子コンピュータ"]
published: true
---

## 導入：量子状態の遷移を制御する
量子コンピュータのアルゴリズムを理解する上で、単一の量子ゲートの操作だけでなく、量子ビットの状態がどのように遷移し、最終的にどのような確率分布として観測されるかを追跡することは極めて重要です。

本記事では、量子ビットに特定の回転演算（Rotation）を適用し、その状態遷移をシミュレーションする「量子ウォード・モデル（簡易的な状態遷移モデル）」をPythonとQiskitを用いて実装します。量子ビットの回転角をパラメータとして入力し、それによって生じる期待値の変化を、実際の実行結果から確認するプロセスを解説します。

## 実装の設計
今回の実装では、以下のステップを踏みます。

1.  Qiskitを用いて、量子ビットに特定の角度 $\theta$ の回転ゲート（$RY$ゲート）を適用する回路を作成します。
2.  初期状態 $|0\rangle$ に対して、回転によって生じる状態 $| \psi \rangle = \cos(\theta/2)|0\rangle + \sin(\theta/2)|1\rangle$ を生成します。
3.  この回路を複数回の試行（Shots）で実行し、状態 $|1\rangle$ が観測される確率（期待値）を算出します。
4.  $\theta$ を変化させたときの、測定結果の確率分布の変化を可視化します。

## Pythonによる実装コード

```python
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram

def run_quantum_transition_simulation(theta_range, shots=1024):
    """
    指定された角度範囲に対して、RYゲート適用後の状態遷移をシミュレーションする
    """
    simulator = AerSimulator()
    results = []
    probabilities = []

    for theta in theta_range:
        # 1. 量子回路の構築
        # 1量子ビットの回路。初期状態は |0>
        qc = QuantumCircuit(1, 1)
        
        # RYゲート（Y軸周りの回転）を適用
        qc.ry(theta, 0)
        
        # 測定
        qc.measure(0, 0)

        # 2. シミュレーションの実行
        job = simulator.run(qc, shots=shots)
        result = job.result()
        counts = result.get_counts()

        # 3. 結果の集計
        # 状態 |1> が観測された割合を計算
        prob_1 = counts.get('1', 0) / shots
        
        results.append(theta)
        probabilities.append(prob_1)

    return results, probabilities

# 実行パラメータの設定
# 0からPIまでの範囲で、20個のステップを作成
theta_values = np.linspace(0, np.pi, 20)
shots_count = 2048

# シミュレーション実行
thetas, probs = run_quantum_transition_simulation(theta_values, shots=shots_count)

# 4. 結果の可視化
plt.figure(figsize=(10, 6))
plt.plot(thetas, probs, 'o-', label='Simulated Probability (State |1>)')
plt.plot(thetas, np.sin(thetas/2)**2, '--', label='Theoretical Probability', color='red')
plt.xlabel('Rotation Angle theta (radians)')
plt.ylabel('Probability of measuring |1>')
plt.title('Quantum State Transition: RY Gate rotation')
plt.legend()
plt.grid(True)
plt.show()
```

## 実装の解説

### 量子回路の構成
`qc.ry(theta, 0)` がこのモデルの核心です。量子ビットに対してY軸周りの回転を与えています。
初期状態 $|0\rangle$ に対し、この操作によって状態は以下のように変化します。
$|0\rangle$ → $\cos(\theta/2)|0\rangle + \sin(\theta/2)|1\rangle$

このとき、測定によって状態 $|1\rangle$ が得られる理論的な確率は $\sin^2(\theta/2)$ となります。

### シミュレータと試行回数
ここでは `AerSimulator` を使用しています。
`shots=2048` と設定することで、統計的な誤差（ショットノイズ）を抑えつつ、理論値（赤い破線）に対してシミュレーション値（青い点）が追従していることを確認できるようにしています。

## 実行結果の考察
実行されたグラフでは、$\theta = \pi$（180度）のときに、確率が最大値 1.0 に達することが確認できます。これは、量子ビットが $|0\rangle$ から $|1\rangle$ へ完全に反転したことを意味します。

逆に $\theta = 0$ のときは、確率 0 となり、状態は $|0\rangle$ のまま維持されます。この「角度（パラメータ）によって状態遷移の確率を制御できる」という性質は、量子アルゴリズムにおけるパラメータ化量子回路（PQC）の基礎的な考え方そのものです。

## まとめ
本記事では、Qiskitを用いて量子ビットの回転と状態遷移をシミュレーションする実装を行いました。
- $RY$ ゲートによる量子状態の制御
- 測定結果の集計による期待値の算出
- 理論値と実測値（シミュレーション値）の比較

このプロセスを理解することは、より複雑な量子アルゴリズム（量子振幅推定や量子機械学習など）において、回路のパラメータが最終的な出力にどのような影響を与えるかを解析する際の土台となります。

さらに高度な量子計算の仕組みに興味がある方は、以下のリソースも参考にしてください。
https://www.jqca.org/vibe-coding?utm_source=zenn&utm_medium=social&utm_campaign=soloos_post
https://www.qai-zen.com/?utm_source=zenn&utm_medium=social&utm_campaign=soloos_post
