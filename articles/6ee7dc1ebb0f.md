---
title: "PythonとQiskitで実装する量子重み付きサンプリング：確率分布の制御と実装"
emoji: "📝"
type: "tech"
topics: ["ai", "llm", "量子コンピュータ"]
published: true
---

量子コンピュータのアルゴリズムを設計する際、特定の確率分布に従って量子状態を生成したり、測定結果の分布を操作したりする技術は極めて重要です。特に、量子ビットの各状態に対して重み（確率）を割り当て、意図した分布からサンプリングを行う手法は、量子機械学習における特徴量変換や、量子生成モデルの構築において基礎的なステップとなります。

本記事では、PythonとQiskitを用いて、量子ビットの各状態に対して任意の重みを割り当てる「量子重み付きサンプリング」の回路実装について解説します。

## 量子重み付きサンプリングの考え方

量子重み付きサンプリングの目的は、$n$ 個の量子ビットを用いて、$2^n$ 個の基底状態に対して、あらかじめ定義された確率分布 $P = \{p_0, p_1, ..., p_{2^n-1}\}$ に従うように回路を構成することです。

単純な一様分布（すべての状態が等確率）であれば、アダマールゲート（Hゲート）を各量子ビットに適用するだけで実現できます。しかし、特定の状態の出現確率を高めたり低めたりしたい場合、回転ゲート（$R_y$ ゲートなど）を用いて、各量子ビットの回転角を調整し、振幅を制御する必要があります。

## 実装のステップ

今回は、2量子ビットを用いた例を考えます。全4状態（$|00\rangle, |01\rangle, |10\rangle, |11\rangle$）に対し、以下のターゲットとする確率分布を考えます。

- $P(00) = 0.5$
- $P(01) = 0.25$
- $P(10) = 0.15$
- $P(11) = 0.1$

この分布を実現するためには、上位の量子ビットから順に、そのビットが $0$ である確率と $1$ である確率を分岐させるようにゲートを配置していくアプローチが有効です。

1. 第1量子ビット（$q_0$）において、測定結果が $0$ になる確率を $P(q_0=0) = P(00) + P(01) = 0.75$ となるように $R_y$ ゲートを適用する。
2. 第2量子ビット（$q_1$）において、条件付き確率 $P(q_1=0 | q_0=0) = P(00) / P(q_0=0) = 0.5 / 0.75 = 2/3$ となるように制御 $R_y$ ゲートを適用する。
3. 同様に、$q_0=1$ のとき、 $P(q_1=0 | q_0=1) = P(10) / P(q_0=1) = 0.15 / 0.25 = 0.6$ となるように制御 $R_y$ ゲートを適用する。

## Qiskitによる実装コード

以下のコードでは、上記の設計に基づき、任意の確率分布に近い重み付けを行う回路を構築し、実際にシミュレータで実行して分布を確認します。

```python
import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskint.visualization import plot_histogram

def create_weighted_circuit(probabilities):
    """
    2量子ビットの確率分布を指定して回路を生成する
    probabilities: [p00, p01, p10, p11] のリスト
    """
    n_qubits = 2
    qc = QuantumCircuit(n_qubits)
    
    # 状態の計算
    p00, p01, p10, p11 = probabilities
    
    # q0の重み付け: P(q0=0) = p00 + p01
    p_q0_is_0 = p00 + p01
    # Ry(theta) で cos(theta/2)^2 = p_q0_is_0 となるthetaを求める
    theta0 = 2 * np.arccos(np.sqrt(p_q0_is_0))
    qc.ry(theta0, 0)
    
    # q1の重み付け (q0=0 のとき): P(q1=0 | q0=0) = p00 / (p00 + p01)
    p_q1_is_0_given_q0_0 = p00 / p_q0_is_0
    theta1_if_0 = 2 * np.arccos(np.sqrt(p_q1_is_0_given_q0_0))
    qc.cry(theta1_if_0, 0, 1)
    
    # q1の重み付け (q0=1 のとき): P(q1=0 | q0=1) = p10 / (p10 + p11)
    p_q1_is_0_given_q0_1 = p10 / (p10 + p11)
    theta1_if_1 = 2 * np.arccos(np.sqrt(p_q1_is_0_given_q0_1))
    # q0=1のときに回転させるため、Xゲートで反転させてからCRY、その後Xで戻す
    qc.x(0)
    qc.cry(theta1_if_1, 0, 1)
    qc.x(0)
    
    qc.measure_all()
    return qc

# 目標とする確率分布
target_probs = [0.5, 0.25, 0.15, 0.1]

# 回路の生成
circuit = create_weighted_circuit(target_probs)

# シミュレータでの実行
simulator = AerSimulator()
job = simulator.run(circuit, shots=1024)
result = job.result()
counts = result.get_counts()

# 結果の表示
print("Target Probabilities:", target_probs)
print("Measured Counts:", counts)

# 割合に変換して比較用に出力
measured_probs = {k: v/1024 for k, v in counts.items()}
print("Measured Probabilities:", measured_probs)
```

## 実装のポイント：回転角の導出

$R_y(\theta)$ ゲートは、状態 $|0\rangle$ を以下の状態に変換します。
$$\cos(\theta/2)|0\rangle + \sin(\theta/2)|1\rangle$$

このとき、測定結果が $|0\rangle$ となる確率は $\cos^2(\theta/2)$、 $|1\rangle$ となる確率は $\sin^2(\theta/2)$ です。
したがって、目標とする確率 $p$ に対して、回転角 $\theta$ は $\theta = 2 \cdot \arccos(\sqrt{p})$ として算出できます。

このロジックを再帰的に適用することで、より多ビットの量子ビットに対しても、ツリー構造のような回路設計によって任意の分布（離散的な分布）を近似的に生成することが可能です。

## まとめ

本記事では、PythonとQiskitを用い、特定の確率分布に従うように量子ビットの状態を操作する回路の実装方法を解説しました。

制御 $R_y$ ゲートを用いることで、上位ビットの測定結果（状態）に応じて下位ビットの回転角を切り替え、条件付き確率を制御できる点がポイントです。この手法は、量子アルゴリズムにおけるサンプリング問題や、量子生成モデルの基礎的な構成要素となります。

AI×量子の研究開発において、こうした低レイヤーの回路操作の理解は、モデルの振る舞いを精密に制御するために不可欠です。

さらなる量子アルゴリズムの実装に関心がある方は、こちらのリソースも参考にしてください。
https://jqca.org/v1-coding
https://www.qai-zen.com/?utm_source=zenn&utm_medium=social&utm_campaign=soloos_post
