---
title: "PythonとQiskitで実装する量子位相推定：固有値の抽出とユニタリ演算子の解析"
emoji: "📝"
type: "tech"
topics: ["ai", "llm", "量子コンピュータ"]
published: true
---

===
量子コンピュータのアルゴリズムを設計する際、特定のユニタリ演算子の固有値、あるいはその位相情報を抽出するプロセスは極めて重要です。このプロセスを担うのが「量子位相推定（Quantum Phase Estimation: QPE）」です。

量子位相推定は、量子振幅推定や量子シャノン・サンプリング、さらにはShorのアルゴリズムの根幹を支える技術です。本記事では、PythonとQiskitを用い、実際に動作する回路構成と、位相がどのように量子ビットにエンコードされるのかを、実装を通じて解説します。

## 量子位相推定の目的

量子位相推定の目的は、未知のユニタリ演算子 $U$ と、その固有状態 $|u\rangle$ が与えられたとき、以下の関係を満たす位相 $\theta$ を求めることです。

$$U|u\rangle = e^{2\pi i \theta}|u\rangle$$

ここで、$\theta$ は $0 \le \theta < 1$ の範囲にある実数です。この $\theta$ を、制御量子ビット（ancilla qubits）のビット列として出力することが、このアルゴリズムのゴールです。

## 回路の構成要素

QPEの回路は、主に以下の3つの要素で構成されます。

1. **制御量子ビット（Precision qubits）**: 位相を格納するための量子ビット群。ビット数が多いほど、推定できる位相の精度（解像度）が高まります。
2. **ターゲット量子ビット（Target qubit）**: 固有状態 $|u\rangle$ を保持する量子ビット。
3. **制御ユニタリ演算（Controlled-$U^{2^j}$）**: 制御量子ビットの状態に応じて、ターゲット量子ビットに $U$ を繰り返し適用する操作。

## Qiskitによる実装

具体的に、位相 $\theta = 0.25$ を推定する例を実装します。
$\theta = 0.25$ は $2\pi \times 0.25 = \pi/2$ に相当します。これを実現するために、ターゲット量子ビットに $R_z(\pi/2)$ ゲートを適用する演算子 $U$ を考えます。

```python
import numpy as np
from qiskit import QuantumCircuit, Aer, execute
from qintit_utils import get_counts # 便宜上のユーティリティ

def create_qpe_circuit(precision_qubits, target_qubits):
    # 制御量子ビットとターゲット量子ビットの合計数
    total_qubits = precision_qubits + target_qubits
    qc = QuantumCircuit(total_qubits, precision_qubits)

    # 1. 制御量子ビットにアダマールゲートを適用して重ね合わせ状態を作る
    for i in range(precision_qubits):
        qc.h(i)

    # 2. ターゲット量子ビットに初期状態 |1> を用意 (固有状態の準備)
    for i in range(precision_qubits, total_qubits):
        qc.x(i)

    # 3. 制御ユニタリ演算の適用 (Controlled-U operations)
    # 今回は U = Rz(pi/2) と仮定。位相 theta = 0.25
    # 制御ビット j に対して U^(2^j) を適用する
    for j in range(precision_qubits):
        for k in range(precision_qubits, total_qubits):
            # 2^j 回の回転を、制御ビット j を使って実行
            angle = (2**j) * (2 * np.pi * 0.25)
            # 実際には回転角の制御を実現するため、cp (Controlled-Phase) ゲートを使用
            # ここでは簡略化のため cp ゲートで実装
            qc.cp(angle / (2**j), j, k) 

    # 4. 逆量子フーリエ変換 (Inverse QFT) の適用
    # 制御量子ビットに対して逆QFTを行う
    for i in reversed(range(precision_qubits)):
        for j in reversed(range(i)):
            qc.cp(-np.pi/2, j, i) # 簡略化した逆回転
        qc.h(i)
    
    return qc

# 回路の構築
precision_bits = 3 # 2^3 = 8段階の精度
target_bits = 1
qc = create_qpe_circuit(precision_bits, target_bits)

# 実行
backend = Aer.get_backend('qasm_simulator')
job = execute(qc, backend, shots=1024)
counts = job.result().get_counts()

print("測定結果（ビット列）:", counts)
```

## 実装のポイントと解説

### 制御ユニタリ演算の構造
QPEの肝は、制御ビット $j$ に対して $U^{2^j}$ を適用する点にあります。これにより、位相 $\theta$ の各ビット成分が、制御ビットの位相差として、アダマール変換後の重ね合わせ状態に「回転」として刻み込まれます。

### 逆量子フーリエ変換 (IQFT)
制御ビットにエンコードされた位相情報は、まだ「位相」の形（回転の差）として存在しています。これを、我々が読み取れる「ビット列（Z基底の確率分布）」に変換するために、量子フーリエ変換の逆操作（IQFT）が必要です。IQFTを通すことで、位相の成分が各量子ビットの $|0\rangle$ または $|1\rangle$ の確率へと変換されます。

### 精度と計算量
精度を高めるには、制御量子ビットの数を増やします。
- 精度 $n$ ビットの精度を得るには、$n$ 個の制御量子ビットが必要です。
- 制御ビットを1つ増やすごとに、推定可能な最小の位相間隔は半分になります。
- 計算量は、制御ビットの数に対して多項式時間で増加しますが、ゲートの深さは $2^n$ に依存するため、実機での実行には制約が生じます。

## 実行結果の解釈

上記コードで $\theta = 0.25$ を設定した場合、期待される結果はバイナリで $0.010$（2進数）付近になります。
3ビットの精度であれば、結果は `001` (1/8 = 0.125) や `010` (2/8 = 0.25) や `011` (3/8 = 0.375) といった、$\theta$ に最も近い値として観測されます。

もし測定結果が `010` となれば、それは $2\pi \times 0.25$ が正しく抽出されたことを意味します。

## まとめ

量子位相推定は、単なる理論上のアルゴリズムではなく、量子計算における「値の抽出」の標準的な手法です。回路設計においては、ターゲットとなる $U$ の性質と、必要な精度に応じた制御量子ビットの選定が鍵となります。

量子コンピュータのアルゴリズムを設計する際は、このように、特定の演算子の性質をどのように量子ビットの位相へとマッピングするかという視点が不可欠です。

より詳細な量子アルゴリズムの実装や、AI×量子への応用について学びたい方は、以下のリソースも参照してください。

https://jqca.org/vibe-coding
https://www.qai-zen.com/?utm_source=zenn&utm_medium=social&utm_campaign=soloos_post
