---
title: "PythonとQiskitで実装する量子分散表現：量子回路を用いたEmbeddingの構築"
emoji: "📝"
type: "tech"
topics: ["ai", "llm", "量子コンピュータ"]
published: true
---

## 導入：高次元データの量子空間へのマッピング

機械学習において、テキストや画像などの非構造化データを扱う際、その特徴を低次元かつ意味を保持したベクトルに変換する「Embedding（分散表現）」は極めて重要なプロセスです。

現在、Word2VecやBERTなどの手法は、古典的なニューラルネットワークを用いて高次元の共起関係を学習し、密なベクトルを生成します。しかし、データの次元数が爆発的に増加する現代のタスクにおいて、古典的な計算リソースでの特徴量抽出には限界が見え始めています。

ここで注目されるのが「AI×量子」のアプローチです。量子回路を用いてデータの情報を量子状態（量子ビットの振幅や位相）にエンコードすることで、古典的な手法では困難な高次元特徴空間へのマッピングを実現できる可能性があります。

本記事では、Qiskitを用いて、古典的なデータを量子回路のパラメータや回転角に変換し、量子回路そのものをEmbedding層として機能させる実装方法を解説します。

## 量子Embeddingの基本概念：振幅エンコーディングと角度エンコーディング

量子回路でデータを扱う手法には、主に以下の2つのアプローチがあります。

1. **角度エンコーディング (Angle Encoding)**
   量子ビットの回転角（$R_x, R_y, R_z$ゲートのパラメータ）に直接データを代入する方法です。実装が単純で、少ない量子ビット数で動作しますが、データの表現力は量子ビット数に依存します。

2. **振幅エンコーディング (Amplitude Encoding)**
   量子状態の確率振幅としてデータを格納する方法です。$n$個の量子ビットに対して$2^n$個の要素を持つベクトルをエンコードできるため、非常に高い圧縮率を持ちますが、回路の深さが深くなりやすいという課題があります。

今回は、実装の分かりやすさと、現在のNISQ（Noisy Intermediate-Scale Quantum）デバイスでの実行可能性を考慮し、角度エンコーディングを用いた「量子埋め込み層」の構築に焦点を当てます。

## 実装：Qiskitによる量子Embedding層の構築

以下に、PythonとQiskitを用いた、入力ベクトルを量子状態に変換する回路の実装例を示します。

```python
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

def create_quantum_embedding_layer(num_qubits, input_dim):
    """
サーキットの構築
    num_qubits: 使用する量子ビット数
    input_dim: 入力データの次元数
    """
    # パラメータ化された回路の作成
    # 入力データの各次元を、回転ゲートの角度として扱う
    params = [Parameter(f'theta_{i}') for i in range(input_dim)]
    qc = QuantumCircuit(num_qubits)

    # 角度エンコーディングの実装
    # 各入力値をRyゲートの回転角に割り当てる
    for i in range(input_dim):
        # 0番目の量子ビットから順に、入力値を回転角として適用
        # 量子ビット数を超えない範囲で適用
        target_qubit = i % num_qubits
        qc.ry(params[i], target_qubit)

    # Entanglement（絡み合い）層の追加
    # データの次元間の相関を表現するためにCNOTゲートを適用
    for i in range(num_qubits - 1):
        qc.cx(i, i + 1)

    return qc, params

# 設定
n_qubits = 3
input_size = 3

# 回路の生成
embedding_circuit, input_params = create_quantum_embedding_layer(n_qubits, input_size)

# 入力データの準備 (例: 正規化された3次元ベクトル)
input_data = np.array([0.1, 0.5, 0.9])

# パラメータに値を代入
bound_circuit = embedding_circuit.assign_parameters(input_data)

# 回路の可視化（コンソール出力）
print("Quantum Embedding Circuit:")
print(bound_circuit.draw(output='text'))
```

### コードの解説

1. **`Parameter` の利用**:
   学習可能な重みとして扱うために、Qiskitの`Parameter`オブジェクトを使用しています。これにより、後のステップで勾配計算（バックプロパゲーション）を行うことが可能になります。

2. **`ry` ゲートによるエンコーディング**:
   `qc.ry(params[i], target_qubit)` により、入力値 `input_data[i]` を量子ビットの回転角として適用しています。これにより、古典的な数値が量子的な位相の差として表現されます。

3. **CNOTゲートによる相関の導入**:
   単に各量子ビットを回転させるだけでは、量子ビット間に相互作用がありません。`qc.cx(i, i+1)` を追加することで、量子もつれ（Entanglement）を生じさせ、入力データの各次元が絡み合った高次元な特徴空間を構築しています。

## 量子Embeddingの評価：状態の検証

構築した回路が、正しく入力データを反映した量子状態を生成しているかを確認するため、期待値の測定を行います。

```python
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector

# シミュレータの準備
simulator = AerSimulator()

# 実行：作成した回路にデータを流し込み、状態ベクトルを取得
# 角度エンコーディングの結果としての状態ベクトルを確認
state = Statevector.from_instruction(bound_circuit)

print("Generated Quantum State Vector:")
print(state.data)

# 確率分布の確認
probabilities = state.probabilities()
for i, prob in enumerate(probabilities):
    print(f"State |{format(i, '03b')}>: {prob:.4f}")
```

この実行結果において、入力値 `[0.1, 0.5, 0.9]` が、各基底状態の確率振幅に反映されていることが確認できます。

## 今後の展望：Hybrid Quantum-Classical Approach

今回実装した量子Embedding層は、単体で動作するものではなく、古典的なニューラルネットワークと組み合わせた「ハイブリッド型」の構成を想定しています。

具体的には、以下のようなパイプラインです：
1. **Classical Input**: テキストや画像から特徴量を抽出（既存のCNNやTransformerを使用）。
2. **Quantum Embedding Layer**: 抽出された特徴量を量子回路のパラメータとして入力し、量子状態へ変換。
3. **Quantum Layer**: 量子回路内で非線形な変換（Entanglementによる相互作用）を適用。
4. **Measurement**: 量子ビットの期待値（Z基底の測定値など）を古典的な値として取り出す。
5. **Classical Output**: 取り出した値を、最終的な分類や回帰を行う全結合層へ渡す。

このように、量子回路を「高次元特徴抽出器」として扱うことで、古典的な計算では捉えきれないデータ内の複雑な相関を、量子的な重ね合わせと絡み合いを利用して表現することが期待されます。

量子コンピュータの実機性能が向上するにつれ、このEmbeddingプロセスが大規模なデータセットに対してどのような計算量優位性（あるいは表現力の向上）をもたらすのか、さらなる研究が進められています。

さらに詳細な量子回路の実装や、量子機械学習の理論については、こちらのサイトも参考にしてください。
https://www.jqca.org/vibe-coding?utm_source=zenn&utm_medium=social&utm_campaign=soloos_post
