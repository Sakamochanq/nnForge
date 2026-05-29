<div align="center">
    <h2>Neural Network Forge</h2>
    <p>画像解析・深層学習を行うための開発環境 🐉</p>
</div>

<br>

<h3>概要</h3>

Neural Network Forge (以下 nnForge) は、私が研究で使用している画像解析・深層学習のための開発環境を保管しておくためのリポジトリです。
他者の環境での再現性を保証するものではありませんが、私の環境を共有することで、同様の環境を構築する際の参考になればと思います。このREADMEでは、
nnForgeの構成や使用方法について説明しています。これは後から自分で見たときに理解できるようにするためでもあります。


<br>
<br>

<h3>技術スタック</h3>

<div>
  <table>
    <thead>
      <tr>
        <th>項目</th>
        <th>技術</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>開発言語</td>
        <td>Python</td>
      </tr>
      <tr>
        <td>フレームワーク</td>
        <td>PyTorch, TensorFlow</td>
      </tr>
      <tr>
        <td>ソフトウェア</td>
        <td>VSCode, PowerShell</td>
      </tr>
      <tr>
        <td>ツール</td>
        <td>git, FFmpeg</td>
      </tr>
    </tbody>
  </table>
</div>

<br>
<br>

<h3>環境構築</h3>


1. Pythonのインストール

    以下のコマンドをターミナルで実行し、Python環境が正しくインストールされていることを確認してください。

    ```bash
    Python --version
    # Python 3.10.12
    ```

<br>

2. リポジトリのクローン

    以下のコマンドをターミナルで実行し、nnForgeリポジトリをクローンしてください。  
    gitがインストールされていない場合は、`Download Zip`から直接ダウンロードしてください。

    ```bash
    git clone https://github.com/Sakamochanq/nnForge.git
    cd nnForge
    ```

<br>

3. 仮想環境の構築と依存関係のインストール

    以下のコマンドをターミナルで実行し、仮想環境を構築し、必要な依存関係をインストールしてください。

    ```bash
    # 仮想環境の構築
    python -m venv .venv
    .venv\Scripts\activate

    # 依存関係のインストール
    pip install -r requirements.txt
    ```

<br>

4. 環境変数の設定

    こちらの開発環境では `SDNET2018` という、AIや機械学習の技術を使ってコンクリートのひび割れや欠陥を検出・分類する研究用画像データセットを使用しています。
    `./src/assets/dataset/SDNET2018` にデータセットを配置します。
    その他の環境変数は `./src/config.py` に記載されているものを適宜変更します。

    ```py
    class config:
    
    # 学習させるデータセット
    dataset = "D:\\Enviroments\\nnForge\\src\\dataset\\images\\SDNET2018\\W";
    
    #画像サイズ
    img_size = 224;
    
    # バッチサイズ
    batch_size = 32;
    
    # 学習回数
    epochs = 30;
    
    # 学習率
    learning_rate = 0.001;
    
    #学習モデルの保存先
    model = "Model.pth";
    ```

<br>

5. 実行

    以下のコマンドをターミナルで実行し、学習と推論を実行してください。

    ```bash
    # 学習の実行
    python main.py

    # 推論の実行
    python runner.py
    ```

<br>
<br>