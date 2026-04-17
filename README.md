## Deeplearning-Test

<br>

### 開発環境

<br>

* Python 3.14.14

<br>

* ローカルに仮想環境の構築

    `.venv` は任意のものを使用する

<br>

requirements.txt

```
torch
ultralytics
numpy
matplotlib
```

<br>
<br>

### 実行方法

<br>

1. 仮想環境の構築

    ```
    python -m venv .venv
    ```

<br>

2. 仮想環境の有効化

    - Windows

        ```
        .venv\Scripts\activate
        ```

    - macOS/Linux

        ```
        source .venv/bin/activate
        ```

<br>

3. 実行

    ```
    #dataset内の画像を学習させる
    py train.py
    ```

    ```
    #学習したモデルから推論する
    py predict.py
    ```

<br>
<br>
<hr>
<br>
<br>

### License

Release under the [MIT](./LICENSE) License.

<br>
<br>

### Author

* [Sakamochanq](https://github.com/Sakamochanq)
    * [Github Copilot](https://github.com/features/copilot)
        * [Claude Haiku 4.5](https://claude.ai/)

<br>
<br>
