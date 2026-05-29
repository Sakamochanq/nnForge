## Deeplearning-Test

<br>

### usage

<br>

git clone
```python
git clone https://github.com/Sakamochanq/DeepLearning-test.git
```

<br>

change branch
```python
git checkout -b "tensorflow"
```

<br>

setup virtual environment
```python
python -m venv .venv

# 仮想環境の有効化
.venv/bin/activate

# パッケージのインストール
pip install -r requirements.txt
```

<br>

create `src/assets/config.py`
```python
ni src/assets/config.py
```

config.py
```python
class config:
    
    # 学習させるデータセット
    dataset = "C:\\Users\\YourName\\Dataset";
    
    #画像サイズ
    img_size = 224;
    
    # バッチサイズ
    batch_size = 32;
    
    # 学習回数
    epochs = 100;
    
    # 学習率
    learning_rate = 0.001;
    
    #学習モデルの保存先
    model = "Model.pth";
```

<br>

Run
```python
cd src

# 学習
python main.py

# 推論
python runner.py
```

<br>
<br>

### License

Release under the [MIT](./LICENSE) License.

<br>

### Author

* [Sakamochanq](https://github.com/Sakamochanq)
    * [Github Copilot](https://github.com/features/copilot)
        * [Claude Haiku 4.5](https://claude.ai/)

<br>
<br>
