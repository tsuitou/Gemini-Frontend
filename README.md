![スクリーンショット 2025-03-17 111153](https://github.com/user-attachments/assets/40be2f76-a085-405b-b6d4-02bd7711f137)
 1. **リポジトリのクローンまたは ZIP ファイルの展開:**

  リポジトリをローカル環境にクローンするか、ZIP ファイルをダウンロードして展開

   ```bash
   git clone https://github.com/tsuitou/Gemini-Frontend.git
   ```

2. **仮想環境の作成:**

   Python の仮想環境を作成

   ```bash
   python -m venv venv
   ```

3. **仮想環境のアクティベート:**

   作成した仮想環境をアクティベート
   Windows環境ならvenv.batを実行するだけでいけます

   *   **Windows の場合:**

        ```bash
        venv\Scripts\activate.bat
        ```

4. **依存関係のインストール:**

   `requirements.txt` ファイルに記載された必要な Python パッケージをインストール

   ```bash
   pip install -r requirements.txt
   ```

5. **環境変数の設定:**

   `env.example` ファイルを `.env` にリネームし、必要な環境変数を設定

   ```
   GOOGLE_API_KEY=<Google API キー>
   SYSTEM_INSTRUCTION=<システムプロンプト>
   SYSTEM_INSTRUCTION_FILE=<システムプロンプトを記述したファイル名>（複数行のシステムプロンプト用。ここで指定したファイルはプロンプト送信のたびに読み込まれます）
   ```

6. **アプリケーションの起動:**

   `app.py` を実行してアプリケーションを起動します。

   ```bash
   python app.py
   ```


7. **ブラウザでアクセス:**

   Web ブラウザから `http://localhost:5000/gemini` にアクセスするとアプリが利用できます
