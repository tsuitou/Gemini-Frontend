![スクリーンショット 2025-03-17 111153](https://github.com/user-attachments/assets/40be2f76-a085-405b-b6d4-02bd7711f137)
1. **リポジトリのクローンまたは ZIP ファイルの展開:**

   まず、このリポジトリをローカル環境にクローンするか、ZIP ファイルをダウンロードして展開します。

   ```bash
   git clone <リポジトリのURL>
   ```

2. **仮想環境の作成:**

   Python の仮想環境を作成します。これにより、プロジェクトに必要な依存関係を隔離することができます。

   ```bash
   python -m venv venv
   ```

3. **仮想環境のアクティベート:**

   作成した仮想環境をアクティベートします。

   *   **Windows の場合:**

        ```bash
        venv\Scripts\activate.bat
        ```

   *   **macOS および Linux の場合:**

        ```bash
        source venv/bin/activate
        ```

4. **依存関係のインストール:**

   `requirements.txt` ファイルに記載された必要な Python パッケージをインストールします。

   ```bash
   pip install -r requirements.txt
   ```

5. **環境変数の設定:**

   `env.example` ファイルを `.env` にリネームし、必要な環境変数を設定します。

   ```
   GOOGLE_API_KEY=<あなたの Google API キー>
   SYSTEM_INSTRUCTION=<システム命令>
   ```

6. **アプリケーションの起動:**

   `app.py` を実行してアプリケーションを起動します。

   ```bash
   python app.py
   ```

7. **ブラウザでアクセス:**

   Web ブラウザから `http://localhost:5000/gemini` にアクセスして、チャットアプリケーションを使用します。
