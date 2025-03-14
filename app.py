# gevent を使う場合のモンキーパッチ（WSGIサーバを gevent にするため）
from gevent import monkey
monkey.patch_all()

import os
import json
import bcrypt
import hashlib
import re
import base64
import time
import sqlite3
import joblib
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from flask_socketio import SocketIO, emit
from google import genai
from google.genai import _transformers as t
from google.genai import types 
from google.genai.types import Tool, GenerateContentConfig, GoogleSearch, ToolCodeExecution
from dotenv import load_dotenv
from pathlib import Path
from filelock import FileLock

# -----------------------------------------------------------
# 1) Flask + SocketIO の初期化
# -----------------------------------------------------------
app = Flask(__name__)
app.jinja_env.variable_start_string = '(('
app.jinja_env.variable_end_string = '))'
socketio = SocketIO(app, async_mode="gevent", cors_allowed_origins="*", max_http_buffer_size=20 * 1024 * 1024,)

# 環境変数の読み込み
load_dotenv()
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable not set")
client = genai.Client(api_key=GOOGLE_API_KEY)

code_execution_tool = Tool(
    code_execution=ToolCodeExecution()
)

google_search_tool = Tool(
    google_search=GoogleSearch()
)

MODELS = os.environ.get("MODELS", "").split(",")
SYSTEM_INSTRUCTION = os.environ.get("SYSTEM_INSTRUCTION")
VERSION = os.environ.get("VERSION")


# -----------------------------------------------------------
# 2) SQLite 用の初期設定
# -----------------------------------------------------------
DB_FILE = "data/database.db"
os.makedirs("data/", exist_ok=True)  # data/ フォルダがなければ作成

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS accounts (
        username TEXT PRIMARY KEY,
        password TEXT,
        auto_login_token TEXT
    )
    """)
    conn.commit()
    conn.close()

def generate_auto_login_token(username: str, version_salt: str):
    raw = (username + version_salt).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()

def hash_password(password):
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

def verify_password(password, hashed):
    return bcrypt.checkpw(password.encode(), hashed.encode())

def register_user(username, password):
    """新規ユーザー登録"""
    if username == "" or password == "":
        return {"status": "error", "message": "ユーザー名かパスワードが空欄です"}
    # 英数字以外の文字がないかチェック
    if any(not re.match(r"^[a-zA-Z0-9]*$", field) for field in (username, password)):
        return {"status": "error", "message": "英数字以外の文字が含まれています。"}

    # すでに同名ユーザーが存在するかチェック
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT username FROM accounts WHERE username=?", (username,))
    existing = c.fetchone()
    if existing:
        conn.close()
        return {"status": "error", "message": "既存のユーザー名です。"}

    # 挿入
    hashed_pw = hash_password(password)
    c.execute("INSERT INTO accounts (username, password) VALUES (?, ?)", (username, hashed_pw))
    conn.commit()
    conn.close()
    return {"status": "success", "message": "登録完了"}

def authenticate(username, password):
    """ユーザー認証"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT password FROM accounts WHERE username=?", (username,))
    row = c.fetchone()
    conn.close()
    if row:
        hashed_pw = row[0]
        return verify_password(password, hashed_pw)
    return False

# ------------------------
# 認証 (SQLite)
# ------------------------
@socketio.on("register")
def handle_register(data):
    username = data.get("username")
    password = data.get("password")
    result = register_user(username, password)
    emit("register_response", result)

@socketio.on("login")
def handle_login(data):
    username = data.get("username")
    password = data.get("password")
    if authenticate(username, password):
        # 認証成功
        # ここでauto_login_tokenを生成してDBに保存し、クライアントに返す
        version_salt = VERSION  # 適宜、環境変数 or DBで管理してもOK
        auto_login_token = generate_auto_login_token(username, version_salt)
        # DBに保存
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("UPDATE accounts SET auto_login_token=? WHERE username=?", (auto_login_token, username))
        conn.commit()
        conn.close()

        # クライアントには username ではなく auto_login_token を返す
        emit("login_response", {
            "status": "success",
            "username": username,  # UI表示用にユーザー名も返すことは可能
            "auto_login_token": auto_login_token
        })
    else:
        emit("login_response", {"status": "error", "message": "ログイン失敗"})

@socketio.on("auto_login")
def handle_auto_login(data):
    token = data.get("token", "")

    # 1) まず、DBから「auto_login_token == token」なユーザを探す
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT username, auto_login_token FROM accounts WHERE auto_login_token = ?", (token,))
    row = c.fetchone()
    conn.close()

    if row:
        username, stored_token = row
        # 2) 現在のVERSIONで再ハッシュしたトークンを計算
        new_hash = generate_auto_login_token(username, VERSION)  # username + "v2" をSHA256など

        # 3) DBに保存されているトークンと合うか確認
        if new_hash == stored_token:
            # 一致 => 自動ログイン成功
            emit("auto_login_response", {
                "status": "success",
                "username": username,
                "auto_login_token": stored_token
            })
        else:
            # 不一致 => バージョンが変わって旧トークンが合わなくなった or 改ざん
            emit("auto_login_response", {
                "status": "error",
                "message": "自動ログイン失敗（バージョン不一致）"
            })
    else:
        # 該当なし => そもそもトークンが無効
        emit("auto_login_response", {
            "status": "error",
            "message": "自動ログイン失敗（トークン無効）"
        })

# -----------------------------------------------------------
# 3) チャット用の定数や共通変数
# -----------------------------------------------------------
cancellation_flags = {}

EXTENSION_TO_MIME = {
    "pdf": "application/pdf", "js": "application/x-javascript",
    "py": "text/x-python", "css": "text/css", "md": "text/md",
    "csv": "text/csv", "xml": "text/xml", "rtf": "text/rtf",
    "txt": "text/plain", "png": "image/png", "jpeg": "image/jpeg",
    "jpg": "image/jpeg", "webp": "image/webp", "heic": "image/heic",
    "heif": "image/heif", "mp4": "video/mp4", "mpeg": "video/mpeg",
    "mov": "video/mov", "avi": "video/avi", "flv": "video/x-flv",
    "mpg": "video/mpg", "webm": "video/webm", "wmv": "video/wmv",
    "3gpp": "video/3gpp", "wav": "audio/wav", "mp3": "audio/mp3",
    "aiff": "audio/aiff", "aac": "audio/aac", "ogg": "audio/ogg",
    "flac": "audio/flac",
}

USER_DIR = "data/"  # ユーザーデータ保存ディレクトリ

def get_user_dir(username):
    user_dir = os.path.join(USER_DIR, username)
    os.makedirs(user_dir, exist_ok=True)
    return user_dir

def get_username_from_token(token):
    """トークンからユーザー名を取得する"""
    if not token:
        return None
    
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT username FROM accounts WHERE auto_login_token = ?", (token,))
    row = c.fetchone()
    conn.close()
    
    if row:
        return row[0]
    return None

# -----------------------------------------------------------
# 4) チャット履歴管理 (従来どおりファイルに保存)
# -----------------------------------------------------------
def load_past_chats(user_dir):
    past_chats_file = os.path.join(user_dir, "past_chats_list")
    lock_file = past_chats_file + ".lock"  # ロック用ファイル(.lock)
    # withブロックを抜けるまでロックが保持される
    with FileLock(lock_file):
        try:
            past_chats = joblib.load(past_chats_file)
        except Exception:
            past_chats = {}
    return past_chats

def save_past_chats(user_dir, past_chats):
    past_chats_file = os.path.join(user_dir, "past_chats_list")
    lock_file = past_chats_file + ".lock"
    with FileLock(lock_file):
        joblib.dump(past_chats, past_chats_file)

def load_chat_messages(user_dir, chat_id):
    messages_file = os.path.join(user_dir, f"{chat_id}-st_messages")
    lock_file = messages_file + ".lock"
    with FileLock(lock_file):
        try:
            messages = joblib.load(messages_file)
        except Exception:
            messages = []
    return messages

def save_chat_messages(user_dir, chat_id, messages):
    messages_file = os.path.join(user_dir, f"{chat_id}-st_messages")
    lock_file = messages_file + ".lock"
    with FileLock(lock_file):
        joblib.dump(messages, messages_file)

def load_gemini_history(user_dir, chat_id):
    history_file = os.path.join(user_dir, f"{chat_id}-gemini_messages")
    lock_file = history_file + ".lock"
    with FileLock(lock_file):
        try:
            history = joblib.load(history_file)
        except Exception:
            history = []
    return history

def save_gemini_history(user_dir, chat_id, history):
    history_file = os.path.join(user_dir, f"{chat_id}-gemini_messages")
    lock_file = history_file + ".lock"
    with FileLock(lock_file):
        joblib.dump(history, history_file)

def delete_chat(user_dir, chat_id):
    try:
        os.remove(os.path.join(user_dir, f"{chat_id}-st_messages"))
        os.remove(os.path.join(user_dir, f"{chat_id}-gemini_messages"))
    except FileNotFoundError:
        pass
    past_chats = load_past_chats(user_dir)
    if chat_id in past_chats:
        del past_chats[chat_id]
        save_past_chats(user_dir, past_chats)

def find_gemini_index(messages, target_user_messages, include_model_responses=True):
    user_count = 0
    for idx, content in enumerate(messages):
        if content.role == "user":
            user_count += 1
            if user_count == target_user_messages:
                if include_model_responses:
                    # ユーザー発言に続くすべてのモデルの応答を含める
                    while idx + 1 < len(messages) and messages[idx + 1].role == "model":
                        idx += 1
                    return idx + 1
                else:
                    # ユーザー発言直後までとする
                    return idx + 1
    return len(messages)

# -----------------------------------------------------------
# 5) Flask ルートと SocketIO イベント
# -----------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")

@socketio.on("set_username")
def handle_set_username(data):
    token = data.get("token")
    username = get_username_from_token(token)
    if username:
        print(f"Token authenticated as user: {username}")
        emit("set_username_response", {"status": "success", "username": username})
    else:
        print("Invalid token received")
        emit("set_username_response", {"status": "error", "message": "無効なトークンです"})

# ------------------------
# チャット関連イベント
# ------------------------

@socketio.on("get_model_list")
def handle_get_model_list():
    api_models = client.models.list()
    api_model_names = [m.name for m in api_models]
    combined_models = sorted(set(api_model_names + [m.strip() for m in MODELS if m.strip()]))
    emit("model_list", {"models": combined_models})

@socketio.on("cancel_stream")
def handle_cancel_stream(data):
    sid = request.sid
    cancellation_flags[sid] = True

@app.route("/upload_large_file", methods=["POST"])
def upload_large_file():
    if "file" not in request.files:
        return jsonify({"status": "error", "message": "ファイルがありません"}), 400
    
    token = request.form.get("token")
    username = get_username_from_token(token)
    if not username:
        return jsonify({"status": "error", "message": "認証エラー"}), 401
    
    file = request.files["file"]
    
    # ユーザーディレクトリに一時ファイルとして保存
    user_dir = get_user_dir(username)
    timestamp = int(time.time())
    safe_filename = f"tmp_{timestamp}_{secure_filename(file.filename)}"
    tmp_path = os.path.join(user_dir, safe_filename)
    
    try:
        # ファイルを一時的に保存
        file.save(tmp_path)
        
        # デバッグ情報をログに出力
        print(f"一時ファイル保存完了: {tmp_path}, サイズ: {os.path.getsize(tmp_path)}")
        
        # Gemini File APIを使ってアップロード - より詳細なエラーハンドリング
        uploaded_file = client.files.upload(
            file=tmp_path,
        )
        print(f"Gemini File APIアップロード成功: file_id={uploaded_file.name}")
        
        # ファイルIDと情報を返す
        return jsonify({
            "status": "success",
            "file_id": uploaded_file.name,
            "file_name": file.filename,
            "file_mime_type": file.content_type
        })
    except Exception as api_error:
        # API固有のエラーをより詳細にログ出力
        if isinstance(api_error, Exception):
            print(f"Gemini File APIエラー: {str(api_error)}")
            print(f"エラータイプ: {type(api_error).__name__}")
            
            # クライアントへのわかりやすいエラーメッセージ
            return jsonify({
                "status": "error", 
                "message": f"ファイルAPIエラー: {str(api_error)}"
            }), 500
        else:
            # 一般的なエラー
            print(f"アップロード処理エラー: {str(api_error)}")
            return jsonify({"status": "error", "message": str(api_error)}), 500
    finally:
        # エラーの有無に関わらず一時ファイルを削除
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
                print(f"一時ファイル削除完了: {tmp_path}")
            except Exception as rm_error:
                print(f"一時ファイル削除エラー: {tmp_path}, エラー: {str(rm_error)}")

@socketio.on("resend_message")
def handle_resend_message(data):
    # キャンセルフラグをリセット
    sid = request.sid
    cancellation_flags[sid] = False

    token = data.get("token")
    username = get_username_from_token(token)
    if not username:
        emit("error", {"message": "認証エラー"})
        return
    
    chat_id = data.get("chat_id")
    message_index = data.get("message_index")
    model_name = data.get("model_name")
    grounding_enabled = data.get("grounding_enabled", False)
    code_execution_enabled = data.get("code_execution_enabled", False)
    
    user_dir = get_user_dir(username)
    messages = load_chat_messages(user_dir, chat_id)
    
    # 指定されたインデックスが範囲外またはユーザーメッセージでない場合はエラー
    if message_index >= len(messages) or messages[message_index]["role"] != "user":
        emit("error", {"message": "再送信できるのはユーザーメッセージのみです"})
        return
    
    # 指定されたメッセージまでの履歴を保持し、そのメッセージ後のモデル応答を削除
    if message_index + 1 < len(messages):
        messages = messages[:message_index + 1]
        
    # Gemini履歴を取得
    gemini_history = load_gemini_history(user_dir, chat_id)
    
    # 対象のユーザーメッセージを特定し格納
    target_user_messages = sum(1 for msg in messages if msg["role"] == "user")
    
    # ユーザーメッセージのインデックスを取得（モデル応答を含まない）
    user_message_index = find_gemini_index(gemini_history, target_user_messages, include_model_responses=False)
    
    # 対象のユーザーメッセージまでの履歴を取得（そのメッセージを含む）
    truncated_history = []
    user_message = None
    
    # ユーザーメッセージまでの履歴（ユーザーメッセージは含まない）
    if user_message_index > 0:
        truncated_history = gemini_history[:user_message_index-1]
    
    # ユーザーメッセージを取得
    if user_message_index > 0 and user_message_index <= len(gemini_history):
        user_message = gemini_history[user_message_index-1]
    else:
        emit("error", {"message": "対象のメッセージが見つかりません"})
        return
    
    # 変更した履歴を保存
    save_chat_messages(user_dir, chat_id, messages)
    save_gemini_history(user_dir, chat_id, truncated_history)
    
    try:
        # Gemini API設定
        if grounding_enabled:
            configs = GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                tools=[google_search_tool],
            )
        elif code_execution_enabled:
            configs = GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                tools=[code_execution_tool],
            )
        else:
            configs = GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
            )

        # チャットインスタンスを作成（ユーザーメッセージを含まない履歴）
        chat = client.chats.create(model=model_name, history=truncated_history)
        
        # モデル応答用のモック要素を追加
        model_message = {
            "role": "model",
            "content": "",
            "timestamp": time.time()
        }
        messages.append(model_message)

        full_response = ""
        usage_metadata = None
        formatted_metadata = ""
        all_grounding_links = ""
        all_grounding_queries = ""

        #履歴用
        input_content = t.t_content(chat._modules._api_client, user_message.parts)
        # ユーザーメッセージのpartsを送信
        for chunk in chat.send_message_stream(user_message.parts, config=configs):
            # クライアントがキャンセルをリクエストしたら中断
            if cancellation_flags.get(sid):
                messages.pop()
                chat.record_history(
                    user_input=input_content,
                    model_output=[],
                    automatic_function_calling_history=[],
                    is_valid=False 
                )
                save_chat_messages(user_dir, chat_id, messages)
                save_gemini_history(user_dir, chat_id, chat.get_history(curated=False))
                emit("stream_cancelled", {"chat_id": chat_id})
                return

            if hasattr(chunk, "usage_metadata") and chunk.usage_metadata:
                usage_metadata = chunk.usage_metadata

            chunk_text = ""  # このチャンクのテキストを蓄積する変数

            if chunk.candidates and chunk.candidates[0].content and chunk.candidates[0].content.parts:
                for part in chunk.candidates[0].content.parts:
                    if part.text:
                        chunk_text += part.text
                    if hasattr(part, 'executable_code') and part.executable_code:
                        chunk_text += f"\n**Executable Code**\n```Python\n{part.executable_code.code}\n```\n"
                    if hasattr(part, 'code_execution_result') and part.code_execution_result:
                        chunk_text += f"\n**Code Execution Result**\n```Python\n{part.code_execution_result}\n```\n"
                    if hasattr(chunk, 'thought') and chunk.thought:
                         chunk_text += f"\nThought:\n{chunk.thought}\n"

            # グラウンディング処理
            if hasattr(chunk, "candidates") and chunk.candidates:
                candidate = chunk.candidates[0]
                if hasattr(candidate, "grounding_metadata") and candidate.grounding_metadata:
                    metadata = candidate.grounding_metadata
                    if hasattr(metadata, "grounding_chunks") and metadata.grounding_chunks:
                        for i, grounding_chunk in enumerate(metadata.grounding_chunks):
                            if hasattr(grounding_chunk, "web") and grounding_chunk.web:
                                all_grounding_links += f"[{i + 1}][{grounding_chunk.web.title}]({grounding_chunk.web.uri}) "
                    if hasattr(metadata, "web_search_queries") and metadata.web_search_queries:
                        for query in metadata.web_search_queries:
                            all_grounding_queries += f"{query} / "

            if chunk_text:
                full_response += chunk_text
                # 最後のメッセージを更新
                messages[-1]["content"] += chunk_text
                # クライアントに通知
                emit("gemini_response_chunk", {"chunk": chunk_text, "chat_id": chat_id})

        # トークン数情報を整形
        if usage_metadata:
            formatted_metadata = "\n\n---\n**" + model_name + "**    Token: " + f"{usage_metadata.total_token_count:,}" + "\n\n"
            full_response += formatted_metadata
            messages[-1]["content"] += formatted_metadata
            emit("gemini_response_chunk", {"chunk": formatted_metadata, "chat_id": chat_id})
            formatted_metadata = ""

        # グラウンディング情報を整形して送信
        if all_grounding_queries:
            all_grounding_queries = " / ".join(
                sorted(set(all_grounding_queries.rstrip(" /").split(" / ")))
            )
        if all_grounding_links:
            formatted_metadata += all_grounding_links + "\n"
        if all_grounding_queries:
            formatted_metadata += "\nQuery: " + all_grounding_queries + "\n"

        # 最後の応答にグラウンディング情報を追加
        if formatted_metadata:
            full_response += formatted_metadata
            messages[-1]["content"] += formatted_metadata
            emit("gemini_response_chunk", {"chunk": formatted_metadata, "chat_id": chat_id})

        # Gemini履歴を保存
        save_chat_messages(user_dir, chat_id, messages)
        save_gemini_history(user_dir, chat_id, chat.get_history(curated=True))
        emit("gemini_response_complete", {"chat_id": chat_id})
        # 再送信完了通知
        emit("message_resent", {"index": message_index})

    except Exception as e:
        # エラー時は一時メッセージを削除して詳細なエラー情報を送信
        if len(messages) > 0 and messages[-1]["role"] == "model":
            messages.pop()
            save_chat_messages(user_dir, chat_id, messages)
        
        emit("gemini_response_error", {"error": str(e), "chat_id": chat_id})
    finally:
        # 応答処理終了後にキャンセルフラグを削除
        cancellation_flags.pop(sid, None)

@socketio.on("send_message")
def handle_message(data):
    # キャンセルフラグをリセット
    sid = request.sid
    cancellation_flags[sid] = False

    token = data.get("token")
    username = get_username_from_token(token)
    if not username:
        emit("error", {"message": "認証エラー"})
        return
    chat_id = data.get("chat_id")
    model_name = data.get("model_name")
    message = data.get("message")
    grounding_enabled = data.get("grounding_enabled", False)
    code_execution_enabled = data.get("code_execution_enabled", False)
    
    # 既存のファイルデータ形式
    file_data_base64 = data.get("file_data")
    file_name = data.get("file_name")
    file_mime_type = data.get("file_mime_type")
    # 新しいファイルID形式
    file_id = data.get("file_id")

    user_dir = get_user_dir(username)
    messages = load_chat_messages(user_dir, chat_id)
    gemini_history = load_gemini_history(user_dir, chat_id)
    chat = client.chats.create(model=model_name, history=gemini_history)

    # 新規チャットの場合、past_chats にタイトルを登録
    past_chats = load_past_chats(user_dir)
    if chat_id not in past_chats:
        chat_title = message[:30]
        current_time = time.time()
        past_chats[chat_id] = {"title": chat_title, "bookmarked": False, "lastUpdated": current_time}
        save_past_chats(user_dir, past_chats)
        emit("history_list", {"history": past_chats})

    # ユーザーのプロンプトと添付ファイル情報を分離して履歴に追加
    user_message = {
        "role": "user",
        "content": message,
        "timestamp": time.time()
    }
    
    # 添付ファイル情報があれば追加
    if file_name or file_id:
        attachment_info = {
            "name": file_name,
            "type": file_mime_type,
            "file_id": file_id
        }
        user_message["attachments"] = [attachment_info]
    
    messages.append(user_message)
    save_chat_messages(user_dir, chat_id, messages)
    
    try:
        # コンテンツの作成方法を分岐
        if file_id:
            # File APIを使った大容量ファイル参照
            file_ref = client.files.get(name=file_id)
            contents = [file_ref, message]
        elif file_data_base64:
            # 既存の小さいファイル処理（base64データ）
            file_data = base64.b64decode(file_data_base64)
            file_part = types.Part.from_bytes(data=file_data, mime_type=file_mime_type)
            contents = [file_part, message]
        else:
            # ファイルなしの場合
            contents = message

        # 以下既存のコード（グラウンディング設定など）
        if grounding_enabled:
            configs = GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                tools=[google_search_tool],
            )
        elif code_execution_enabled:
            configs = GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                tools=[code_execution_tool],
            )
        else:
            configs = GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
            )
        
        #履歴用
        input_content = t.t_content(chat._modules._api_client, contents)
        # ストリーミング応答開始
        full_response = ""
        usage_metadata = None
        formatted_metadata = ""
        all_grounding_links = ""
        all_grounding_queries = ""

        # モデルからの応答を追加するためのエントリを作成
        model_message = {
            "role": "model",
            "content": "",  # ストリーミングで埋めていく
            "timestamp": time.time()
        }
        messages.append(model_message)

        for chunk in chat.send_message_stream(message=contents, config=configs):
            # クライアントがキャンセルをリクエストしたら中断
            if cancellation_flags.get(sid):
                messages.pop()
                chat.record_history(
                    user_input=input_content,
                    model_output=[],
                    automatic_function_calling_history=[],
                    is_valid=False 
                )
                save_chat_messages(user_dir, chat_id, messages)
                save_gemini_history(user_dir, chat_id, chat.get_history(curated=False))
                emit("stream_cancelled", {"chat_id": chat_id})
                return

            if hasattr(chunk, "usage_metadata") and chunk.usage_metadata:
                usage_metadata = chunk.usage_metadata

            chunk_text = ""  # このチャンクのテキストを蓄積する変数

            if chunk.candidates and chunk.candidates[0].content and chunk.candidates[0].content.parts:
                for part in chunk.candidates[0].content.parts:
                    if part.text:
                        chunk_text += part.text
                    if hasattr(part, 'executable_code') and part.executable_code:
                        chunk_text += f"\n**Executable Code**\n```Python\n{part.executable_code.code}\n```\n"
                    if hasattr(part, 'code_execution_result') and part.code_execution_result:
                        chunk_text += f"\n**Code Execution Result**\n```Python\n{part.code_execution_result}\n```\n"
                    if hasattr(chunk, 'thought') and chunk.thought:
                         chunk_text += f"\nThought:\n{chunk.thought}\n"

            # グラウンディング処理をここで行う
            if hasattr(chunk, "candidates") and chunk.candidates:
                candidate = chunk.candidates[0]
                if hasattr(candidate, "grounding_metadata") and candidate.grounding_metadata:
                    metadata = candidate.grounding_metadata
                    if hasattr(metadata, "grounding_chunks") and metadata.grounding_chunks:
                        for i, grounding_chunk in enumerate(metadata.grounding_chunks):
                            if hasattr(grounding_chunk, "web") and grounding_chunk.web:
                                all_grounding_links += f"[{i + 1}][{grounding_chunk.web.title}]({grounding_chunk.web.uri}) "
                    if hasattr(metadata, "web_search_queries") and metadata.web_search_queries:
                        for query in metadata.web_search_queries:
                            all_grounding_queries += f"{query} / "

            if chunk_text:
                full_response += chunk_text
                # 最後のメッセージを更新
                messages[-1]["content"] += chunk_text
                # クライアントに通知
                emit("gemini_response_chunk", {"chunk": chunk_text, "chat_id": chat_id})

        # トークン数情報を整形
        if usage_metadata:
            formatted_metadata = "\n\n---\n**" + model_name + "**    Token: " + f"{usage_metadata.total_token_count:,}" + "\n\n"
            full_response += formatted_metadata
            messages[-1]["content"] += formatted_metadata
            emit("gemini_response_chunk", {"chunk": formatted_metadata, "chat_id": chat_id})
            formatted_metadata = ""

        # グラウンディング情報を整形して送信
        if all_grounding_queries:
            all_grounding_queries = " / ".join(
                sorted(set(all_grounding_queries.rstrip(" /").split(" / ")))
            )
        if all_grounding_links:
            formatted_metadata += all_grounding_links + "\n"
        if all_grounding_queries:
            formatted_metadata += "\nQuery: " + all_grounding_queries + "\n"

        # 最後の応答にグラウンディング情報を追加
        if formatted_metadata:
            full_response += formatted_metadata
            messages[-1]["content"] += formatted_metadata
            emit("gemini_response_chunk", {"chunk": formatted_metadata, "chat_id": chat_id})

        # Gemini履歴を保存
        save_chat_messages(user_dir, chat_id, messages)
        save_gemini_history(user_dir, chat_id, chat.get_history(curated=True))
        emit("gemini_response_complete", {"chat_id": chat_id})

    except Exception as e:
        # エラー時は一時メッセージを削除して詳細なエラー情報を送信
        if len(messages) > 0 and messages[-1]["role"] == "model":
            messages.pop()
            save_chat_messages(user_dir, chat_id, messages)
        
        emit("gemini_response_error", {"error": str(e), "chat_id": chat_id})
    finally:
        # 応答処理終了後にキャンセルフラグを削除
        cancellation_flags.pop(sid, None)

@socketio.on("delete_message")
def handle_delete_message(data):
    token = data.get("token")
    username = get_username_from_token(token)
    if not username:
        emit("error", {"message": "認証エラー"})
        return
    chat_id = data.get("chat_id")
    message_index = data.get("message_index")

    user_dir = get_user_dir(username)
    messages = load_chat_messages(user_dir, chat_id)
    gemini_history = load_gemini_history(user_dir, chat_id)

    if message_index == 0:
        delete_chat(user_dir, chat_id)
    else:
        deleted_message_role = messages[message_index]["role"]
        messages = messages[:message_index]
        target_user_messages = sum(1 for msg in messages if msg["role"] == "user")
        if deleted_message_role == "model":
            gemini_index = find_gemini_index(gemini_history, target_user_messages, include_model_responses=False)
        else:
            gemini_index = find_gemini_index(gemini_history, target_user_messages, include_model_responses=True)
        gemini_history = gemini_history[:gemini_index]
        save_chat_messages(user_dir, chat_id, messages)
        save_gemini_history(user_dir, chat_id, gemini_history)

    emit("message_deleted", {"index": message_index})

@socketio.on("edit_message")
def handle_edit_message(data):
    token = data.get("token")
    username = get_username_from_token(token)
    if not username:
        emit("error", {"message": "認証エラー"})
        return
    
    chat_id = data.get("chat_id")
    message_index = data.get("message_index")
    new_text = data.get("new_text")
    
    user_dir = get_user_dir(username)
    messages = load_chat_messages(user_dir, chat_id)
    gemini_history = load_gemini_history(user_dir, chat_id)
    
    # ユーザーメッセージかチェック
    if message_index >= len(messages) or messages[message_index]["role"] != "user":
        emit("error", {"message": "編集できるのはユーザーメッセージのみです"})
        return
    
    # messagesの内容を更新
    messages[message_index]["content"] = new_text
    
    # message_indexに対応するgemini_historyのインデックスを特定
    user_message_count = sum(1 for i, msg in enumerate(messages) if msg["role"] == "user" and i <= message_index)
    gemini_index = find_gemini_user_index(gemini_history, user_message_count)
    
    if gemini_index is not None:
        for part in gemini_history[gemini_index].parts:
            if part.text is not None:
                part.text = new_text
                break  # 最初に見つかった要素を更新したらループを抜ける
        
        # 変更を保存
        save_chat_messages(user_dir, chat_id, messages)
        save_gemini_history(user_dir, chat_id, gemini_history)
        
        emit("message_edited", {"index": message_index, "new_text": new_text})
    else:
        emit("error", {"message": "該当するメッセージが見つかりません"})

def find_gemini_user_index(messages, target_user_count):
    """gemini履歴内の特定のユーザーメッセージのインデックスを検索"""
    user_count = 0
    for idx, content in enumerate(messages):
        if content.role == "user":
            user_count += 1
            if user_count == target_user_count:
                return idx
    return None

@socketio.on("get_history_list")
def handle_get_history_list(data):
    token = data.get("token")
    username = get_username_from_token(token)
    if not username:
        emit("error", {"message": "認証エラー"})
        return
    user_dir = get_user_dir(username)
    past_chats = load_past_chats(user_dir)
    emit("history_list", {"history": past_chats})

@socketio.on("load_chat")
def handle_load_chat(data):
    token = data.get("token")
    username = get_username_from_token(token)
    if not username:
        emit("error", {"message": "認証エラー"})
        return
    chat_id = data.get("chat_id")
    user_dir = get_user_dir(username)
    messages = load_chat_messages(user_dir, chat_id)
    emit("chat_loaded", {"messages": messages, "chat_id": chat_id})

@socketio.on("new_chat")
def handle_new_chat(data):
    token = data.get("token")
    username = get_username_from_token(token)
    if not username:
        emit("error", {"message": "認証エラー"})
        return
    new_chat_id = f"{time.time()}"
    emit("chat_created", {"chat_id": new_chat_id})

@socketio.on("delete_chat")
def handle_delete_chat(data):
    token = data.get("token")
    username = get_username_from_token(token)
    if not username:
        emit("error", {"message": "認証エラー"})
        return
    chat_id = data.get("chat_id")
    user_dir = get_user_dir(username)
    delete_chat(user_dir, chat_id)
    emit("chat_deleted", {"chat_id": chat_id})

@socketio.on("rename_chat")
def handle_rename_chat(data):
    token = data.get("token")
    username = get_username_from_token(token)
    if not username:
        emit("error", {"message": "認証エラー"})
        return
    chat_id = data.get("chat_id")
    new_title = data.get("new_title")
    
    user_dir = get_user_dir(username)
    past_chats = load_past_chats(user_dir)
    
    if chat_id in past_chats:
        past_chats[chat_id]["title"] = new_title
        save_past_chats(user_dir, past_chats)
        emit("chat_renamed", {"chat_id": chat_id, "new_title": new_title})
        emit("history_list", {"history": past_chats})

# ブックマーク切り替え用のSocketIOイベント
@socketio.on("toggle_bookmark")
def handle_toggle_bookmark(data):
    token = data.get("token")
    username = get_username_from_token(token)
    if not username:
        emit("error", {"message": "認証エラー"})
        return
    chat_id = data.get("chat_id")
    
    user_dir = get_user_dir(username)
    past_chats = load_past_chats(user_dir)
    
    if chat_id in past_chats:
        past_chats[chat_id]["bookmarked"] = not past_chats[chat_id].get("bookmarked", False)
        save_past_chats(user_dir, past_chats)
        emit("bookmark_toggled", {
            "chat_id": chat_id, 
            "bookmarked": past_chats[chat_id]["bookmarked"]
        })
        emit("history_list", {"history": past_chats})

# -----------------------------------------------------------
# 6) メイン実行
# -----------------------------------------------------------
if __name__ == "__main__":
    # SQLite初期化
    init_db()

    # geventベースでサーバ起動（geventインストール済みの場合に自動で使用）
    socketio.run(app, debug=True, host="0.0.0.0", port=5000)
