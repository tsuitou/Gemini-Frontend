# gevent を使う場合のモンキーパッチ（WSGIサーバを gevent にするため）
from gevent import monkey
monkey.patch_all()

import os
import uuid
import io
import json
import bcrypt
import hashlib
import re
import base64
import time
import sqlite3
import joblib
import copy
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
socketio = SocketIO(app, async_mode="gevent", cors_allowed_origins="*", max_http_buffer_size=20 * 1024 * 1024, ping_timeout=120, ping_interval=25, binary=True)

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
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD")

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
    c.execute("""
    CREATE TABLE IF NOT EXISTS admin_sessions (
        session_id TEXT PRIMARY KEY,
        created_at INTEGER
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
    if not re.match(r"^[a-zA-Z0-9]*$", username):
        return {"status": "error", "message": "ユーザー名には英数字のみ使用可能です。"}
            
    if not re.match(r"^[a-zA-Z0-9!@#$%^&*()_+\-=\[\]{};':\"\\|,.<>\/?]*$", password):
        return {"status": "error", "message": "パスワードには英数字と記号のみ使用可能です。"}

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
IMAGE_DIR = "static/images"

os.makedirs(IMAGE_DIR, exist_ok=True)

def save_generated_image(username, base64_data, mime_type):
    # ユーザー別ディレクトリ作成
    user_image_dir = os.path.join(IMAGE_DIR, username)
    os.makedirs(user_image_dir, exist_ok=True)
    
    # 拡張子を取得
    ext = mime_type.split('/')[-1]
    if ext == 'jpeg':
        ext = 'jpg'  # 一般的な拡張子に統一
    
    # ファイル名生成（ユニークなIDを使用）
    filename = f"{str(uuid.uuid4())}.{ext}"
    filepath = os.path.join(user_image_dir, filename)
    
    try:
        # Base64データをバイナリに変換して保存
        image_data = base64.b64decode(base64_data)
        with open(filepath, 'wb') as f:
            f.write(image_data)
        
        # 公開用URL（相対パス）
        url = '/static/images/' + username + '/' + filename
        print(url)
        return url
    except Exception as e:
        print(f"画像保存エラー: {str(e)}")
        return None

def delete_image_file(image_url):
    if not image_url:
        return
    
    try:
        # URLの先頭の/を削除してファイルパスに変換
        filepath = image_url.lstrip('/')
        if os.path.exists(filepath):
            os.remove(filepath)
            print(f"画像ファイル削除: {filepath}")
    except Exception as e:
        print(f"画像ファイル削除エラー: {str(e)}")

def delete_chat_images(messages):
    for message in messages:
        if message.get("role") == "model" and message.get("images"):
            for image_url in message["images"]:
                delete_image_file(image_url)

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
    lock_file = past_chats_file + ".lock"
    lock = FileLock(lock_file, timeout=10)  # タイムアウトを10秒に設定
    try:
        lock.acquire()
        try:
            past_chats = joblib.load(past_chats_file)
        except Exception:
            past_chats = {}
        return past_chats
    finally:
        lock.release()  # 例外が発生してもロックを確実に解除

def save_past_chats(user_dir, past_chats):
    past_chats_file = os.path.join(user_dir, "past_chats_list")
    lock_file = past_chats_file + ".lock"
    lock = FileLock(lock_file, timeout=10)
    try:
        lock.acquire()
        joblib.dump(past_chats, past_chats_file)
    finally:
        lock.release()

def load_chat_messages(user_dir, chat_id):
    messages_file = os.path.join(user_dir, f"{chat_id}-st_messages")
    lock_file = messages_file + ".lock"
    lock = FileLock(lock_file, timeout=10)
    try:
        lock.acquire()
        try:
            messages = joblib.load(messages_file)
        except Exception:
            messages = []
        return messages
    finally:
        lock.release()

def save_chat_messages(user_dir, chat_id, messages):
    messages_file = os.path.join(user_dir, f"{chat_id}-st_messages")
    lock_file = messages_file + ".lock"
    lock = FileLock(lock_file, timeout=10)
    try:
        lock.acquire()
        joblib.dump(messages, messages_file)
    finally:
        lock.release()

def load_gemini_history(user_dir, chat_id):
    history_file = os.path.join(user_dir, f"{chat_id}-gemini_messages")
    lock_file = history_file + ".lock"
    lock = FileLock(lock_file, timeout=10)
    try:
        lock.acquire()
        try:
            history = joblib.load(history_file)
        except Exception:
            history = []
        return history
    finally:
        lock.release()

def save_gemini_history(user_dir, chat_id, history):
    history_file = os.path.join(user_dir, f"{chat_id}-gemini_messages")
    lock_file = history_file + ".lock"
    lock = FileLock(lock_file, timeout=10)
    try:
        lock.acquire()
        joblib.dump(history, history_file)
    finally:
        lock.release()

def delete_chat(user_dir, chat_id):
    try:
        # 画像を削除するために先にメッセージを読み込む
        messages = load_chat_messages(user_dir, chat_id)
        # 画像ファイルを削除
        delete_chat_images(messages)

        file_st_messages = os.path.join(user_dir, f"{chat_id}-st_messages")
        if os.path.exists(file_st_messages):
            os.remove(file_st_messages)

        file_gemini_messages = os.path.join(user_dir, f"{chat_id}-gemini_messages")
        if os.path.exists(file_gemini_messages):
            os.remove(file_gemini_messages)

        lock_file_message = os.path.join(user_dir, f"{chat_id}-st_messages.lock")
        if os.path.exists(lock_file_message):
            os.remove(lock_file_message)

        lock_file_gemini = os.path.join(user_dir, f"{chat_id}-gemini_messages.lock")
        if os.path.exists(lock_file_gemini):
            os.remove(lock_file_gemini)
        
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

def find_gemini_user_index(messages, target_user_count):
    user_count = 0
    for idx, content in enumerate(messages):
        if content.role == "user":
            user_count += 1
            if user_count == target_user_count:
                return idx
    return None

def fix_comprehensive_history(comprehensive_history):
    if not comprehensive_history:
        return []
    
    # ステップ1: 空のContentを削除し、モデル応答を修正/結合
    fixed_history = []
    i = 0
    
    while i < len(comprehensive_history):
        current = comprehensive_history[i]
        
        # 空のpartsを持つContentをスキップ
        if not hasattr(current, 'parts') or not current.parts:
            i += 1
            continue
        
        # モデル応答の場合、連続する応答を結合
        if current.role == "model":
            # 現在と後続のモデル応答からパーツを収集
            all_parts = list(current.parts) if current.parts else []
            
            # 連続するモデル応答を探索
            j = i + 1
            while j < len(comprehensive_history) and comprehensive_history[j].role == "model":
                next_content = comprehensive_history[j]
                if hasattr(next_content, 'parts') and next_content.parts:
                    all_parts.extend(next_content.parts)
                j += 1
            
            # 有効なパーツがある場合のみコンテンツを作成
            if all_parts:
                # 修正用に現在のコンテンツのディープコピーを作成
                new_content = copy.deepcopy(current)
                
                # 空のパーツを除去し、すべてのデータ型（画像含む）を保持
                valid_parts = []
                for part in all_parts:
                    # テキストコンテンツの確認
                    if hasattr(part, 'text') and part.text is not None and part.text != "":
                        valid_parts.append(part)
                        continue
                    
                    # Blob（画像）またはinline_dataの確認
                    if hasattr(part, 'inline_data') and part.inline_data is not None:
                        valid_parts.append(part)
                        continue
                        
                    # パーツが__dictを持ち、非Noneの値があれば有効と見なす
                    if hasattr(part, '__dict__') and part.__dict__:
                        if any(v is not None for k, v in part.__dict__.items()):
                            valid_parts.append(part)
                            continue
                
                if valid_parts:
                    new_content.parts = valid_parts
                    fixed_history.append(new_content)
            
            # 結合したコンテンツ分だけインデックスを進める
            i = j
        else:
            # ユーザーコンテンツの場合、有効なパーツがあれば追加
            valid_user_parts = []
            for part in current.parts:
                # テキストコンテンツの確認
                if hasattr(part, 'text') and part.text is not None and part.text != "":
                    valid_user_parts.append(part)
                    continue
                
                # Blob（画像）またはinline_dataの確認
                if hasattr(part, 'inline_data') and part.inline_data is not None:
                    valid_user_parts.append(part)
                    continue
                    
                # パーツが__dictを持ち、非Noneの値があれば有効と見なす
                if hasattr(part, '__dict__') and part.__dict__:
                    if any(v is not None for k, v in part.__dict__.items()):
                        valid_user_parts.append(part)
                        continue
            
            if valid_user_parts:
                user_content = copy.deepcopy(current)
                user_content.parts = valid_user_parts
                fixed_history.append(user_content)
            
            i += 1
    
    # 空のコンテンツを削除しモデル応答を結合した修正済み履歴を返す
    return fixed_history

# -----------------------------------------------------------
# 5) Flask ルートと SocketIO イベント
# -----------------------------------------------------------
@app.route("/gemini/")
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
    filtered_models = [model for model in combined_models if "gemini" in model]
    emit("model_list", {"models": filtered_models})

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
        # エラー処理（既存コードと同じ）
        if isinstance(api_error, Exception):
            print(f"Gemini File APIエラー: {str(api_error)}")
            print(f"エラータイプ: {type(api_error).__name__}")
            
            return jsonify({
                "status": "error", 
                "message": f"ファイルAPIエラー: {str(api_error)}"
            }), 500
        else:
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
    image_generation_enabled = data.get("image_generation_enabled", False)  # 新しいフラグ
    
    user_dir = get_user_dir(username)
    messages = load_chat_messages(user_dir, chat_id)
    
    # 指定されたインデックスが範囲外またはユーザーメッセージでない場合はエラー
    if message_index >= len(messages) or messages[message_index]["role"] != "user":
        emit("error", {"message": "再送信できるのはユーザーメッセージのみです"})
        return
        
    messages_to_delete = messages[message_index:]
    delete_chat_images(messages_to_delete)
    
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
                response_modalities=['Text'],
            )
        elif code_execution_enabled:
            configs = GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                tools=[code_execution_tool],
                response_modalities=['Text'],
            )
        elif image_generation_enabled:
            configs = GenerateContentConfig(
                response_modalities=['Text', 'Image']
            )
        else:
            configs = GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                response_modalities=['Text'],
            )

        # チャットインスタンスを作成（ユーザーメッセージを含まない履歴）
        chat = client.chats.create(model=model_name, history=truncated_history, config=configs)
        
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
        for chunk in chat.send_message_stream(user_message.parts):
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
                save_gemini_history(user_dir, chat_id, fix_comprehensive_history(chat.get_history(curated=False)))
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
                    if hasattr(part, 'inline_data') and part.inline_data is not None:
                        try:
                            mime = part.inline_data.mime_type
                            data = part.inline_data.data
                            
                            # バイナリデータの場合は適切にBase64エンコード
                            if isinstance(data, bytes):
                                data = base64.b64encode(data).decode('utf-8')
                            
                            # 画像をファイルとして保存
                            image_url = save_generated_image(username, data, mime)
                            
                            if image_url:
                                # マークダウン形式の画像参照を追加
                                chunk_text += f"\n![Generated Image]({image_url})\n"
                                
                                # 画像URLを記録（削除時に使用）
                                if "images" not in messages[-1]:
                                    messages[-1]["images"] = []
                                messages[-1]["images"].append(image_url)
                            else:
                                chunk_text += "\n[画像の保存に失敗しました]\n"
                            
                        except Exception as img_error:
                            print(f"画像処理エラー: {str(img_error)}")
                            # エラーがあっても処理を続行
                            chunk_text += "\n[画像の処理中にエラーが発生しました]\n"

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
        save_gemini_history(user_dir, chat_id, fix_comprehensive_history(chat.get_history(curated=False)))
        emit("gemini_response_complete", {"chat_id": chat_id})
        # 再送信完了通知
        emit("message_resent", {"index": message_index})

    except Exception as e:
        # エラー時は一時メッセージを削除して詳細なエラー情報を送信
        if len(messages) > 0 and messages[-1]["role"] == "model":
            messages.pop()
            chat.record_history(
                user_input=input_content,
                model_output=[],
                automatic_function_calling_history=[],
                is_valid=False 
            )
            save_chat_messages(user_dir, chat_id, messages)
            save_gemini_history(user_dir, chat_id, fix_comprehensive_history(chat.get_history(curated=False)))
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
    image_generation_enabled = data.get("image_generation_enabled", False)
    
    # 複数ファイル情報を取得
    files = data.get("files", [])

    user_dir = get_user_dir(username)
    messages = load_chat_messages(user_dir, chat_id)
    gemini_history = load_gemini_history(user_dir, chat_id)

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
    if files:
        attachments = []
        for file_info in files:
            attachment = {
                "name": file_info.get("file_name"),
                "type": file_info.get("file_mime_type"),
                "file_id": file_info.get("file_id")
            }
            attachments.append(attachment)
        user_message["attachments"] = attachments
    
    messages.append(user_message)
    save_chat_messages(user_dir, chat_id, messages)
    
    try:
        # コンテンツの作成 - 複数ファイル対応
        contents = []
        
        # ファイル処理 - フロントエンドから送られた全ファイルを追加
        for file_info in files:
            file_id = file_info.get("file_id")
            file_data_base64 = file_info.get("file_data")
            file_name = file_info.get("file_name")
            file_mime_type = file_info.get("file_mime_type")
            
            if file_id:
                # File APIを使った大容量ファイル参照
                file_ref = client.files.get(name=file_id)
                contents.append(file_ref)
            elif file_data_base64:
                # Base64エンコードされた小さいファイル
                file_data = base64.b64decode(file_data_base64)
                file_part = types.Part.from_bytes(data=file_data, mime_type=file_mime_type)
                contents.append(file_part)
        
        # メッセージテキストを最後に追加
        contents.append(types.Part.from_text(text=message))
        

        if grounding_enabled:
            configs = GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                tools=[google_search_tool],
                response_modalities=['Text']
            )
        elif code_execution_enabled:
            configs = GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                tools=[code_execution_tool],
                response_modalities=['Text']
            )
        elif image_generation_enabled:
            configs = GenerateContentConfig(
                response_modalities=['Text', 'Image']
            )
        else:
            configs = GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                response_modalities=['Text']
            )
        chat = client.chats.create(model=model_name, history=gemini_history, config=configs)
        #履歴用
        input_content = t.t_content(chat._modules._api_client, contents)
        
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
        
        # ストリーミング応答開始
        for chunk in chat.send_message_stream(message=contents):
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
                save_gemini_history(user_dir, chat_id, fix_comprehensive_history(chat.get_history(curated=False)))
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
                    if hasattr(part, 'inline_data') and part.inline_data is not None:
                        try:
                            mime = part.inline_data.mime_type
                            data = part.inline_data.data
                            
                            # バイナリデータの場合は適切にBase64エンコード
                            if isinstance(data, bytes):
                                data = base64.b64encode(data).decode('utf-8')
                            
                            # 画像をファイルとして保存
                            image_url = save_generated_image(username, data, mime)
                            
                            if image_url:
                                # マークダウン形式の画像参照を追加
                                chunk_text += f"\n![Generated Image]({image_url})\n"
                                
                                # 画像URLを記録（削除時に使用）
                                if "images" not in messages[-1]:
                                    messages[-1]["images"] = []
                                messages[-1]["images"].append(image_url)
                            else:
                                chunk_text += "\n[画像の保存に失敗しました]\n"
                            
                        except Exception as img_error:
                            print(f"画像処理エラー: {str(img_error)}")
                            # エラーがあっても処理を続行
                            chunk_text += "\n[画像の処理中にエラーが発生しました]\n"

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
        save_gemini_history(user_dir, chat_id, fix_comprehensive_history(chat.get_history(curated=False)))
        emit("gemini_response_complete", {"chat_id": chat_id})

    except Exception as e:
        # エラー時は一時メッセージを削除して詳細なエラー情報を送信
        if len(messages) > 0 and messages[-1]["role"] == "model":
            messages.pop()
            chat.record_history(
                user_input=input_content,
                model_output=[],
                automatic_function_calling_history=[],
                is_valid=False 
            )
            save_chat_messages(user_dir, chat_id, messages)
            save_gemini_history(user_dir, chat_id, fix_comprehensive_history(chat.get_history(curated=False)))
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
        # 削除されるメッセージ以降の画像を削除
        messages_to_delete = messages[message_index:]
        delete_chat_images(messages_to_delete)
        
        # 以下は既存のコード
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

@socketio.on("disconnect")
def handle_disconnect():
    """クライアント切断時のクリーンアップ"""
    sid = request.sid
    # もしキャンセルフラグが残っていれば削除
    cancellation_flags.pop(sid, None)
    print(f"[disconnect] sid={sid} cleaned up.")

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

@socketio.on("edit_model_message")
def handle_edit_model_message(data):
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
    
    # モデルメッセージかチェック
    if message_index >= len(messages) or messages[message_index]["role"] != "model":
        emit("error", {"message": "編集できるのはモデルメッセージのみです"})
        return
    
    # メッセージの内容を更新（st_messagesの内容更新）
    messages[message_index]["content"] = new_text
    
    # 変更を保存
    save_chat_messages(user_dir, chat_id, messages)
    
    # gemini_historyの対応するメッセージを探して更新
    gemini_history = load_gemini_history(user_dir, chat_id)
    
    # 現在のメッセージ位置までのユーザーメッセージ数をカウント
    user_message_count = sum(1 for i, msg in enumerate(messages) if msg["role"] == "user" and i < message_index)
    
    # 対応するモデルメッセージを探す
    model_found = False
    user_count = 0
    
    for idx, content in enumerate(gemini_history):
        if content.role == "user":
            user_count += 1
            
        # 対象のユーザーメッセージの後の最初のモデル応答を見つけたら
        if content.role == "model" and user_count == user_message_count:
            model_found = True
            
            try:
                # テキストを持つパートを見つける
                text_parts = []
                for part in content.parts:
                    if hasattr(part, 'text') and part.text is not None and part.text != "":
                        text_parts.append(part)
                
                # テキストパートが見つかった場合
                if text_parts:
                    # 最後のテキストパート以外のすべてのテキストをNoneに設定
                    for i in range(len(text_parts) - 1):
                        text_parts[i].text = None
                    
                    # 最後のテキストパートに新しいテキストを設定
                    text_parts[-1].text = new_text
                else:
                    # テキストパートが見つからない場合、最初のパートにテキストを追加（可能な場合）
                    if hasattr(content, 'parts') and len(content.parts) > 0:
                        if hasattr(content.parts[0], 'text'):
                            content.parts[0].text = new_text
                
                # fix_comprehensive_historyを呼び出して空のパートを削除
                gemini_history = fix_comprehensive_history(gemini_history)
            except Exception as e:
                print(f"モデルメッセージ編集エラー: {str(e)}")
                emit("error", {"message": f"モデルメッセージの編集に失敗しました: {str(e)}"})
                return
            
            break
    
    if not model_found:
        emit("error", {"message": "対応するモデルメッセージが見つかりませんでした"})
        return
    
    # 変更したgemini_historyを保存
    save_gemini_history(user_dir, chat_id, gemini_history)
    
    # クライアントに編集成功を通知
    emit("model_message_edited", {"index": message_index, "new_text": new_text})

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
# 7) 管理
# -----------------------------------------------------------
# 管理者ページ
@app.route("/gemini-admin/")
def admin_page():
    return render_template("admin.html")

# 管理者認証
@app.route("/gemini-admin/auth", methods=["POST"])
def admin_auth():
    if not ADMIN_PASSWORD:
        return jsonify({"status": "error", "message": "管理者パスワードが設定されていません。"}), 403
    
    password = request.json.get("password")
    if not password:
        return jsonify({"status": "error", "message": "パスワードを入力してください。"}), 400
    
    if password == ADMIN_PASSWORD:
        # 認証成功
        session_id = hashlib.sha256(os.urandom(24)).hexdigest()
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        # 古いセッションをクリーンアップ（24時間以上前）
        c.execute("DELETE FROM admin_sessions WHERE created_at < ?", (int(time.time()) - 86400,))
        # 新しいセッションを追加
        c.execute("INSERT INTO admin_sessions (session_id, created_at) VALUES (?, ?)", 
                 (session_id, int(time.time())))
        conn.commit()
        conn.close()
        return jsonify({"status": "success", "session_id": session_id})
    else:
        return jsonify({"status": "error", "message": "パスワードが正しくありません。"}), 401

# 管理者セッション検証
def verify_admin_session(session_id):
    if not session_id:
        return False
    
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT session_id FROM admin_sessions WHERE session_id = ?", (session_id,))
    row = c.fetchone()
    conn.close()
    
    return row is not None

# ユーザー一覧取得
@app.route("/gemini-admin/users", methods=["GET"])
def get_users():
    session_id = request.headers.get("X-Admin-Session")
    if not verify_admin_session(session_id):
        return jsonify({"status": "error", "message": "管理者認証が必要です。"}), 401
    
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT username FROM accounts ORDER BY username")
    users = [row[0] for row in c.fetchall()]
    conn.close()
    
    return jsonify({"status": "success", "users": users})

# パスワードリセット
@app.route("/gemini-admin/reset-password", methods=["POST"])
def reset_password():
    session_id = request.headers.get("X-Admin-Session")
    if not verify_admin_session(session_id):
        return jsonify({"status": "error", "message": "管理者認証が必要です。"}), 401
    
    username = request.json.get("username")
    new_password = request.json.get("password")
    
    if not username or not new_password:
        return jsonify({"status": "error", "message": "ユーザー名とパスワードが必要です。"}), 400
    
    # パスワードハッシュの再計算
    hashed_pw = hash_password(new_password)
    
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("UPDATE accounts SET password = ? WHERE username = ?", (hashed_pw, username))
    if c.rowcount == 0:
        conn.close()
        return jsonify({"status": "error", "message": "ユーザーが見つかりません。"}), 404
    
    conn.commit()
    conn.close()
    
    return jsonify({"status": "success", "message": f"{username}のパスワードをリセットしました。"})

# ユーザーのチャット一覧取得
@app.route("/gemini-admin/user-chats", methods=["GET"])
def get_user_chats():
    session_id = request.headers.get("X-Admin-Session")
    if not verify_admin_session(session_id):
        return jsonify({"status": "error", "message": "管理者認証が必要です。"}), 401
    
    username = request.args.get("username")
    if not username:
        return jsonify({"status": "error", "message": "ユーザー名が必要です。"}), 400
    
    user_dir = get_user_dir(username)
    try:
        past_chats = load_past_chats(user_dir)
        # チャットIDでソート（新しい順）
        sorted_chats = sorted(
            [{"id": k, **v} for k, v in past_chats.items()],
            key=lambda x: float(x["id"]),
            reverse=True
        )
        return jsonify({"status": "success", "chats": sorted_chats})
    except Exception as e:
        return jsonify({"status": "error", "message": f"チャット一覧の取得に失敗しました: {str(e)}"}), 500

# チャットメッセージの取得
@app.route("/gemini-admin/chat-messages", methods=["GET"])
def get_chat_messages():
    session_id = request.headers.get("X-Admin-Session")
    if not verify_admin_session(session_id):
        return jsonify({"status": "error", "message": "管理者認証が必要です。"}), 401
    
    username = request.args.get("username")
    chat_id = request.args.get("chat_id")
    
    if not username or not chat_id:
        return jsonify({"status": "error", "message": "ユーザー名とチャットIDが必要です。"}), 400
    
    user_dir = get_user_dir(username)
    try:
        messages = load_chat_messages(user_dir, chat_id)
        return jsonify({"status": "success", "messages": messages})
    except Exception as e:
        return jsonify({"status": "error", "message": f"メッセージの取得に失敗しました: {str(e)}"}), 500

# セッションログアウト
@app.route("/gemini-admin/logout", methods=["POST"])
def admin_logout():
    session_id = request.headers.get("X-Admin-Session")
    if not session_id:
        return jsonify({"status": "success"})
    
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("DELETE FROM admin_sessions WHERE session_id = ?", (session_id,))
    conn.commit()
    conn.close()
    
    return jsonify({"status": "success"})


# -----------------------------------------------------------
# 7) メイン実行
# -----------------------------------------------------------
if __name__ == "__main__":
    # SQLite初期化
    init_db()

    # geventベースでサーバ起動（geventインストール済みの場合に自動で使用）
    socketio.run(app, debug=False, host="0.0.0.0", port=5000)
