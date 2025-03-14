import os
import argparse
from dotenv import load_dotenv
from google import genai

load_dotenv()
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')

def list_files(client):
    """ファイル一覧を表示する関数"""
    print("=== ファイル一覧 ===")
    files = list(client.files.list())
    
    if not files:
        print("ファイルは存在しません。")
        return []
    
    for i, file in enumerate(files, 1):
        print(f"{i}. {file.name} (作成日時: {file.create_time})")
    
    return files

def delete_files(client, files, dry_run=True):
    """ファイルを削除する関数"""
    if not files:
        print("削除するファイルはありません。")
        return
    
    print(f"\n=== {'削除予定' if dry_run else '削除'} ファイル ===")
    for i, file in enumerate(files, 1):
        if dry_run:
            print(f"{i}. {file.name} (削除予定)")
        else:
            try:
                response = client.files.delete(name=file.name)
                print(f"{i}. {file.name} → 削除成功")
            except Exception as e:
                print(f"{i}. {file.name} → 削除失敗: {e}")

def main():
    parser = argparse.ArgumentParser(description='GenAI File APIを使用してファイル一覧取得・削除を行うツール')
    parser.add_argument('--delete', action='store_true', help='ファイルを実際に削除します（指定しない場合は削除予定の表示のみ）')
    parser.add_argument('--api-key', help='GenAI API Keyを指定します（環境変数GOOGLE_API_KEYが設定されていない場合に必要）')
    args = parser.parse_args()
    
    try:
        # API Keyの設定
        if args.api_key:
            genai.configure(api_key=args.api_key)
        elif 'GOOGLE_API_KEY' not in os.environ:
            print("エラー: API Keyが必要です。--api-keyオプションで指定するか、GOOGLE_API_KEY環境変数を設定してください。")
            return
        
        # クライアントの初期化
        client = genai.Client()
        
        # ファイル一覧取得
        files = list_files(client)
        
        # ファイル削除
        if args.delete:
            input_text = input("\n全てのファイルを削除します。よろしいですか？ [y/N]: ")
            if input_text.lower() == 'y':
                delete_files(client, files, dry_run=False)
            else:
                print("削除をキャンセルしました。")
        else:
            print("\n注意: 実際に削除するには --delete オプションを付けて実行してください。")
            delete_files(client, files, dry_run=True)
            
    except Exception as e:
        print(f"エラーが発生しました: {e}")

if __name__ == "__main__":
    main()