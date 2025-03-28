// Vue.js アプリケーション
const { createApp, ref } = Vue;

createApp({
	setup() {
		// ====== 認証関連 ======
		const isLoggedIn = ref(false);
		const adminPassword = ref('');
		const loginError = ref('');
		const sessionId = ref('');

		// ====== ユーザー管理 ======
		const users = ref([]);
		const selectedUser = ref(null);
		const userChats = ref([]);
		const viewingChat = ref(null);
		const chatMessages = ref([]);
		const loading = ref(false);

		// ====== パスワードリセット ======
		const showPasswordModal = ref(false);
		const modalUser = ref('');
		const newPassword = ref('');
		const confirmPassword = ref('');
		const modalError = ref('');

		// ====== 通知 ======
		const showNotification = ref(false);
		const notificationMessage = ref('');
		const notificationType = ref('success');

		// 初期化
		const initialize = () => {
			// セッションIDの取得（ローカルストレージから）
			const storedSessionId = localStorage.getItem('adminSessionId');
			if (storedSessionId) {
				sessionId.value = storedSessionId;
				isLoggedIn.value = true;
				loadUsers();
			}
		};

		// ログイン処理
		const login = async () => {
			if (!adminPassword.value) {
				loginError.value = 'パスワードを入力してください';
				return;
			}

			try {
				loginError.value = '';
				const response = await fetch('/gemini-admin/auth', {
					method: 'POST',
					headers: {
						'Content-Type': 'application/json',
					},
					body: JSON.stringify({ password: adminPassword.value }),
				});

				const data = await response.json();
				if (data.status === 'success') {
					sessionId.value = data.session_id;
					localStorage.setItem('adminSessionId', data.session_id);
					isLoggedIn.value = true;
					adminPassword.value = '';
					loadUsers();
				} else {
					loginError.value = data.message || 'ログインに失敗しました';
				}
			} catch (error) {
				loginError.value = 'サーバーエラーが発生しました';
				console.error(error);
			}
		};

		// ログアウト処理
		const logout = async () => {
			try {
				await fetch('/gemini-admin/logout', {
					method: 'POST',
					headers: {
						'X-Admin-Session': sessionId.value,
					},
				});
			} catch (error) {
				console.error('ログアウトエラー:', error);
			} finally {
				sessionId.value = '';
				localStorage.removeItem('adminSessionId');
				isLoggedIn.value = false;
				resetState();
			}
		};

		// ユーザー一覧の読み込み
		const loadUsers = async () => {
			if (!sessionId.value) return;

			loading.value = true;
			try {
				const response = await fetch('/gemini-admin/users', {
					headers: {
						'X-Admin-Session': sessionId.value,
					},
				});

				const data = await response.json();
				if (data.status === 'success') {
					users.value = data.users;
				} else {
					showNotificationMessage(data.message || 'ユーザー一覧の取得に失敗しました', 'error');
				}
			} catch (error) {
				showNotificationMessage('ユーザー一覧の取得中にエラーが発生しました', 'error');
				console.error(error);
			} finally {
				loading.value = false;
			}
		};

		// ユーザーのチャット一覧表示
		const viewUserChats = async (username) => {
			selectedUser.value = username;
			loading.value = true;

			try {
				const response = await fetch(`/gemini-admin/user-chats?username=${encodeURIComponent(username)}`, {
					headers: {
						'X-Admin-Session': sessionId.value,
					},
				});

				const data = await response.json();
				if (data.status === 'success') {
					userChats.value = data.chats;
				} else {
					showNotificationMessage(data.message || 'チャット一覧の取得に失敗しました', 'error');
				}
			} catch (error) {
				showNotificationMessage('チャット一覧の取得中にエラーが発生しました', 'error');
				console.error(error);
			} finally {
				loading.value = false;
			}
		};

		// チャットメッセージの表示
		const viewChatMessages = async (chat) => {
			viewingChat.value = chat;
			loading.value = true;

			try {
				const response = await fetch(`/gemini-admin/chat-messages?username=${encodeURIComponent(selectedUser.value)}&chat_id=${encodeURIComponent(chat.id)}`, {
					headers: {
						'X-Admin-Session': sessionId.value,
					},
				});

				const data = await response.json();
				if (data.status === 'success') {
					chatMessages.value = data.messages;
				} else {
					showNotificationMessage(data.message || 'メッセージの取得に失敗しました', 'error');
				}
			} catch (error) {
				showNotificationMessage('メッセージの取得中にエラーが発生しました', 'error');
				console.error(error);
			} finally {
				loading.value = false;
			}
		};

		// ユーザー一覧に戻る
		const backToUserList = () => {
			selectedUser.value = null;
			viewingChat.value = null;
			userChats.value = [];
			chatMessages.value = [];
		};

		// ユーザーのチャット一覧に戻る
		const backToUserChats = () => {
			viewingChat.value = null;
			chatMessages.value = [];
		};

		// パスワードリセットモーダルの表示
		const showResetPasswordModal = (username) => {
			modalUser.value = username;
			newPassword.value = '';
			confirmPassword.value = '';
			modalError.value = '';
			showPasswordModal.value = true;
		};

		// パスワードリセット処理
		const resetPassword = async () => {
			if (!newPassword.value) {
				modalError.value = '新しいパスワードを入力してください';
				return;
			}

			if (newPassword.value !== confirmPassword.value) {
				modalError.value = 'パスワードが一致しません';
				return;
			}

			try {
				const response = await fetch('/gemini-admin/reset-password', {
					method: 'POST',
					headers: {
						'Content-Type': 'application/json',
						'X-Admin-Session': sessionId.value,
					},
					body: JSON.stringify({
						username: modalUser.value,
						password: newPassword.value,
					}),
				});

				const data = await response.json();
				if (data.status === 'success') {
					showNotificationMessage(data.message || 'パスワードをリセットしました', 'success');
					showPasswordModal.value = false;
				} else {
					modalError.value = data.message || 'パスワードのリセットに失敗しました';
				}
			} catch (error) {
				modalError.value = 'サーバーエラーが発生しました';
				console.error(error);
			}
		};

		// 通知メッセージの表示
		const showNotificationMessage = (message, type = 'success') => {
			notificationMessage.value = message;
			notificationType.value = type;
			showNotification.value = true;

			// 3秒後に非表示
			setTimeout(() => {
				showNotification.value = false;
			}, 3000);
		};

		// 状態のリセット
		const resetState = () => {
			selectedUser.value = null;
			viewingChat.value = null;
			userChats.value = [];
			chatMessages.value = [];
			users.value = [];
		};

		// 日付のフォーマット（タイムスタンプ -> 人間が読める形式）
		const formatDate = (timestamp) => {
			const date = new Date(parseFloat(timestamp) * 1000);
			return date.toLocaleString();
		};

		// 初期化を実行
		initialize();

		return {
			// 状態
			isLoggedIn,
			adminPassword,
			loginError,
			users,
			selectedUser,
			userChats,
			viewingChat,
			chatMessages,
			loading,
			showPasswordModal,
			modalUser,
			newPassword,
			confirmPassword,
			modalError,
			showNotification,
			notificationMessage,
			notificationType,

			// メソッド
			login,
			logout,
			loadUsers,
			viewUserChats,
			viewChatMessages,
			backToUserList,
			backToUserChats,
			showResetPasswordModal,
			resetPassword,
			formatDate,
		};
	}
}).mount('#app');
