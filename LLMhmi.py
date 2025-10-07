# optimized_local_llm_hmi.py
# -*- coding: utf-8 -*-
"""
優化版 Local LLM HMI（PySide6 單檔 Demo）
改進：
- Enter 送出、Shift+Enter 換行、常用快捷鍵
- 現代化暗色主題、動畫與更佳視覺層次
- 拖放與貼上圖片成附件、即時回饋
- 更完整的錯誤處理與提示
- 響應式佈局 + 進度條 + 匯出/批次匯出
- Prompt Engineering 範本（Zero-Shot / Few-Shot / CoT / 結構化JSON）

後端串流目前為示範，請在 StreamWorker.run() 內替換為實際 llama.cpp/Ollama 串流。
Run:
    pip install PySide6
    python optimized_local_llm_hmi.py
"""
from __future__ import annotations
import sys, os, json, time, zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

from PySide6.QtCore import (
    Qt, QTimer, Signal, QObject, QThread, QEvent
)
from PySide6.QtGui import (
    QDragEnterEvent, QDropEvent, QTextCursor, QKeySequence, QShortcut, QImage
)
from PySide6.QtWidgets import (
    QApplication, QWidget, QMainWindow, QVBoxLayout, QHBoxLayout, QLabel,
    QListWidget, QListWidgetItem, QLineEdit, QPushButton, QSplitter, QTextBrowser,
    QPlainTextEdit, QTabWidget, QGroupBox, QFormLayout, QSpinBox, QDoubleSpinBox,
    QCheckBox, QComboBox, QFileDialog, QMessageBox, QProgressBar, QTableWidget,
    QTableWidgetItem, QAbstractItemView, QFrame, QTextEdit
)

APP_NAME = "Local LLM HMI — 優化版"
DATA_DIR = Path.home() / ".optimized_llm_hmi"
DATA_DIR.mkdir(exist_ok=True)
SESS_PATH = DATA_DIR / "sessions.json"
EXPORT_DIR = DATA_DIR / "exports"
EXPORT_DIR.mkdir(exist_ok=True)
PASTE_DIR = DATA_DIR / "pasted"
PASTE_DIR.mkdir(exist_ok=True)

# 現代化配色方案
COLORS = {
    "bg_primary": "#0d1117",
    "bg_secondary": "#161b22",
    "bg_tertiary": "#21262d",
    "border": "#30363d",
    "text_primary": "#c9d1d9",
    "text_secondary": "#8b949e",
    "accent": "#58a6ff",
    "success": "#3fb950",
    "warning": "#d29922",
    "error": "#f85149",
    "user_bubble": "#1f6feb",
    "assistant_bubble": "#21262d",
}

# ------------------------------ Data Models ------------------------------ #
@dataclass
class Message:
    role: str  # "user" | "assistant" | "system"
    content: str
    ts: float
    attachments: List[str] = field(default_factory=list)

@dataclass
class Session:
    id: str
    title: str = "未命名對話"
    model: str = "llama3.1:q4"
    pinned: bool = False
    tags: List[str] = field(default_factory=list)
    sys_prompt: str = "你是謹慎且專業的助理，回答請使用繁體中文。"
    params: Dict[str, float] = field(default_factory=lambda: {
        "temperature": 0.7,
        "top_p": 0.9,
        "max_new_tokens": 512
    })
    stop: List[str] = field(default_factory=list)
    messages: List[Message] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

# ------------------------- Prompt Engineering --------------------------- #
PROMPT_TECHNIQUES: Dict[str, str] = {
    "Zero-Shot（零樣本）": (
        "角色：{role}\n"
        "任務：{task}\n"
        "目標：{goal}\n"
        "限制：{constraints}\n"
        "輸出格式：{format}\n"
        "語言：{language}\n"
    ),
    "Few-Shot（示例驅動）": (
        "角色：{role}\n"
        "任務：{task}\n"
        "範例：\n{examples}\n\n"
        "請依相同風格處理新輸入。\n"
        "限制：{constraints}\n輸出格式：{format}\n語言：{language}"
    ),
    "Chain-of-Thought（思維鏈）": (
        "角色：{role}\n"
        "任務：{task}\n"
        "請逐步思考並展示推理過程：\n"
        "1. 理解問題\n2. 分析要點\n3. 逐步推理\n4. 得出結論\n"
        "語言：{language}"
    ),
    "結構化輸出（JSON）": (
        "你只輸出有效 JSON（不要加解釋）。\n"
        "遵循此 JSON Schema：\n{schema}\n"
        "任務：{task}\n語言：{language}\n"
    ),
}

DEFAULT_VARIABLES = {
    "role": "資深系統設計師兼教學助理",
    "task": "回應使用者問題，並在需要時提供簡潔步驟或程式碼",
    "goal": "可執行、可複現、適合初學者理解",
    "constraints": "保持簡潔；若不確定請明確說明",
    "format": "條列步驟＋重點說明；若為程式碼請附註解",
    "language": "繁體中文",
    "examples": "[範例輸入]…\n[範例輸出]…",
    "schema": '{"type":"object","properties":{"answer":{"type":"string"}}}'
}

# ------------------------------ Utilities ------------------------------- #

def now_ts() -> float:
    return time.time()


def human_ts(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")


def relative_time(ts: float) -> str:
    now = time.time()
    diff = now - ts
    if diff < 60:
        return "剛剛"
    elif diff < 3600:
        return f"{int(diff/60)} 分鐘前"
    elif diff < 86400:
        return f"{int(diff/3600)} 小時前"
    elif diff < 604800:
        return f"{int(diff/86400)} 天前"
    else:
        return human_ts(ts)


def naive_token_estimate(text: str) -> int:
    cjk_count = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    ascii_count = len(text) - cjk_count
    return max(1, int(cjk_count / 2 + ascii_count / 4))

# ------------------------------ Styled Widgets --------------------------- #
class StyledButton(QPushButton):
    def __init__(self, text: str, primary: bool = False):
        super().__init__(text)
        self.primary = primary
        self._apply_style()

    def _apply_style(self):
        if self.primary:
            self.setStyleSheet(f"""
                QPushButton {{
                    background: {COLORS['accent']};
                    color: #ffffff; border: none; border-radius: 6px;
                    padding: 8px 16px; font-weight: 600; font-size: 14px;
                }}
                QPushButton:hover {{ background: #4c96ed; }}
                QPushButton:pressed {{ background: #3a84db; }}
            """)
        else:
            self.setStyleSheet(f"""
                QPushButton {{
                    background: {COLORS['bg_tertiary']};
                    color: {COLORS['text_primary']};
                    border: 1px solid {COLORS['border']};
                    border-radius: 6px; padding: 8px 16px; font-size: 14px;
                }}
                QPushButton:hover {{ background: {COLORS['border']}; border-color: {COLORS['text_secondary']}; }}
                QPushButton:pressed {{ background: {COLORS['bg_secondary']}; }}
            """)


class AnimatedListWidget(QListWidget):
    def __init__(self):
        super().__init__()
        self.setStyleSheet(f"""
            QListWidget {{
                background: {COLORS['bg_secondary']};
                border: 1px solid {COLORS['border']}; border-radius: 8px; padding: 4px; outline: none;
            }}
            QListWidget::item {{ color: {COLORS['text_primary']}; padding: 10px; margin: 2px; border-radius: 6px; }}
            QListWidget::item:hover {{ background: {COLORS['bg_tertiary']}; }}
            QListWidget::item:selected {{ background: {COLORS['accent']}; color: #ffffff; }}
        """)

# ------------------------------ Worker (Stream) -------------------------- #
class StreamWorker(QObject):
    chunk = Signal(str)
    done = Signal(float)
    error = Signal(str)
    progress = Signal(int)

    def __init__(self, prompt: str, params: Dict[str, float]):
        super().__init__()
        self._stop = False
        self.prompt = prompt
        self.params = params

    def stop(self):
        self._stop = True

    def run(self):
        t0 = time.time()
        try:
            # TODO: 替換為實際串流（llama.cpp 或 Ollama）
            demo_text = (
                "我收到您的訊息了！以下是優化版界面的主要改進：\n\n"
                "🎯 **操作體驗提升**\n"
                "• Enter 直接送出訊息（Shift+Enter 換行）\n"
                "• 支援 Ctrl+Enter 快捷送出\n"
                "• Ctrl+N 新建對話、Ctrl+D 複製對話\n"
                "• ESC 停止生成\n\n"
                "🎨 **視覺設計優化**\n"
                "• 現代化暗色主題，護眼且專業\n"
                "• 平滑動畫和過渡效果\n"
                "• 更清晰的視覺層次\n"
                "• 響應式佈局設計\n\n"
                "⚡ **功能增強**\n"
                "• 智慧提示和錯誤處理\n"
                "• 拖放文件時有視覺回饋\n"
                "• 自動保存和恢復\n"
                "• 更豐富的快捷操作"
            )
            total = len(demo_text)
            for i, ch in enumerate(demo_text):
                if self._stop:
                    break
                self.chunk.emit(ch)
                self.progress.emit(int((i + 1) / total * 100))
                time.sleep(0.01)
            self.done.emit(time.time() - t0)
        except Exception as e:
            self.error.emit(str(e))

# ------------------------------ Main Window ------------------------------ #
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.resize(1400, 820)
        self.setStyleSheet(f"""
    QMainWindow {{
        background: {COLORS['bg_primary']};
    }}
    QWidget {{
        color: {COLORS['text_primary']};
        font-family: 'Segoe UI', 'Microsoft JhengHei', sans-serif;
    }}
    QLabel {{
        color: {COLORS['text_secondary']};
    }}
    QLineEdit, QPlainTextEdit, QTextEdit {{
        background: {COLORS['bg_secondary']};
        border: 1px solid {COLORS['border']};
        border-radius: 6px;
        padding: 8px;
        color: {COLORS['text_primary']};
    }}
    QLineEdit:focus, QPlainTextEdit:focus, QTextEdit:focus {{
        border-color: {COLORS['accent']};
        outline: none;
    }}
    QTabWidget::pane {{
        background: {COLORS['bg_secondary']};
        border: 1px solid {COLORS['border']};
        border-radius: 8px;
    }}
    QTabBar::tab {{
        background: {COLORS['bg_tertiary']};
        color: {COLORS['text_secondary']};
        padding: 8px 16px;
        margin-right: 4px;
        border-radius: 6px 6px 0 0;
    }}
    QTabBar::tab:selected {{
        background: {COLORS['accent']};
        color: #ffffff;
    }}
    QSplitter::handle {{
        background: {COLORS['border']};
        width: 2px;
    }}
    QScrollBar:vertical {{
        background: {COLORS['bg_secondary']};
        width: 12px;
        border-radius: 6px;
    }}
    QScrollBar::handle:vertical {{
        background: {COLORS['border']};
        border-radius: 6px;
        min-height: 20px;
    }}
    QScrollBar::handle:vertical:hover {{
        background: {COLORS['text_secondary']};
    }}
""")


        self.sessions: Dict[str, Session] = {}
        self.current_sid: Optional[str] = None
        self.pending_files: List[Path] = []
        self.stream_thread: Optional[QThread] = None
        self.stream_worker: Optional[StreamWorker] = None
        self.last_latency: float = 0.0
        self.last_error: Optional[str] = None
        self.is_generating: bool = False

        self._load_sessions()
        if not self.sessions:
            sid = self._new_session()
            self.current_sid = sid
        else:
            self.current_sid = max(self.sessions.keys(), key=lambda x: self.sessions[x].created_at)

        self._build_ui()
        self._setup_shortcuts()
        self._refresh_all()

        # 自動保存
        self.auto_save_timer = QTimer(self)
        self.auto_save_timer.timeout.connect(self._save_sessions)
        self.auto_save_timer.start(30000)

    # ------------------------- Persistence ------------------------- #
    def _load_sessions(self):
        if SESS_PATH.exists():
            try:
                raw = json.loads(SESS_PATH.read_text(encoding="utf-8"))
                for sid, s in raw.items():
                    sess = Session(
                        id=sid,
                        title=s.get("title", "未命名對話"),
                        model=s.get("model", "llama3.1:q4"),
                        pinned=s.get("pinned", False),
                        tags=s.get("tags", []),
                        sys_prompt=s.get("sys_prompt", "你是謹慎且專業的助理，回答請使用繁體中文。"),
                        params=s.get("params", {"temperature":0.7, "top_p":0.9, "max_new_tokens":512}),
                        stop=s.get("stop", []),
                        created_at=s.get("created_at", time.time())
                    )
                    for m in s.get("messages", []):
                        sess.messages.append(Message(**m))
                    self.sessions[sid] = sess
            except Exception as e:
                print(f"載入會話失敗: {e}")
                self.sessions = {}

    def _save_sessions(self):
        out = {}
        for sid, s in self.sessions.items():
            out[sid] = {
                "title": s.title,
                "model": s.model,
                "pinned": s.pinned,
                "tags": s.tags,
                "sys_prompt": s.sys_prompt,
                "params": s.params,
                "stop": s.stop,
                "messages": [m.__dict__ for m in s.messages],
                "created_at": s.created_at
            }
        try:
            SESS_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:
            print(f"保存會話失敗: {e}")

    # ------------------------- UI Build --------------------------- #
    def _build_ui(self):
        central = QWidget(); self.setCentralWidget(central)
        root = QHBoxLayout(central); root.setContentsMargins(0, 0, 0, 0); root.setSpacing(0)
        self.splitter = QSplitter(Qt.Horizontal)
        root.addWidget(self.splitter)

        left = self._build_left_panel()
        center = self._build_center_panel()
        right = self._build_right_panel()

        self.splitter.addWidget(left)
        self.splitter.addWidget(center)
        self.splitter.addWidget(right)
        self.splitter.setSizes([280, 760, 360])

    def _build_left_panel(self) -> QWidget:
        left = QWidget(); left.setMinimumWidth(260)
        left.setStyleSheet(f"background: {COLORS['bg_secondary']};")
        lv = QVBoxLayout(left); lv.setContentsMargins(12, 12, 12, 12)
        title_label = QLabel("💬 會話列表"); title_label.setStyleSheet(f"font-size:16px;font-weight:bold;color:{COLORS['text_primary']};margin-bottom:8px;")
        self.search_box = QLineEdit(); self.search_box.setPlaceholderText("🔍 搜尋會話...")
        self.search_box.textChanged.connect(self._refresh_session_list)
        self.session_list = AnimatedListWidget(); self.session_list.itemSelectionChanged.connect(self._on_session_selected)
        btn_row = QHBoxLayout()
        self.btn_new = StyledButton("➕ 新建", primary=True)
        self.btn_dup = StyledButton("📋 複製")
        self.btn_del = StyledButton("🗑️ 刪除")
        self.btn_new.clicked.connect(self._on_new_session)
        self.btn_dup.clicked.connect(self._on_dup_session)
        self.btn_del.clicked.connect(self._on_del_session)
        btn_row.addWidget(self.btn_new); btn_row.addWidget(self.btn_dup); btn_row.addWidget(self.btn_del)
        model_label = QLabel("🤖 快速切換模型"); model_label.setStyleSheet("margin-top:12px;font-weight:bold;")
        self.quick_model = QComboBox(); self.quick_model.addItems([
            "llama3.1:q4","llama3.1:7b-q5","qwen2.5:7b-q4","mistral:7b-instruct","gemma2:9b","phi3:mini"
        ])
        self.quick_model.currentTextChanged.connect(self._on_quick_model)
        self.quick_model.setStyleSheet(f"""
            QComboBox {{ background: {COLORS['bg_tertiary']}; border:1px solid {COLORS['border']}; border-radius:6px; padding:8px; color:{COLORS['text_primary']}; }}
            QComboBox:hover {{ border-color: {COLORS['accent']}; }}
            QComboBox::drop-down {{ border: none; }}
        """)
        lv.addWidget(title_label)
        lv.addWidget(self.search_box)
        lv.addWidget(self.session_list, 1)
        lv.addLayout(btn_row)
        lv.addWidget(model_label)
        lv.addWidget(self.quick_model)
        return left

    def _build_center_panel(self) -> QWidget:
        center = QWidget(); center.setStyleSheet(f"background:{COLORS['bg_primary']};")
        cv = QVBoxLayout(center); cv.setContentsMargins(16, 16, 16, 16)
        self.breadcrumb = QLabel("載入中...")
        self.breadcrumb.setStyleSheet(f"""
            background:{COLORS['bg_secondary']}; border:1px solid {COLORS['border']};
            border-radius:8px; padding:12px; color:{COLORS['text_secondary']}; font-size:13px;
        """)
        self.chat = QTextBrowser(); self.chat.setOpenExternalLinks(True)
        self.chat.setStyleSheet(f"""
            QTextBrowser {{ background:{COLORS['bg_secondary']}; border:1px solid {COLORS['border']}; border-radius:12px; padding:16px; }}
        """)
        self.progress_bar = QProgressBar(); self.progress_bar.setVisible(False)
        self.progress_bar.setStyleSheet(f"""
            QProgressBar {{ background:{COLORS['bg_tertiary']}; border:1px solid {COLORS['border']}; border-radius:4px; height:6px; }}
            QProgressBar::chunk {{ background:{COLORS['accent']}; border-radius:3px; }}
        """)
        attach_frame = QFrame(); attach_frame.setStyleSheet(f"""
            QFrame {{ background:{COLORS['bg_tertiary']}; border:1px dashed {COLORS['border']}; border-radius:8px; padding:8px; }}
        """)
        attach_layout = QHBoxLayout(attach_frame)
        self.attach_label = QLabel("📎 拖曳檔案到此處或貼上圖片"); self.attach_label.setStyleSheet(f"color:{COLORS['text_secondary']};")
        self.btn_clear_attach = StyledButton("清空"); self.btn_clear_attach.clicked.connect(self._clear_attachments); self.btn_clear_attach.setVisible(False)
        attach_layout.addWidget(self.attach_label, 1); attach_layout.addWidget(self.btn_clear_attach)
        self.input = SmartTextEdit(self); self.input.setPlaceholderText("💬 輸入訊息... (Enter 送出 / Shift+Enter 換行)"); self.input.setMaximumHeight(120); self.input.setAcceptDrops(True)
        btn_row = QHBoxLayout()
        self.btn_send = StyledButton("📤 送出", primary=True)
        self.btn_stop = StyledButton("⏹️ 停止"); self.btn_stop.setVisible(False)
        self.btn_regen = StyledButton("🔄 重新生成")
        self.btn_export = StyledButton("💾 匯出")
        self.btn_send.clicked.connect(self._on_send)
        self.btn_stop.clicked.connect(self._on_stop)
        self.btn_regen.clicked.connect(self._on_regen)
        self.btn_export.clicked.connect(self._on_export_dialog)
        self.token_label = QLabel("📊 Tokens: 0/8192"); self.token_label.setStyleSheet(f"color:{COLORS['text_secondary']};font-size:12px;")
        btn_row.addWidget(self.btn_send); btn_row.addWidget(self.btn_stop); btn_row.addWidget(self.btn_regen); btn_row.addStretch(); btn_row.addWidget(self.token_label); btn_row.addWidget(self.btn_export)
        cv.addWidget(self.breadcrumb)
        cv.addWidget(self.chat, 1)
        cv.addWidget(self.progress_bar)
        cv.addWidget(attach_frame)
        cv.addWidget(self.input)
        cv.addLayout(btn_row)
        return center

    def _build_right_panel(self) -> QWidget:
        right = QTabWidget(); right.setMinimumWidth(340); right.setStyleSheet(f"background:{COLORS['bg_secondary']};")
        right.addTab(self._build_params_tab(), "⚙️ 參數")
        right.addTab(self._build_model_tab(), "🤖 模型")
        right.addTab(self._build_prompt_tab(), "✍️ Prompt")
        right.addTab(self._build_export_tab(), "💾 匯出")
        right.addTab(self._build_logs_tab(), "📊 日誌")
        return right

    def _build_params_tab(self) -> QWidget:
        w = QWidget(); f = QFormLayout(w); f.setSpacing(12)
        self.sp_temp = QDoubleSpinBox(); self.sp_temp.setRange(0.0, 2.0); self.sp_temp.setSingleStep(0.1); self.sp_temp.setValue(0.7); self.sp_temp.valueChanged.connect(self._update_params)
        self.sp_top_p = QDoubleSpinBox(); self.sp_top_p.setRange(0.0, 1.0); self.sp_top_p.setSingleStep(0.05); self.sp_top_p.setValue(0.9); self.sp_top_p.valueChanged.connect(self._update_params)
        self.sp_max_new = QSpinBox(); self.sp_max_new.setRange(16, 8192); self.sp_max_new.setValue(512); self.sp_max_new.valueChanged.connect(self._update_params)
        self.ed_stop = QLineEdit(); self.ed_stop.setPlaceholderText("以逗號分隔，如: Human:, AI:")
        self.ck_memory = QCheckBox("啟用記憶（將重要摘要寫入系統提示）")
        f.addRow("🌡️ Temperature", self.sp_temp)
        f.addRow("📊 Top-p", self.sp_top_p)
        f.addRow("📝 最大生成長度", self.sp_max_new)
        f.addRow("🛑 停止詞", self.ed_stop)
        f.addRow(self.ck_memory)
        preset_label = QLabel("預設組合："); preset_combo = QComboBox(); preset_combo.addItems(["創意寫作","精確回答","程式碼生成","平衡模式"]); preset_combo.currentTextChanged.connect(self._apply_preset)
        f.addRow(preset_label, preset_combo)
        return w

    def _build_model_tab(self) -> QWidget:
        w = QWidget(); f = QFormLayout(w); f.setSpacing(12)
        self.ed_endpoint = QLineEdit("http://localhost:11434")
        self.cb_model = QComboBox(); self.cb_model.addItems(["llama3.1:q4","llama3.1:8b-q5","qwen2.5:7b-q4","mistral:7b-instruct","gemma2:9b","phi3:mini"])
        self.cb_quant = QComboBox(); self.cb_quant.addItems(["q4","q5","q8","fp16"]) 
        self.sp_threads = QSpinBox(); self.sp_threads.setRange(1, 64); self.sp_threads.setValue(os.cpu_count() or 8)
        self.lb_vram = QLabel("GPU/VRAM：偵測中..."); self.lb_vram.setStyleSheet(f"color:{COLORS['text_secondary']};")
        self.btn_test = StyledButton("🔌 測試連線"); self.btn_test.clicked.connect(self._on_test_endpoint)
        f.addRow("🌐 端點", self.ed_endpoint)
        f.addRow("🤖 模型", self.cb_model)
        f.addRow("📊 量化", self.cb_quant)
        f.addRow("⚡ 執行緒數", self.sp_threads)
        f.addRow(self.lb_vram)
        f.addRow(self.btn_test)
        return w

    def _build_prompt_tab(self) -> QWidget:
        w = QWidget(); v = QVBoxLayout(w)
        row1 = QHBoxLayout(); self.cb_tech = QComboBox(); self.cb_tech.addItems(list(PROMPT_TECHNIQUES.keys()))
        self.btn_apply_tech = StyledButton("套用範本"); self.btn_apply_tech.clicked.connect(self._on_apply_prompt_template)
        row1.addWidget(QLabel("📚 提示技巧：")); row1.addWidget(self.cb_tech, 1); row1.addWidget(self.btn_apply_tech)
        self.var_table = QTableWidget(0, 2); self.var_table.setHorizontalHeaderLabels(["變數","值"]); self.var_table.horizontalHeader().setStretchLastSection(True); self.var_table.setEditTriggers(QAbstractItemView.AllEditTriggers)
        self.var_table.setStyleSheet(f"""
            QTableWidget {{ background:{COLORS['bg_tertiary']}; border:1px solid {COLORS['border']}; border-radius:6px; }}
            QHeaderView::section {{ background:{COLORS['bg_secondary']}; color:{COLORS['text_primary']}; padding:8px; border:none; }}
        """)
        self._populate_vars(DEFAULT_VARIABLES)
        self.sys_prompt_edit = QPlainTextEdit(); self.sys_prompt_edit.setPlaceholderText("系統提示（System Prompt）..."); self.sys_prompt_edit.setMaximumHeight(150)
        self.preview = QPlainTextEdit(); self.preview.setReadOnly(True); self.preview.setPlaceholderText("範本渲染預覽..."); self.preview.setMaximumHeight(150); self.preview.setStyleSheet(f"background:{COLORS['bg_primary']}; border:1px solid {COLORS['border']};")
        row2 = QHBoxLayout(); self.btn_preview = StyledButton("👁️ 預覽"); self.btn_set_sys = StyledButton("✅ 套用"); self.btn_reset_sys = StyledButton("🔄 重置")
        self.btn_preview.clicked.connect(self._on_preview_render)
        self.btn_set_sys.clicked.connect(self._on_set_sys_prompt)
        self.btn_reset_sys.clicked.connect(self._on_reset_sys)
        row2.addWidget(self.btn_preview); row2.addWidget(self.btn_set_sys); row2.addWidget(self.btn_reset_sys)
        v.addLayout(row1)
        v.addWidget(QLabel("📝 變數設定："))
        v.addWidget(self.var_table, 1)
        v.addWidget(QLabel("⚙️ 系統提示："))
        v.addWidget(self.sys_prompt_edit)
        v.addWidget(QLabel("👁️ 預覽："))
        v.addWidget(self.preview)
        v.addLayout(row2)
        return w

    def _build_export_tab(self) -> QWidget:
        w = QWidget(); v = QVBoxLayout(w)
        current_group = QGroupBox("當前會話"); current_layout = QHBoxLayout()
        self.btn_exp_txt = StyledButton("📄 TXT"); self.btn_exp_md = StyledButton("📝 MD"); self.btn_exp_json = StyledButton("🔧 JSON")
        self.btn_exp_txt.clicked.connect(lambda: self._export_current("txt"))
        self.btn_exp_md.clicked.connect(lambda: self._export_current("md"))
        self.btn_exp_json.clicked.connect(lambda: self._export_current("json"))
        current_layout.addWidget(self.btn_exp_txt); current_layout.addWidget(self.btn_exp_md); current_layout.addWidget(self.btn_exp_json)
        current_group.setLayout(current_layout)
        batch_group = QGroupBox("批次匯出"); batch_layout = QVBoxLayout()
        self.batch_list = QListWidget(); self.batch_list.setSelectionMode(QAbstractItemView.MultiSelection)
        self.batch_list.setStyleSheet(f"QListWidget {{ background:{COLORS['bg_tertiary']}; border:1px solid {COLORS['border']}; border-radius:6px; }}")
        self.btn_batch_zip = StyledButton("📦 匯出為 ZIP", primary=True); self.btn_batch_zip.clicked.connect(self._batch_export_zip)
        batch_layout.addWidget(QLabel("選擇要匯出的會話："))
        batch_layout.addWidget(self.batch_list)
        batch_layout.addWidget(self.btn_batch_zip)
        batch_group.setLayout(batch_layout)
        v.addWidget(current_group)
        v.addWidget(batch_group)
        return w

    def _build_logs_tab(self) -> QWidget:
        w = QWidget(); f = QFormLayout(w); f.setSpacing(12)
        self.lb_ctx_ratio = QLabel("—"); self.lb_tok_stats = QLabel("—"); self.lb_latency = QLabel("—"); self.lb_error = QLabel("無錯誤")
        for label in [self.lb_ctx_ratio, self.lb_tok_stats, self.lb_latency, self.lb_error]:
            label.setStyleSheet(f"background:{COLORS['bg_tertiary']}; border:1px solid {COLORS['border']}; border-radius:4px; padding:8px; color:{COLORS['text_primary']};")
        self.btn_clear_logs = StyledButton("🗑️ 清除日誌"); self.btn_clear_logs.clicked.connect(self._clear_logs)
        f.addRow("📊 上下文佔比", self.lb_ctx_ratio)
        f.addRow("🔢 Token 統計", self.lb_tok_stats)
        f.addRow("⏱️ 延遲（秒）", self.lb_latency)
        f.addRow("❌ 錯誤訊息", self.lb_error)
        f.addRow(self.btn_clear_logs)
        return w

    # ----------------------- Keyboard Shortcuts ----------------------- #
    def _setup_shortcuts(self):
        QShortcut(QKeySequence("Ctrl+N"), self, self._on_new_session)
        QShortcut(QKeySequence("Ctrl+D"), self, self._on_dup_session)
        QShortcut(QKeySequence("Ctrl+S"), self, self._save_sessions)
        QShortcut(QKeySequence("Ctrl+E"), self, self._on_export_dialog)
        QShortcut(QKeySequence("Escape"), self, self._on_stop)
        QShortcut(QKeySequence("F5"), self, self._on_regen)

    # ----------------------- UI Refresh Helpers ----------------------- #
    def _refresh_all(self):
        self._refresh_session_list()
        self._refresh_batch_list()
        self._refresh_chat()
        self._refresh_breadcrumb()
        self._refresh_logs()
        self._update_token_count()

    def _refresh_session_list(self):
        text = self.search_box.text().strip().lower()
        self.session_list.clear()
        items = sorted(self.sessions.items(), key=lambda kv: (not kv[1].pinned, -kv[1].messages[-1].ts if kv[1].messages else -kv[1].created_at))
        for sid, s in items:
            if text and (text not in s.title.lower() and not any(text in t.lower() for t in s.tags)):
                continue
            icon = "📌 " if s.pinned else "💬 "
            time_str = relative_time(s.messages[-1].ts) if s.messages else "新會話"
            display_text = f"{icon}{s.title}\n    {time_str}"
            it = QListWidgetItem(display_text); it.setData(Qt.UserRole, sid)
            if self.current_sid == sid: it.setSelected(True)
            self.session_list.addItem(it)

    def _refresh_batch_list(self):
        self.batch_list.clear()
        for sid, s in self.sessions.items():
            it = QListWidgetItem(f"📄 {s.title}"); it.setData(Qt.UserRole, sid)
            self.batch_list.addItem(it)

    def _refresh_chat(self):
        s = self.sessions[self.current_sid]
        html_parts = []
        if s.sys_prompt:
            summary = (s.sys_prompt[:50] + "...") if len(s.sys_prompt) > 50 else s.sys_prompt
            html_parts.append(self._bubble("⚙️ System", summary, align="center", muted=True))
        for m in s.messages:
            if m.role == "user":
                who = "👤 您"; align = "right"; bg_color = COLORS['user_bubble']
            else:
                who = "🤖 AI 助理"; align = "left"; bg_color = COLORS['assistant_bubble']
            content = m.content
            if m.attachments:
                attach_text = "\n\n📎 附件：\n" + "\n".join(f"• {a}" for a in m.attachments)
                content += attach_text
            html_parts.append(self._bubble(who, content, align=align, bg_color=bg_color))
        self.chat.setHtml("".join(html_parts))
        self.chat.moveCursor(QTextCursor.End)

    def _refresh_breadcrumb(self):
        s = self.sessions[self.current_sid]
        sys_short = (s.sys_prompt[:40] + "...") if len(s.sys_prompt) > 40 else s.sys_prompt
        info = (f"🤖 模型：{s.model} | "
                f"🌡️ 溫度：{s.params.get('temperature', 0.7)} | "
                f"💬 訊息：{len(s.messages)} | "
                f"⚙️ System：{sys_short}")
        self.breadcrumb.setText(info)
        self.sp_temp.setValue(float(s.params.get("temperature", 0.7)))
        self.sp_top_p.setValue(float(s.params.get("top_p", 0.9)))
        self.sp_max_new.setValue(int(s.params.get("max_new_tokens", 512)))
        self.ed_stop.setText(",".join(s.stop))
        self.quick_model.setCurrentText(s.model)
        self.cb_model.setCurrentText(s.model)

    def _refresh_logs(self):
        s = self.sessions[self.current_sid]
        total_txt = s.sys_prompt + "\n" + "\n".join(m.content for m in s.messages)
        toks = naive_token_estimate(total_txt)
        ratio_val = min(100, int(toks/8192*100))
        ratio = f"{ratio_val}% (≈{toks}/8192 tokens)"
        color = COLORS['success'] if ratio_val < 50 else (COLORS['warning'] if ratio_val < 80 else COLORS['error'])
        self.lb_ctx_ratio.setText(ratio)
        self.lb_ctx_ratio.setStyleSheet(f"background:{COLORS['bg_tertiary']}; border:1px solid {color}; border-radius:4px; padding:8px; color:{color};")
        self.lb_tok_stats.setText(f"估計 tokens：{toks}")
        self.lb_latency.setText(f"{self.last_latency:.2f} 秒")
        self.lb_error.setText(self.last_error or "無錯誤")

    def _update_token_count(self):
        s = self.sessions[self.current_sid]
        total_txt = s.sys_prompt + "\n" + "\n".join(m.content for m in s.messages)
        toks = naive_token_estimate(total_txt)
        ratio = toks / 8192
        color = COLORS['success'] if ratio < 0.5 else (COLORS['warning'] if ratio < 0.8 else COLORS['error'])
        self.token_label.setText(f"📊 Tokens: {toks}/8192")
        self.token_label.setStyleSheet(f"color:{color}; font-size:12px;")

    # --------------------------- Bubbles ---------------------------- #
    def _bubble(self, title: str, text: str, align: str = "left", muted: bool=False, bg_color: str=None) -> str:
        if not bg_color:
            bg_color = COLORS['bg_tertiary'] if muted else COLORS['assistant_bubble']
        margin_left = "auto" if align == "right" else "0"
        margin_right = "0" if align == "right" else "auto"
        return f"""
        <div style='margin:12px 0; display:flex; justify-content:{align};'>
            <div style='
                max-width:75%; background:{bg_color}; color:{'#ffffff' if align=='right' else COLORS['text_primary']};
                border-radius:18px; padding:12px 16px; margin-left:{margin_left}; margin-right:{margin_right};
                box-shadow:0 2px 8px rgba(0,0,0,0.1);
            '>
                <div style='opacity:.8;font-size:12px;margin-bottom:6px;'>{title}</div>
                <div style='white-space:pre-wrap;font-size:14px;line-height:1.6;'>{text}</div>
            </div>
        </div>
        """

    # --------------------------- Events ----------------------------- #
    def _on_session_selected(self):
        items = self.session_list.selectedItems()
        if not items: return
        sid = items[0].data(Qt.UserRole)
        if sid == self.current_sid: return
        self.current_sid = sid
        self._refresh_all(); self._save_sessions()

    def _on_new_session(self):
        sid = self._new_session(); self.current_sid = sid
        self._refresh_all(); self._save_sessions(); self.input.setFocus()

    def _new_session(self) -> str:
        sid = f"s{int(time.time()*1000)}"; self.sessions[sid] = Session(id=sid); return sid

    def _on_dup_session(self):
        s = self.sessions[self.current_sid]
        sid = f"s{int(time.time()*1000)}"
        self.sessions[sid] = Session(
            id=sid, title=s.title + "（複製）", model=s.model, pinned=s.pinned, tags=s.tags[:],
            sys_prompt=s.sys_prompt, params=dict(s.params), stop=list(s.stop),
            messages=[Message(m.role, m.content, m.ts, m.attachments[:]) for m in s.messages]
        )
        self.current_sid = sid
        self._refresh_all(); self._save_sessions()

    def _on_del_session(self):
        if len(self.sessions) <= 1:
            QMessageBox.warning(self, "無法刪除", "至少要保留一個會話。"); return
        reply = QMessageBox.question(self, "確認刪除", f"確定要刪除「{self.sessions[self.current_sid].title}」嗎？")
        if reply == QMessageBox.Yes:
            del self.sessions[self.current_sid]
            self.current_sid = next(iter(self.sessions))
            self._refresh_all(); self._save_sessions()

    def _on_quick_model(self, model: str):
        s = self.sessions[self.current_sid]; s.model = model
        self._refresh_breadcrumb(); self._save_sessions()

    def _update_params(self):
        s = self.sessions[self.current_sid]
        s.params["temperature"] = float(self.sp_temp.value())
        s.params["top_p"] = float(self.sp_top_p.value())
        s.params["max_new_tokens"] = int(self.sp_max_new.value())
        self._refresh_breadcrumb()

    def _apply_preset(self, preset: str):
        presets = {
            "創意寫作": {"temperature": 1.2, "top_p": 0.95, "max_new_tokens": 1024},
            "精確回答": {"temperature": 0.3, "top_p": 0.8, "max_new_tokens": 512},
            "程式碼生成": {"temperature": 0.2, "top_p": 0.9, "max_new_tokens": 2048},
            "平衡模式": {"temperature": 0.7, "top_p": 0.9, "max_new_tokens": 512}
        }
        if preset in presets:
            p = presets[preset]
            self.sp_temp.setValue(p["temperature"])
            self.sp_top_p.setValue(p["top_p"])
            self.sp_max_new.setValue(p["max_new_tokens"])

    def _on_test_endpoint(self):
        endpoint = self.ed_endpoint.text()
        QMessageBox.information(self, "測試連線", f"測試端點：{endpoint}\n\n此為示範，實際請連接 Ollama/llama.cpp")

    def _on_send(self):
        text = self.input.toPlainText().strip()
        if not text: return
        self.input.clear()
        s = self.sessions[self.current_sid]
        s.params["temperature"] = float(self.sp_temp.value())
        s.params["top_p"] = float(self.sp_top_p.value())
        s.params["max_new_tokens"] = int(self.sp_max_new.value())
        s.stop = [x.strip() for x in self.ed_stop.text().split(',') if x.strip()]
        attachments = []
        if self.pending_files:
            attachments = [str(p) for p in self.pending_files]
            self._clear_attachments()
        s.messages.append(Message("user", text, now_ts(), attachments))
        if s.title == "未命名對話" and len(s.messages) == 1:
            s.title = (text[:20] + "...") if len(text) > 20 else text
        self._refresh_all(); self._save_sessions()
        self._start_stream_worker(self._compose_prompt_for_backend(text))

    def _compose_prompt_for_backend(self, user_text: str) -> str:
        s = self.sessions[self.current_sid]
        parts = [f"[SYSTEM]\n{s.sys_prompt}"]
        for m in s.messages[-10:]:
            role = m.role.upper(); parts.append(f"[{role}]\n{m.content}")
        return "\n\n".join(parts)

    def _start_stream_worker(self, prompt: str):
        self._on_stop()
        self.is_generating = True
        self.btn_send.setVisible(False); self.btn_stop.setVisible(True)
        self.progress_bar.setVisible(True); self.progress_bar.setValue(0)
        self.stream_thread = QThread(); self.stream_worker = StreamWorker(prompt, self.sessions[self.current_sid].params)
        self.stream_worker.moveToThread(self.stream_thread)
        self.stream_thread.started.connect(self.stream_worker.run)
        self.stream_worker.chunk.connect(self._on_stream_chunk)
        self.stream_worker.progress.connect(self.progress_bar.setValue)
        self.stream_worker.done.connect(self._on_stream_done)
        self.stream_worker.error.connect(self._on_stream_error)
        self.stream_thread.start()
        self.sessions[self.current_sid].messages.append(Message("assistant", "", now_ts()))

    def _on_stream_chunk(self, ch: str):
        s = self.sessions[self.current_sid]
        if s.messages and s.messages[-1].role == "assistant":
            s.messages[-1].content += ch
            self._refresh_chat(); self._update_token_count()

    def _on_stream_done(self, latency: float):
        self.last_latency = latency
        self._teardown_stream(); self._refresh_logs(); self._save_sessions()
        self.progress_bar.setValue(100)
        QTimer.singleShot(1000, lambda: self.progress_bar.setVisible(False))

    def _on_stream_error(self, err: str):
        self.last_error = err
        self._teardown_stream(); self._refresh_logs(); self._save_sessions()
        QMessageBox.warning(self, "錯誤", f"生成時發生錯誤：\n{err}")

    def _teardown_stream(self):
        if self.stream_worker: self.stream_worker.stop()
        if self.stream_thread:
            self.stream_thread.quit(); self.stream_thread.wait()
        self.stream_worker = None; self.stream_thread = None
        self.is_generating = False
        self.btn_send.setVisible(True); self.btn_stop.setVisible(False)

    def _on_stop(self):
        if self.is_generating and self.stream_worker:
            self.stream_worker.stop()

    def _on_regen(self):
        s = self.sessions[self.current_sid]
        for i in range(len(s.messages) - 1, -1, -1):
            if s.messages[i].role == "user":
                s.messages = s.messages[:i+1]
                self._refresh_chat()
                self._start_stream_worker(self._compose_prompt_for_backend(s.messages[i].content))
                return
        QMessageBox.information(self, "重新生成", "找不到上一則使用者訊息。")

    def _on_export_dialog(self):
        right_tabs = self.splitter.widget(2)
        right_tabs.setCurrentIndex(3)  # Export tab

    def _export_current(self, kind: str):
        s = self.sessions[self.current_sid]
        fname = f"{s.title or 'session'}_{int(time.time())}.{kind}"
        path = EXPORT_DIR / fname
        try:
            if kind == "txt":
                lines = []
                for m in s.messages:
                    lines.append(f"[{human_ts(m.ts)}] {m.role}:")
                    lines.append(m.content)
                    if m.attachments:
                        lines.append(f"附件: {', '.join(m.attachments)}")
                    lines.append("")
                path.write_text("\n".join(lines), encoding="utf-8")
            elif kind == "md":
                md = [
                    f"# {s.title}",
                    f"*匯出時間: {human_ts(time.time())}*",
                    f"*模型: {s.model}*",
                    "", "---", "",
                    f"**系統提示:**\n> {s.sys_prompt}",
                    "", "---", ""
                ]
                for m in s.messages:
                    who = "使用者" if m.role == "user" else ("AI 助理" if m.role=="assistant" else "系統")
                    md.append(f"### {who} - {human_ts(m.ts)}")
                    md.append("")
                    md.append(m.content)
                    if m.attachments:
                        md.append("")
                        md.append("**附件:**")
                        for a in m.attachments:
                            md.append(f"- {a}")
                    md.append("")
                path.write_text("\n".join(md), encoding="utf-8")
            elif kind == "json":
                obj = {
                    "session": {
                        "id": s.id, "title": s.title, "model": s.model,
                        "created_at": s.created_at, "sys_prompt": s.sys_prompt,
                        "params": s.params, "messages": [m.__dict__ for m in s.messages]
                    }
                }
                path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
            QMessageBox.information(self, "匯出完成", f"已輸出到：{path}")
        except Exception as e:
            QMessageBox.warning(self, "匯出失敗", str(e))

    def _batch_export_zip(self):
        selected = [it.data(Qt.UserRole) for it in self.batch_list.selectedItems()]
        if not selected:
            QMessageBox.information(self, "批次匯出", "請先在列表選取會話。"); return
        zip_path = EXPORT_DIR / f"sessions_{int(time.time())}.zip"
        try:
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
                for sid in selected:
                    s = self.sessions[sid]
                    data = {
                        "title": s.title, "model": s.model, "sys_prompt": s.sys_prompt,
                        "params": s.params, "tags": s.tags, "messages": [m.__dict__ for m in s.messages]
                    }
                    z.writestr(f"{s.title or sid}.json", json.dumps(data, ensure_ascii=False, indent=2))
            QMessageBox.information(self, "批次匯出完成", f"已輸出 zip：{zip_path}")
        except Exception as e:
            QMessageBox.warning(self, "批次匯出失敗", str(e))

    # --------------------- Prompt Tab Handlers ---------------------- #
    def _populate_vars(self, d: Dict[str, str]):
        self.var_table.setRowCount(0)
        for k, v in d.items():
            r = self.var_table.rowCount(); self.var_table.insertRow(r)
            self.var_table.setItem(r, 0, QTableWidgetItem(k))
            self.var_table.setItem(r, 1, QTableWidgetItem(v))

    def _gather_vars(self) -> Dict[str, str]:
        out = {}
        for r in range(self.var_table.rowCount()):
            k_item = self.var_table.item(r, 0); v_item = self.var_table.item(r, 1)
            if not k_item or not v_item: continue
            k = (k_item.text() or "").strip(); v = (v_item.text() or "").strip()
            if k: out[k] = v
        return out

    def _on_preview_render(self):
        tech = self.cb_tech.currentText(); tpl = PROMPT_TECHNIQUES.get(tech, "")
        vars = {**DEFAULT_VARIABLES, **self._gather_vars()}
        try:
            rendered = tpl.format(**vars)
        except KeyError as e:
            rendered = f"[變數缺失] {e}"
        self.preview.setPlainText(rendered)

    def _on_apply_prompt_template(self):
        self._on_preview_render()
        self.sys_prompt_edit.setPlainText(self.preview.toPlainText())

    def _on_set_sys_prompt(self):
        s = self.sessions[self.current_sid]
        new_sys = self.sys_prompt_edit.toPlainText().strip()
        if new_sys:
            s.sys_prompt = new_sys
            self._refresh_chat(); self._refresh_breadcrumb(); self._save_sessions()
        else:
            QMessageBox.information(self, "System Prompt", "內容是空的，已忽略。")

    def _on_reset_sys(self):
        self.sys_prompt_edit.setPlainText("你是謹慎且專業的助理，回答請使用繁體中文。")

    # ---------------------- Attachments (Drag/Drop & Paste) ------------- #
    def add_attachments(self, paths: List[Path]):
        for p in paths:
            if p.exists() and p not in self.pending_files:
                self.pending_files.append(p)
        self._render_attach_label()

    def add_pasted_image(self, qimage: QImage):
        ts = int(time.time()*1000)
        out_path = PASTE_DIR / f"pasted_{ts}.png"
        qimage.save(str(out_path))
        self.add_attachments([out_path])

    def _clear_attachments(self):
        self.pending_files.clear(); self._render_attach_label()

    def _render_attach_label(self):
        if self.pending_files:
            self.attach_label.setText("附件：\n" + "\n".join(str(p) for p in self.pending_files))
            self.btn_clear_attach.setVisible(True)
        else:
            self.attach_label.setText("📎 拖曳檔案到此處或貼上圖片")
            self.btn_clear_attach.setVisible(False)

    # ------------------------------ Logs -------------------------------- #
    def _clear_logs(self):
        self.last_latency = 0.0; self.last_error = None
        self._refresh_logs()

# ---------------------- Custom TextEdit (Enter & Drops) ------------------ #
class SmartTextEdit(QTextEdit):
    def __init__(self, main: MainWindow):
        super().__init__()
        self.main = main
        self.setAcceptDrops(True)

    def keyPressEvent(self, e):
        if e.key() in (Qt.Key_Return, Qt.Key_Enter):
            if e.modifiers() & Qt.ShiftModifier:
                # 換行
                super().keyPressEvent(e)
            else:
                # 送出
                self.main._on_send()
        else:
            super().keyPressEvent(e)

    def dragEnterEvent(self, e: QDragEnterEvent):
        if e.mimeData().hasUrls() or e.mimeData().hasImage():
            e.acceptProposedAction()
        else:
            super().dragEnterEvent(e)

    def dropEvent(self, e: QDropEvent):
        urls = e.mimeData().urls()
        paths = [Path(u.toLocalFile()) for u in urls if u.isLocalFile()]
        if paths:
            self.main.add_attachments(paths)
        elif e.mimeData().hasImage():
            img = e.mimeData().imageData()
            if isinstance(img, QImage):
                self.main.add_pasted_image(img)
        else:
            super().dropEvent(e)

    def insertFromMimeData(self, source):
        # 支援貼上圖片成附件
        if source.hasImage():
            img = source.imageData()
            if isinstance(img, QImage):
                self.main.add_pasted_image(img)
                return
        super().insertFromMimeData(source)


# ------------------------------- Main ------------------------------------ #
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    win = MainWindow(); win.show()
    sys.exit(app.exec())
