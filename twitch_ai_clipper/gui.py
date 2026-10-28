"""Графический интерфейс (PyQt6, тёмная тема): запуск пайплайна без командной строки.

Открывается командой ``python -m twitch_ai_clipper gui`` или двойным кликом ``gui.bat``.
Собирает аргументы CLI из формы и запускает их через QProcess, показывая живой лог.
Настройки между запусками сохраняются в QSettings.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from PyQt6.QtCore import QProcess, QSettings, Qt
from PyQt6.QtGui import QFont, QTextCursor
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

ORG_NAME = "twitch-ai-clipper"
APP_NAME = "gui"

# Маркер "рабочего" запуска exe: GUI запускает CLI-exe с этим флагом,
# чтобы CLI работал на строках и кормил лог QProcess.
GUI_WORKER_MARK = "--gui-worker"

COMMANDS: tuple[tuple[str, str], ...] = (
    ("Полный прогон (run)", "run"),
    ("Только чат (chat)", "chat"),
    ("Расшифровка (transcribe)", "transcribe"),
    ("Найти моменты (detect)", "detect"),
    ("Нарезать клипы (clip)", "clip"),
    ("Пересобрать отчёт (report)", "report"),
    ("Записи канала (list)", "list"),
    ("Сервер галереи (serve)", "serve"),
)

WHISPER_MODELS = ("", "tiny", "base", "small", "medium", "large-v3")
QUALITIES = ("1080p60", "best", "1080p", "720p60", "720p", "480p", "360p", "audio")
LLM_CHOICES = ("auto", "none", "ollama", "openai")
PRESETS = ("", "sparse", "dense", "full")
MODES = ("default", "toparachevt")
BROWSERS = ("", "chrome", "firefox", "edge", "brave", "opera")

# Какие флаги допустимы каждой команде (лишний флаг упастил бы argparse CLI).
COMMON_FLAGS = frozenset({
    "output_dir", "whisper_model", "language", "llm", "llm_model",
    "quality", "cookies_from_browser", "limit_seconds", "force",
})
ALL_FLAGS = COMMON_FLAGS | {
    "preset", "mode", "max_clips", "min_moments", "min_gap", "min_clip_length",
    "clip_padding", "vertical", "burn_subs", "precise_cut", "no_clips", "dump_chat",
    "max_clip_length", "top_candidates",
}
COMMAND_FLAGS: dict[str, frozenset[str]] = {
    "run": ALL_FLAGS,
    "chat": COMMON_FLAGS | {"dump_chat"},
    "transcribe": COMMON_FLAGS,
    "detect": ALL_FLAGS - {"vertical", "burn_subs", "precise_cut", "dump_chat", "clip_padding"},
    "clip": COMMON_FLAGS | {"mode", "vertical", "burn_subs", "precise_cut"},
    "report": {"output_dir", "mode"},
    "list": set(),
    "serve": {"output_dir"},
}
TARGET_LABELS: dict[str, str] = {
    "run": "VOD / ссылка / канал / файл:",
    "chat": "VOD / ссылка / канал / файл:",
    "transcribe": "VOD / ссылка / канал / файл:",
    "detect": "VOD / ссылка / канал / файл:",
    "clip": "VOD / ссылка / канал / файл:",
    "report": "ID VOD (папка в output):",
    "list": "Имя канала:",
    "serve": "",
}

DARK_STYLE = """
QWidget { background-color: #0e0e10; color: #efeff1; font-size: 13px; }
QMainWindow, #panel { background-color: #0e0e10; }
#card { background-color: #18181b; border: 1px solid #2a2a2d; border-radius: 10px; }
QLabel { background: transparent; }
QLabel#title { font-size: 17px; font-weight: 600; }
QLabel#section { color: #adadb8; font-size: 12px; }
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
    background-color: #1f1f23; color: #efeff1;
    border: 1px solid #3a3a3d; border-radius: 6px; padding: 5px 8px;
}
QLineEdit:focus, QComboBox:focus { border-color: #9147ff; }
QComboBox::drop-down { border: none; width: 22px; }
QComboBox::down-arrow {
    image: none; border-left: 5px solid transparent; border-right: 5px solid transparent;
    border-top: 6px solid #adadb8; width: 0; height: 0; margin-right: 8px;
}
QComboBox QAbstractItemView {
    background-color: #1f1f23; color: #efeff1; selection-background-color: #9147ff;
}
QCheckBox { background: transparent; spacing: 8px; color: #dedee3; }
QCheckBox::indicator {
    width: 16px; height: 16px; border-radius: 4px;
    border: 1px solid #3a3a3d; background-color: #1f1f23;
}
QCheckBox::indicator:checked { background-color: #9147ff; border-color: #9147ff; }
QPushButton {
    background-color: #232326; color: #efeff1; border: 1px solid #3a3a3d;
    border-radius: 8px; padding: 7px 16px;
}
QPushButton:hover { background-color: #2f2f33; border-color: #4a4a4f; }
QPushButton:pressed { background-color: #1f1f23; }
QPushButton:disabled { color: #6b6b70; background-color: #1c1c1f; }
QPushButton#primary {
    background-color: #9147ff; color: white; font-weight: 600; border: none; min-width: 110px;
}
QPushButton#primary:hover { background-color: #a356ff; }
QPushButton#primary:disabled { background-color: #47315f; color: #b9a7cc; }
QPushButton#danger { background-color: #3a1d26; border-color: #77394b; color: #ff7b9d; }
QPlainTextEdit {
    background-color: #0d0d0f; color: #cadede; color: #cfdede;
    border: 1px solid #2a2a2d; border-radius: 8px;
    font-family: Consolas, 'Courier New', monospace; font-size: 12px; padding: 6px;
}
QScrollBar:vertical { background: #18181b; width: 12px; border-radius: 6px; }
QScrollBar::handle:vertical { background: #3a3a3d; border-radius: 6px; min-height: 30px; }
QScrollBar::handle:vertical:hover { background: #55555a; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QStatusBar { background: #18181b; color: #adadb8; border-top: 1px solid #2a2a2d; }
"""


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Twitch AI Clipper")
        self.resize(1080, 740)
        self.settings = QSettings(ORG_NAME, APP_NAME)
        self.process: QProcess | None = None

        self._build_ui()
        self._load_settings()
        self._sync_command_fields()

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        outer = QHBoxLayout(central)
        outer.setContentsMargins(14, 14, 14, 14)
        outer.setSpacing(14)

        # ---------------- левая колонка: форма
        left = QFrame()
        left.setObjectName("panel")
        left.setFixedWidth(460)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(16, 14, 16, 16)
        left_layout.setSpacing(10)

        title = QLabel("AI-нарезка стримов Twitch")
        title.setObjectName("title")
        left_layout.addWidget(title)

        subtitle = QLabel("Заполните поля и нажмите «Запустить» — команды вводить не нужно.")
        subtitle.setObjectName("section")
        subtitle.setWordWrap(True)
        left_layout.addWidget(subtitle)

        form_frame = QFrame()
        form_frame.setObjectName("card")
        form = QGridLayout(form_frame)
        form.setContentsMargins(14, 14, 14, 14)
        form.setSpacing(9)
        form.setColumnStretch(1, 1)  # поле растягивается, метка — нет
        form.setColumnMinimumWidth(1, 220)
        form_row = 0

        def add_row(label_text: str, widget: Any) -> None:
            nonlocal form_row
            form.addWidget(QLabel(label_text), form_row, 0, alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)  # noqa: E501
            form.addWidget(widget, form_row, 1)
            form_row += 1

        self.command_combo = QComboBox()
        self.command_labels = [label for label, _ in COMMANDS]
        self.command_combo.addItems(self.command_labels)
        current = next((c for c in COMMANDS if c[1] == "run"), COMMANDS[0])
        self.command_combo.setCurrentText(current[0])
        self.command_combo.currentTextChanged.connect(self._on_command_change)
        add_row("Команда:", self.command_combo)

        self.target = QLineEdit()
        self.target.setPlaceholderText("123456789 / twitch.tv/канал / C:\\video\\clip.mp4")
        target_row = QHBoxLayout()
        target_row.setContentsMargins(0, 0, 0, 0)
        target_row.setSpacing(6)
        target_row.addWidget(self.target, 1)
        pick_file = QPushButton("Файл…")
        pick_file.setFixedWidth(56)
        pick_file.clicked.connect(self._pick_file)
        target_row.addWidget(pick_file)
        self.target_label = QLabel(TARGET_LABELS["run"])
        target_wrap = QWidget()
        target_wrap.setLayout(target_row)
        form.addWidget(self.target_label, form_row, 0)
        form.addWidget(target_wrap, form_row, 1)
        form_row += 1

        self.output_dir = QLineEdit("output")
        self.output_dir.setPlaceholderText("output")
        add_row("Папка результатов:", self.output_dir)

        def combo(values: tuple[str, ...], current_value: str) -> QComboBox:
            widget = QComboBox()
            widget.addItems(list(values))
            if current_value:
                widget.setCurrentText(current_value)
            return widget

        self.whisper_model = combo(WHISPER_MODELS, "medium")
        add_row("Модель Whisper:", self.whisper_model)

        self.language = QLineEdit("ru")
        add_row("Язык стрима:", self.language)

        self.llm = combo(LLM_CHOICES, "auto")
        add_row("Провайдер LLM:", self.llm)

        self.llm_model = QLineEdit()
        self.llm_model.setPlaceholderText("пусто = из config.toml")
        add_row("Модель LLM:", self.llm_model)

        self.quality = combo(QUALITIES, "1080p60")
        add_row("Качество клипов:", self.quality)

        self.mode = combo(MODES, "toparachevt")
        add_row("Мод оформления:", self.mode)

        self.preset = combo(PRESETS, "")
        add_row("Пресет детекта:", self.preset)

        self.cookies = combo(BROWSERS, "")
        add_row("Cookies из браузера:", self.cookies)

        self.limit_seconds = QLineEdit("0")
        add_row("Лимит анализа, сек (0=весь):", self.limit_seconds)

        self.max_clips = QLineEdit("0")
        add_row("Максимум клипов (0=конф.):", self.max_clips)

        self.min_moments = QLineEdit("0")
        add_row("Минимум моментов (0=конф.):", self.min_moments)

        self.min_gap = QLineEdit("0")
        add_row("Мин. зазор, сек (0=конф.):", self.min_gap)

        self.min_clip_length = QLineEdit("0")
        add_row("Мин. длина клипа, сек (0=конф.):", self.min_clip_length)

        left_layout.addWidget(form_frame)

        checks_frame = QFrame()
        checks_frame.setObjectName("card")
        checks_grid = QGridLayout(checks_frame)
        checks_grid.setContentsMargins(14, 12, 14, 12)
        checks_grid.setSpacing(8)
        self.checks: dict[str, QCheckBox] = {}
        check_items = [
            ("no_clips", "Без нарезки", "--no-clips"),
            ("vertical", "Вертикальные 9:16", "--vertical"),
            ("burn_subs", "Субтитры в клипы", "--burn-subs"),
            ("precise_cut", "Точная резка", "--precise-cut"),
            ("dump_chat", "Лог чата в txt", "--dump-chat"),
            ("force", "Игнорировать кеш", "--force"),
        ]
        for offset, (key, label, flag) in enumerate(check_items):
            check = QCheckBox(label)
            check.setToolTip("Флаг CLI: " + flag)
            self.checks[key] = check
            checks_grid.addWidget(check, offset % 3, offset // 3)
        left_layout.addWidget(checks_frame)
        left_layout.addStretch(1)

        # ---------------- правая колонка: лог + кнопки
        right = QFrame()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(10)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(5000)
        self.log.setFont(QFont("Consolas", 9))
        self.log.setPlaceholderText("Лог запуска появится здесь…")
        right_layout.addWidget(self.log, 1)

        buttons = QHBoxLayout()
        buttons.setSpacing(8)
        self.run_button = QPushButton("▶  Запустить")
        self.run_button.setObjectName("primary")
        self.run_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.run_button.clicked.connect(self._run)
        self.stop_button = QPushButton("■  Стоп")
        self.stop_button.setObjectName("danger")
        self.stop_button.setEnabled(False)
        self.stop_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.stop_button.clicked.connect(self._stop)
        clear = QPushButton("Очистить лог")
        clear.clicked.connect(self.log.clear)
        open_gallery = QPushButton("Открыть галерею")
        open_gallery.clicked.connect(self._open_gallery)
        open_output = QPushButton("Открыть папку output")
        open_output.clicked.connect(self._open_output)
        buttons.addWidget(self.run_button)
        buttons.addWidget(self.stop_button)
        buttons.addWidget(clear)
        buttons.addStretch(1)
        buttons.addWidget(open_gallery)
        buttons.addWidget(open_output)
        right_layout.addLayout(buttons)

        outer.addWidget(left)
        outer.addWidget(right, 1)

        self.statusBar().showMessage("Готов к запуску")

    # ------------------------------------------------------- поля (значения)
    def _num(self, line_edit: QLineEdit) -> float:
        try:
            return float(line_edit.text().replace(",", ".").strip())
        except (TypeError, ValueError):
            return 0.0

    def _flag_text(self, line_edit: QLineEdit) -> str:
        return line_edit.text().strip()

    def _command_code(self) -> str:
        return dict(COMMANDS).get(self.command_combo.currentText(), "run")

    # ------------------------------------------------------ смена команды
    def _on_command_change(self, *_: Any) -> None:
        self._sync_command_fields()
        self._log(f"Режим: {self.command_combo.currentText()}")

    def _sync_command_fields(self) -> None:
        code = self._command_code()
        allowed = COMMAND_FLAGS[code]
        widget_flags: dict[str, list[Any]] = {
            "output_dir": [self.output_dir],
            "whisper_model": [self.whisper_model],
            "language": [self.language],
            "llm": [self.llm],
            "llm_model": [self.llm_model],
            "quality": [self.quality],
            "mode": [self.mode],
            "preset": [self.preset],
            "cookies_from_browser": [self.cookies],
            "limit_seconds": [self.limit_seconds],
            "max_clips": [self.max_clips],
            "min_moments": [self.min_moments],
            "min_gap": [self.min_gap],
            "min_clip_length": [self.min_clip_length],
            "no_clips": [self.checks["no_clips"]],
            "vertical": [self.checks["vertical"]],
            "burn_subs": [self.checks["burn_subs"]],
            "precise_cut": [self.checks["precise_cut"]],
            "dump_chat": [self.checks["dump_chat"]],
            "force": [self.checks["force"]],
        }
        for key, widgets in widget_flags.items():
            state = True if key in allowed else False
            for widget in widgets:
                widget.setEnabled(state)
        target_label = TARGET_LABELS[code] or "Цель: (не нужно)"
        self.target_label.setText(target_label)
        if code == "serve":
            self.target.setEnabled(False)
            self.target.setPlaceholderText("не нужно для этой команды")
        else:
            self.target.setEnabled(True)
            if hasattr(self, "target_placeholder"):
                self.target.setPlaceholderText(self.target_placeholder)

    # ------------------------------------------------------------- настройки
    def _load_settings(self) -> None:
        s = self.settings
        def text(key: str, widget: Any, default: str = "") -> None:
            widget.setCurrentText(str(s.value(key, default))) if hasattr(widget, "setCurrentText") else widget.setText(str(s.value(key, default)))
        text("command_label", self.command_combo)
        if self.command_combo.currentText() not in self.command_labels:
            self.command_combo.setCurrentText(self.command_labels[0])
        text("target", self.target)
        text("output_dir", self.output_dir, "output")
        text("whisper_model", self.whisper_model, "medium")
        text("language", self.language, "ru")
        text("llm", self.llm, "auto")
        text("llm_model", self.llm_model)
        text("quality", self.quality, "1080p60")
        text("text_mode", self.mode, "toparachevt")
        text("preset", self.preset, "")
        text("cookies_from_browser", self.cookies, "")
        for key, widget in (
            ("limit_seconds", self.limit_seconds), ("max_clips", self.max_clips),
            ("min_moments", self.min_moments), ("min_gap", self.min_gap),
            ("min_clip_length", self.min_clip_length),
        ):
            widget.setText(str(s.value(key, "0")))
        for key, check in self.checks.items():
            check.setChecked(str(s.value("check_" + key, "false")).lower() == "true")

    def _save_settings(self) -> None:
        s = self.settings
        s.setValue("command_label", self.command_combo.currentText())
        s.setValue("target", self.target.text())
        s.setValue("output_dir", self.output_dir.text())
        s.setValue("whisper_model", self.whisper_model.currentText())
        s.setValue("language", self.language.text())
        s.setValue("llm", self.llm.currentText())
        s.setValue("llm_model", self.llm_model.text())
        s.setValue("quality", self.quality.currentText())
        s.setValue("text_mode", self.mode.currentText())
        s.setValue("preset", self.preset.currentText())
        s.setValue("cookies_from_browser", self.cookies.currentText())
        for key, widget in (
            ("limit_seconds", self.limit_seconds), ("max_clips", self.max_clips),
            ("min_moments", self.min_moments), ("min_gap", self.min_gap),
            ("min_clip_length", self.min_clip_length),
        ):
            s.setValue(key, widget.text())
        for key, check in self.checks.items():
            s.setValue("check_" + key, "true" if check.isChecked() else "false")

    # ------------------------------------------------------------------ сборка команды
    def _build_cmd(self) -> list[str]:
        code = self._command_code()
        if getattr(sys, "frozen", False):
            # В exe нет "python -m twitch_ai_clipper": запускаем соседний CLI-exe,
            # который умеет читать свой stdout через QProcess.
            cli_exe = Path(sys.executable).with_name("TwitchAIClipperCli.exe")
            if not cli_exe.exists():
                QMessageBox.warning(
                    self, "Нет CLI-exe",
                    f"Рядом с програмой не найден TwitchAIClipperCli.exe (нужен для запуска задач):\n{cli_exe}",
                )
                return []
            cmd = [str(cli_exe), launcher.GUI_WORKER_MARK, code]
        else:
            cmd = [sys.executable, "-u", "-m", "twitch_ai_clipper", code]
        allowed = COMMAND_FLAGS[code]
        if code != "serve":
            target = self.target.text().strip()
            if target:
                cmd.append(target)
            if code == "list":
                return cmd
        if "output_dir" in allowed and self.output_dir.text().strip():
            cmd += ["-o", self.output_dir.text().strip()]
        if "whisper_model" in allowed and self.whisper_model.currentText():
            cmd += ["--whisper-model", self.whisper_model.currentText()]
        if "language" in allowed and self._flag_text(self.language):
            cmd += ["--language", self._flag_text(self.language)]
        if "llm" in allowed:
            cmd += ["--llm", self.llm.currentText()]
        if "llm_model" in allowed and self._flag_text(self.llm_model):
            cmd += ["--llm-model", self._flag_text(self.llm_model)]
        if "quality" in allowed:
            cmd += ["--quality", self.quality.currentText()]
        if "mode" in allowed and self.mode.currentText():
            cmd += ["--mode", self.mode.currentText()]
        if "preset" in allowed and self.preset.currentText():
            cmd += ["--preset", self.preset.currentText()]
        if "cookies_from_browser" in allowed and self.cookies.currentText():
            cmd += ["--cookies-from-browser", self.cookies.currentText()]
        if "limit_seconds" in allowed and self._num(self.limit_seconds) > 0:
            cmd += ["--limit-seconds", str(int(self._num(self.limit_seconds)))]
        if "max_clips" in allowed and self._num(self.max_clips) > 0:
            cmd += ["--max-clips", str(int(self._num(self.max_clips)))]
        if "min_moments" in allowed and self._num(self.min_moments) > 0:
            cmd += ["--min-moments", str(int(self._num(self.min_moments)))]
        if "min_gap" in allowed and self._num(self.min_gap) > 0:
            cmd += ["--min-gap", str(self._num(self.min_gap))]
        if "min_clip_length" in allowed and self._num(self.min_clip_length) > 0:
            cmd += ["--min-clip-length", str(self._num(self.min_clip_length))]
        for key, flag in (
            ("vertical", "--vertical"), ("burn_subs", "--burn-subs"),
            ("precise_cut", "--precise-cut"), ("no_clips", "--no-clips"),
            ("dump_chat", "--dump-chat"), ("force", "--force"),
        ):
            if key in allowed and self.checks[key].isChecked():
                cmd.append(flag)
        return cmd

    # ------------------------------------------------------------------ запуск
    def _run(self) -> None:
        if self.process is not None:
            return
        code = self._command_code()
        if code != "serve" and not self.target.text().strip():
            QMessageBox.warning(self, "Нет цели", "Укажите цель: ссылку/ID VOD, канал или файл.")
            return
        self._save_settings()
        self._sync_command_fields()
        cmd = self._build_cmd()
        self._log("$ " + " ".join(cmd))

        self.process = QProcess(self)
        self.process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        env = [
            line for line in QProcess.systemEnvironment()
            if not (line.startswith("PYTHONIOENCODING=") or line.startswith("PYTHONUTF8="))
        ]
        env += ["PYTHONIOENCODING=utf-8", "PYTHONUTF8=1"]  # логи клипов без кракозябр
        self.process.setEnvironment(env)
        self.process.readyReadStandardOutput.connect(self._read_output)
        self.process.finished.connect(self._finish)
        self.process.start(cmd[0], cmd[1:])
        self.run_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.statusBar().showMessage("Выполняется…")

    def _read_output(self) -> None:
        if self.process is None:
            return
        data = self.process.readAllStandardOutput()
        text = bytes(data).decode("utf-8", errors="replace")
        self._log(text)

    def _finish(self, exit_code: int, _status: Any = None) -> None:
        self.process = None
        self.run_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        if exit_code == 0:
            self.statusBar().showMessage("Готово ✅", 10000)
        else:
            self.statusBar().showMessage(f"Завершено с кодом {exit_code} — подробности в логе", 10000)

    def _stop(self) -> None:
        if self.process is None:
            return
        if sys.platform == "win32":
            # /T убивает и подпроцессы (yt-dlp, ffmpeg), иначе останутся висеть.
            import subprocess

            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(self.process.processId())],
                capture_output=True, check=False,
            )
        else:
            self.process.terminate()
        self.statusBar().showMessage("Останавливаем…")
        self.process.waitForFinished(8000)

    # ------------------------------------------------------------------ прочее
    def _log(self, text: str) -> None:
        if not text:
            return
        self.log.appendPlainText(text)
        self.log.moveCursor(QTextCursor.MoveOperation.End)

    def _pick_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Локальное видео",
            "",
            "Видео/аудио (*.mp4 *.mkv *.mov *.avi *.webm *.m4a *.mp3 *.wav);;Все файлы (*.*)",
        )
        if path:
            self.target.setText(path)

    def _open_output(self) -> None:
        raw = self.output_dir.text().strip() or "output"
        path = Path(raw)
        if not path.is_absolute():
            path = Path.cwd() / path
        if path.exists():
            os.startfile(str(path))  # noqa: S606
        else:
            QMessageBox.information(self, "Нет папки", f"Папка ещё не создана: {path}")

    def _open_gallery(self) -> None:
        import webbrowser

        webbrowser.open("http://127.0.0.1:8765/")

    def closeEvent(self, event: Any) -> None:  # noqa: N802  (название из Qt)
        self._save_settings()
        running = self.process is not None and self.process.state() != QProcess.ProcessState.NotRunning
        if running:
            answer = QMessageBox.question(
                self, "Идёт процесс", "Процесс ещё выполняется. Прервать и выйти?",
                QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel,
            )
            if answer != QMessageBox.StandardButton.Ok:
                event.ignore()
                return
            self._stop()
        event.accept()


def launch() -> int:
    """Создаёт окно и запускает цикл Qt."""
    try:
        app = QApplication(sys.argv)
    except Exception as exc:  # noqa: BLE001
        print(f"Не удалось открыть окно GUI: {exc}")
        return 1
    app.setStyle("Fusion")
    app.setStyleSheet(DARK_STYLE)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(launch())
