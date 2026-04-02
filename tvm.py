import sys
import os
import ctypes
import hashlib
import sqlite3
import shutil
import json
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import cv2
import numpy as np

from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QMainWindow,
    QCheckBox, QPushButton, QSpinBox, QLabel, QFileDialog,
    QListWidget, QListWidgetItem, QInputDialog, QSplitter,
    QMessageBox, QMenu, QTabWidget, QProgressBar, QGroupBox, 
    QTextEdit, QComboBox, QTreeView, QAbstractItemView
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize, QFileSystemWatcher, QTimer, QDir
from PyQt6.QtGui import QIcon, QPixmap, QFont, QFileSystemModel, QAction

CONFIG_FILE = "config.json"
# 썸네일 그리드 설정 (1980x1980 고화질)
THUMBNAIL_SIZE = 1980  
CELL_SIZE = THUMBNAIL_SIZE // 3  
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.m4v', '.ts', '.mts'}

def resource_path(relative_path):
    try: base_path = sys._MEIPASS
    except Exception: base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

def safe_imwrite(file_path, img, params=None):
    """백신 차단 및 한글 경로 문제를 방지하는 안전 저장 함수"""
    try:
        ext = os.path.splitext(file_path)[1] or ".jpg"
        ret, buf = cv2.imencode(ext, img, params)
        if ret:
            with open(file_path, 'wb') as f:
                f.write(buf)
            return True
    except Exception as e:
        print(f"저장 에러: {e}")
    return False

# ==================== 데이터베이스 및 로직 ====================
class VideoDatabase:
    def __init__(self, folder_path):
        thumb_dir = os.path.join(folder_path, '.thumbs')
        os.makedirs(thumb_dir, exist_ok=True)
        self.conn = sqlite3.connect(os.path.join(thumb_dir, 'videos.db'))
        self.conn.execute('CREATE TABLE IF NOT EXISTS videos (path TEXT PRIMARY KEY, name TEXT, size INTEGER, modified REAL, thumbnail_path TEXT)')
        self.conn.commit()
    def insert_or_update(self, path, name, size, modified, thumbnail_path):
        self.conn.execute('INSERT OR REPLACE INTO videos VALUES (?, ?, ?, ?, ?)', (path, name, size, modified, thumbnail_path))
        self.conn.commit()
    def close(self): self.conn.close()

class ThumbnailWorker(QThread):
    progress_signal = pyqtSignal(str, str, dict)
    finished_signal = pyqtSignal()
    folder_progress_signal = pyqtSignal(int, int)

    def __init__(self, folder_path, max_cores=4):
        super().__init__()
        self.folder_path = folder_path
        self.running = True
        self.max_cores = max_cores

    def run(self):
        valid_files = [f for f in os.listdir(self.folder_path) if os.path.splitext(f)[1].lower() in VIDEO_EXTENSIONS]
        
        thumb_dir = os.path.join(self.folder_path, '.thumbs')
        os.makedirs(thumb_dir, exist_ok=True)
        try: db = VideoDatabase(self.folder_path)
        except: db = None

        # 1. 존재하는 비디오 파일들의 예상 썸네일 해시 계산
        expected_thumbs = {}
        for file in valid_files:
            v_path = os.path.join(self.folder_path, file)
            f_hash = hashlib.md5(v_path.encode('utf-8')).hexdigest()[:16]
            expected_thumbs[f"{f_hash}.jpg"] = v_path

        # 2. 고아(Orphaned) 썸네일 삭제
        existing_thumbs = [f for f in os.listdir(thumb_dir) if f.endswith('.jpg')]
        for t_file in existing_thumbs:
            if t_file not in expected_thumbs:
                t_path = os.path.join(thumb_dir, t_file)
                try:
                    os.remove(t_path)
                    if db:
                        db.conn.execute('DELETE FROM videos WHERE thumbnail_path = ?', (t_path,))
                        db.conn.commit()
                except Exception as e:
                    print(f"고아 썸네일 삭제 오류 {t_file}: {e}")

        if not valid_files:
            if db: db.close()
            self.finished_signal.emit()
            return

        self.folder_progress_signal.emit(0, len(valid_files))
        completed_count = 0
        
        # 3. 썸네일 동기화 및 신규 생성
        with ThreadPoolExecutor(max_workers=self.max_cores) as executor:
            future_to_file = {}
            for file in valid_files:
                if not self.running: break
                v_path = os.path.join(self.folder_path, file)
                f_hash = hashlib.md5(v_path.encode('utf-8')).hexdigest()[:16]
                t_path = os.path.join(thumb_dir, f"{f_hash}.jpg")

                meta = {"path": v_path, "name": file, "date": 0, "size": 0}
                try:
                    stat = os.stat(v_path)
                    meta["date"], meta["size"] = stat.st_mtime, stat.st_size
                except: pass

                if os.path.exists(t_path):
                    self.progress_signal.emit(v_path, t_path, meta)
                    if db: db.insert_or_update(v_path, meta['name'], meta['size'], meta['date'], t_path)
                    completed_count += 1
                    self.folder_progress_signal.emit(completed_count, len(valid_files))
                else:
                    future = executor.submit(self.process_video, v_path, t_path, meta)
                    future_to_file[future] = (v_path, t_path, meta)

            for future in concurrent.futures.as_completed(future_to_file):
                if not self.running: break
                v_path, t_path, meta, success = future.result()
                if os.path.exists(t_path):
                    self.progress_signal.emit(v_path, t_path, meta)
                    if db: db.insert_or_update(v_path, meta['name'], meta['size'], meta['date'], t_path)
                completed_count += 1
                self.folder_progress_signal.emit(completed_count, len(valid_files))

        if db: db.close()
        self.finished_signal.emit()

    def process_video(self, v_path, t_path, meta):
        try:
            cap = cv2.VideoCapture(v_path, cv2.CAP_ANY)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total <= 0: return v_path, t_path, meta, False
            
            frames = []
            for i in range(1, 10):
                cap.set(cv2.CAP_PROP_POS_FRAMES, int((total / 10) * i))
                ret, frame = cap.read()
                frames.append(cv2.resize(frame, (CELL_SIZE, CELL_SIZE)) if ret else np.zeros((CELL_SIZE, CELL_SIZE, 3), dtype=np.uint8))
            cap.release()
            
            grid = np.vstack([np.hstack(frames[0:3]), np.hstack(frames[3:6]), np.hstack(frames[6:9])])
            safe_imwrite(t_path, grid, [cv2.IMWRITE_JPEG_QUALITY, 85])
            return v_path, t_path, meta, True
        except: return v_path, t_path, meta, False

# ==================== 메인 GUI ====================
class VideoThumbnailManager(QMainWindow):
    def __init__(self):
        super().__init__()
        self.current_folder = ""
        self.worker = None
        
        try: ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID('mystudio.tvm.v3')
        except: pass
        self.setWindowIcon(QIcon(resource_path('tvm.ico')))
        
        # 설정 로드 (최대화 상태 기억 추가)
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, 'r') as f: self.config = json.load(f)
            except:
                self.config = {"width": 1400, "height": 900, "last_folder": "", "maximized": False}
        else: self.config = {"width": 1400, "height": 900, "last_folder": "", "maximized": False}
        
        # 이전 창 크기 및 최대화 상태 복원
        self.resize(self.config.get("width", 1400), self.config.get("height", 900))
        if self.config.get("maximized", False):
            self.showMaximized()

        self.init_ui()
        
        if self.config.get("last_folder", "") and os.path.exists(self.config["last_folder"]):
            QTimer.singleShot(100, lambda: self.scan_folder(self.config["last_folder"]))

    def init_ui(self):
        self.setWindowTitle("TVM Pro - 스마트 동기화 및 자동 크기 조절")
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # 상단 제어 바
        top_bar = QHBoxLayout()
        self.btn_select = QPushButton("📁 폴더 열기")
        self.btn_select.clicked.connect(lambda: self.scan_folder(QFileDialog.getExistingDirectory(self, "폴더 선택")))
        top_bar.addWidget(self.btn_select)
        
        top_bar.addStretch()
        top_bar.addWidget(QLabel("정렬 방식:"))
        self.sort_combo = QComboBox()
        self.sort_combo.addItems(["이름순 (오름차순)", "이름순 (내림차순)", "용량순 (큰 파일)", "용량순 (작은 파일)", "날짜순 (최신)", "날짜순 (과거)"])
        self.sort_combo.currentIndexChanged.connect(self.sort_thumbnails)
        top_bar.addWidget(self.sort_combo)
        layout.addLayout(top_bar)

        # 메인 스플리터 (폴더 트리 | 썸네일 리스트)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # 1. 왼쪽: 윈도우 탐색기 스타일 폴더 트리
        self.dir_model = QFileSystemModel()
        self.dir_model.setRootPath("")
        self.dir_model.setFilter(QDir.Filter.NoDotAndDotDot | QDir.Filter.AllDirs)
        self.tree_view = QTreeView()
        self.tree_view.setModel(self.dir_model)
        for i in range(1, 4): self.tree_view.hideColumn(i) # 크기, 종류, 날짜 숨김
        self.tree_view.clicked.connect(lambda idx: self.scan_folder(self.dir_model.fileInfo(idx).absoluteFilePath()))
        splitter.addWidget(self.tree_view)

        # 2. 오른쪽: 썸네일 리스트
        self.list_widget = QListWidget()
        self.list_widget.setViewMode(QListWidget.ViewMode.IconMode)
        self.list_widget.setIconSize(QSize(280, 280))
        self.list_widget.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.list_widget.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.list_widget.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.list_widget.customContextMenuRequested.connect(self.show_context_menu)
        self.list_widget.setUniformItemSizes(True)
        splitter.addWidget(self.list_widget)
        
        splitter.setSizes([350, 1050])
        
        # [핵심 수정] 빈 공간이 생기지 않도록 스플리터가 세로 공간을 모두 채우게 설정
        layout.addWidget(splitter, 1)

        # 하단 상태 바
        self.pbar = QProgressBar()
        layout.addWidget(self.pbar)
        self.lbl_status = QLabel("준비 완료")
        layout.addWidget(self.lbl_status)

    def scan_folder(self, folder):
        if not folder or not os.path.exists(folder): return
        self.current_folder = folder
        self.list_widget.clear()
        self.lbl_status.setText(f"스캔 중: {folder}")
        self.pbar.setValue(0)
        
        if self.worker: self.worker.stop(); self.worker.wait()
        self.worker = ThumbnailWorker(folder)
        self.worker.progress_signal.connect(self.add_item)
        self.worker.folder_progress_signal.connect(lambda c, t: (self.pbar.setMaximum(t), self.pbar.setValue(c)))
        self.worker.finished_signal.connect(lambda: (self.lbl_status.setText("동기화 및 스캔 완료"), self.sort_thumbnails()))
        self.worker.start()

    def add_item(self, v_path, t_path, meta):
        item = QListWidgetItem(QIcon(QPixmap(t_path)), meta['name'])
        item.setData(Qt.ItemDataRole.UserRole, meta)
        item.setSizeHint(QSize(300, 320))
        self.list_widget.addItem(item)

    def sort_thumbnails(self):
        items_data = []
        for i in range(self.list_widget.count()):
            it = self.list_widget.item(i)
            items_data.append({"icon": it.icon(), "meta": it.data(Qt.ItemDataRole.UserRole)})
        
        idx = self.sort_combo.currentIndex()
        if idx == 0: items_data.sort(key=lambda x: x['meta']['name'].lower())
        elif idx == 1: items_data.sort(key=lambda x: x['meta']['name'].lower(), reverse=True)
        elif idx == 2: items_data.sort(key=lambda x: x['meta']['size'], reverse=True)
        elif idx == 3: items_data.sort(key=lambda x: x['meta']['size'])
        elif idx == 4: items_data.sort(key=lambda x: x['meta']['date'], reverse=True)
        elif idx == 5: items_data.sort(key=lambda x: x['meta']['date'])

        self.list_widget.clear()
        for d in items_data:
            new_it = QListWidgetItem(d['icon'], d['meta']['name'])
            new_it.setData(Qt.ItemDataRole.UserRole, d['meta'])
            new_it.setSizeHint(QSize(300, 320))
            self.list_widget.addItem(new_it)

    def show_context_menu(self, pos):
        sel = self.list_widget.selectedItems()
        if not sel: return
        menu = QMenu()
        
        act_rename = None
        if len(sel) == 1:
            act_rename = menu.addAction("✏️ 이름 변경")
            menu.addSeparator()
            
        act_move = menu.addAction(f"📁 선택 이동 ({len(sel)}개)")
        act_del = menu.addAction(f"🗑️ 선택 삭제 ({len(sel)}개)")
        res = menu.exec(self.list_widget.mapToGlobal(pos))
        
        if res == act_rename:
            self.rename_file(sel[0])
        elif res == act_move:
            dst = QFileDialog.getExistingDirectory(self, "이동 폴더 선택")
            if dst:
                for it in sel:
                    try: shutil.move(it.data(Qt.ItemDataRole.UserRole)['path'], dst); self.list_widget.takeItem(self.list_widget.row(it))
                    except: pass
                self.scan_folder(self.current_folder)
                
        elif res == act_del:
            if QMessageBox.question(self, "삭제", "정말 삭제할까요?") == QMessageBox.StandardButton.Yes:
                for it in sel:
                    try: os.remove(it.data(Qt.ItemDataRole.UserRole)['path']); self.list_widget.takeItem(self.list_widget.row(it))
                    except: pass
                self.scan_folder(self.current_folder)

    def rename_file(self, item):
        old_path = item.data(Qt.ItemDataRole.UserRole)["path"]
        dir_name = os.path.dirname(old_path)
        old_name = os.path.basename(old_path)
        
        new_name, ok = QInputDialog.getText(self, "이름 변경", "새 파일 이름:", text=old_name)
        if ok and new_name and new_name != old_name:
            new_path = os.path.join(dir_name, new_name)
            try:
                os.rename(old_path, new_path)
                item.setText(new_name)
                meta = item.data(Qt.ItemDataRole.UserRole)
                meta["path"] = new_path
                meta["name"] = new_name.lower()
                item.setData(Qt.ItemDataRole.UserRole, meta)
                
                old_hash = hashlib.md5(old_path.encode('utf-8')).hexdigest()[:16]
                new_hash = hashlib.md5(new_path.encode('utf-8')).hexdigest()[:16]
                old_thumb = os.path.join(self.current_folder, '.thumbs', f"{old_hash}.jpg")
                new_thumb = os.path.join(self.current_folder, '.thumbs', f"{new_hash}.jpg")
                if os.path.exists(old_thumb):
                    os.rename(old_thumb, new_thumb)
                    
                QMessageBox.information(self, "성공", "파일 이름이 변경되었습니다.")
            except Exception as e:
                QMessageBox.critical(self, "오류", str(e))

    def closeEvent(self, event):
        self.config["last_folder"] = self.current_folder
        self.config["width"], self.config["height"] = self.width(), self.height()
        # [핵심 수정] 종료 시 창이 최대화 상태였는지 기억합니다.
        self.config["maximized"] = self.isMaximized()
        
        with open(CONFIG_FILE, 'w') as f: json.dump(self.config, f)
        if self.worker: self.worker.stop(); self.worker.wait()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = VideoThumbnailManager()
    window.show()
    sys.exit(app.exec())