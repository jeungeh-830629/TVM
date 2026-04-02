import os
import sys
import shutil
import subprocess
from pathlib import Path

def build():
    # 이전 빌드 청소
    for d in ['build', 'dist']:
        if os.path.exists(d): shutil.rmtree(d)
    
    options = [
        '--name=TVM',
        '--windowed',
        '--onedir',           # 폴더형 빌드 (백신 오탐 방지 핵심)
        '--collect-all=cv2',  # OpenCV 라이브러리 전체 수집
        '--hidden-import=PyQt6.QtCore',
        '--hidden-import=PyQt6.QtGui',
        '--hidden-import=PyQt6.QtWidgets',
        'tvm.py'
    ]
    
    if os.path.exists('tvm.ico'):
        options.append('--icon=tvm.ico')
        options.append('--add-data=tvm.ico;.')

    print("🚀 빌드 시작...")
    subprocess.run([sys.executable, '-m', 'PyInstaller'] + options)
    print("\n✅ 빌드 완료! 'dist/TVM_Pro' 폴더를 확인하세요.")

if __name__ == '__main__':
    build()