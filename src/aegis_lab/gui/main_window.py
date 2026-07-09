import sys
import time
import zmq
import json
import psutil
import os
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QTabWidget, QLabel, QPushButton, QTableWidget, QTableWidgetItem,
    QHeaderView, QProgressBar, QDialog, QTextEdit, QSplitter, QFrame,
    QGridLayout
)
from PyQt6.QtCore import QTimer, Qt, QThread, pyqtSignal
import pyqtgraph as pg

from framewerx.aegis_lab.hardware.thermal import ThermalGuardian
from framewerx.aegis_lab.hardware.discovery import HardwareDiscovery
from framewerx.aegis_lab.hardware.telemetry import LevelZeroTelemetry
from framewerx.aegis_lab.gui.widgets.graph_view import GraphView
from framewerx.aegis_lab.gui.widgets.chat_view import ChatView
from framewerx.aegis_lab.gui.widgets.ablation_map_view import AblationMapView

class RealTimeSubscriber(QThread):
    """
    Subscribes to high-level events (e.g. atom updates) from the orchestrator.
    """
    atom_received = pyqtSignal(dict)
    
    def __init__(self, port=5557):
        super().__init__()
        self.port = port
        self.running = True
        self.context = None
        self.socket = None
        
    def run(self):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        # Multiple subscribers can connect to one PUB bind
        self.socket.connect(f"tcp://localhost:{self.port}")
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "atoms")
        self.socket.setsockopt(zmq.RCVTIMEO, 1000)
        
        while self.running:
            try:
                topic = self.socket.recv_string()
                data = self.socket.recv_json()
                if topic == "atoms":
                    self.atom_received.emit(data)
            except zmq.Again:
                continue
            except Exception as e:
                # Use print or a logger if available
                print(f"Subscriber error: {e}")
                
    def stop(self):
        self.running = False
        if self.socket is not None:
            self.socket.close()
            self.socket = None
        if self.context is not None:
            self.context.destroy(linger=0)
            self.context = None

DARK_STYLESHEET = """
QMainWindow, QWidget {
    background-color: #09111f;
    color: #d9e2f2;
    font-family: "DejaVu Sans";
}
QTabWidget::pane {
    border: 1px solid #22324a;
    top: -1px;
    background: #0b1525;
}
QTabBar::tab {
    background-color: #101c31;
    color: #89a0bf;
    padding: 10px 16px;
    border: 1px solid #22324a;
    border-bottom: none;
    margin-right: 4px;
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
}
QTabBar::tab:selected {
    background-color: #0b1525;
    color: #7dd3fc;
    border-bottom: 2px solid #7dd3fc;
}
QPushButton {
    background-color: #12304d;
    border: 1px solid #29547a;
    border-radius: 8px;
    padding: 7px 15px;
    color: #ccecff;
    font-weight: bold;
}
QPushButton:hover {
    background-color: #174166;
    border-color: #4d8ec1;
}
QTableWidget {
    background-color: #0d1728;
    alternate-background-color: #101d31;
    gridline-color: #22324a;
    border: 1px solid #22324a;
    border-radius: 10px;
}
QHeaderView::section {
    background-color: #101c31;
    color: #94a8c6;
    padding: 6px;
    border: 1px solid #22324a;
}
QProgressBar {
    border: 1px solid #22324a;
    border-radius: 4px;
    text-align: center;
    background-color: #101c31;
    color: #d9e2f2;
}
QProgressBar::chunk {
    background-color: #0ea5e9;
}
QTextEdit {
    background-color: #0d1728;
    border: 1px solid #22324a;
    color: #d9e2f2;
    border-radius: 10px;
}
QLineEdit {
    background-color: #0d1728;
    border: 1px solid #22324a;
    color: #d9e2f2;
    border-radius: 8px;
    padding: 6px 10px;
}
QScrollBar:vertical {
    background: #09111f;
    width: 12px;
    margin: 0px;
}
QScrollBar::handle:vertical {
    background: #24405f;
    min-height: 30px;
    border-radius: 6px;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}
"""

class OrchestratorClient:
    def __init__(self, url="tcp://localhost:5555"):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.connect(url)
        self.socket.setsockopt(zmq.RCVTIMEO, 2000)

    def request(self, msg_type, **kwargs):
        try:
            kwargs["type"] = msg_type
            self.socket.send_json(kwargs)
            return self.socket.recv_json()
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def close(self):
        if self.socket is not None:
            self.socket.close()
            self.socket = None
        if self.context is not None:
            self.context.destroy(linger=0)
            self.context = None

class DashboardTab(QWidget):
    def __init__(self, client):
        super().__init__()
        self.client = client
        self.layout = QVBoxLayout(self)
        
        self.splitter = QSplitter(Qt.Orientation.Vertical)
        self.layout.addWidget(self.splitter)
        
        # Jobs Table
        self.job_widget = QWidget()
        self.job_layout = QVBoxLayout(self.job_widget)
        self.job_label = QLabel("Active Jobs")
        self.job_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #58a6ff;")
        self.job_layout.addWidget(self.job_label)
        
        self.job_table = QTableWidget(0, 4)
        self.job_table.setHorizontalHeaderLabels(["Job ID", "Type", "Status", "Progress"])
        self.job_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.job_table.setAlternatingRowColors(True)
        self.job_table.itemSelectionChanged.connect(self.job_selected)
        self.job_layout.addWidget(self.job_table)
        self.splitter.addWidget(self.job_widget)
        
        # Stages Table (DAG View approximation)
        self.stage_widget = QWidget()
        self.stage_layout = QVBoxLayout(self.stage_widget)
        self.stage_label = QLabel("Job Stages")
        self.stage_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #58a6ff;")
        self.stage_layout.addWidget(self.stage_label)
        
        # New GraphView
        self.graph_view = GraphView()
        self.stage_layout.addWidget(self.graph_view)
        
        self.stage_table = QTableWidget(0, 4)
        self.stage_table.setHorizontalHeaderLabels(["Stage ID", "Name", "Status", "Action"])
        self.stage_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.stage_table.setAlternatingRowColors(True)
        self.stage_layout.addWidget(self.stage_table)
        
        # Add 3D Ablation Map
        self.map_widget = QWidget()
        self.map_layout = QVBoxLayout(self.map_widget)
        self.map_label = QLabel("3D Ablation Map (Atoms)")
        self.map_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #a5d6ff;")
        self.map_layout.addWidget(self.map_label)
        self.ablation_map = AblationMapView()
        self.map_layout.addWidget(self.ablation_map)

        # Horizontal splitter for Stages and Map
        self.lower_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.lower_splitter.addWidget(self.stage_widget)
        self.lower_splitter.addWidget(self.map_widget)
        self.splitter.addWidget(self.lower_splitter)
        
        self.refresh_btn = QPushButton("Refresh All")
        self.refresh_btn.clicked.connect(self.refresh_jobs)
        self.layout.addWidget(self.refresh_btn)
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.refresh_jobs)
        self.timer.start(5000)
        self.selected_job_id = None

    def refresh_jobs(self):
        resp = self.client.request("list_jobs")
        if "error" in resp: return
            
        jobs = resp if isinstance(resp, list) else []
        self.job_table.setRowCount(len(jobs))
        for i, job in enumerate(jobs):
            self.job_table.setItem(i, 0, QTableWidgetItem(job["job_id"]))
            self.job_table.setItem(i, 1, QTableWidgetItem(job["job_type"]))
            self.job_table.setItem(i, 2, QTableWidgetItem(job["status"]))
            prog = QProgressBar()
            prog.setValue(100 if job["status"] == "succeeded" else 50 if job["status"] == "running" else 0)
            self.job_table.setCellWidget(i, 3, prog)
            
        if self.selected_job_id:
            self.refresh_stages()

    def job_selected(self):
        row = self.job_table.currentRow()
        if row >= 0:
            self.selected_job_id = self.job_table.item(row, 0).text()
            self.refresh_stages()

    def refresh_stages(self):
        if not self.selected_job_id: return
        resp = self.client.request("get_job_status", job_id=self.selected_job_id)
        if "error" in resp: return
        
        stages = resp.get("stages", [])
        self.graph_view.update_from_stages(stages)
        self.stage_table.setRowCount(len(stages))
        for i, stage in enumerate(sorted(stages, key=lambda x: x["ordinal"])):
            self.stage_table.setItem(i, 0, QTableWidgetItem(stage["stage_id"]))
            self.stage_table.setItem(i, 1, QTableWidgetItem(stage["stage_name"]))
            self.stage_table.setItem(i, 2, QTableWidgetItem(stage["status"]))
            
            if stage["status"] == "pending" and "gate" in stage["stage_name"]:
                btn = QPushButton("Approve")
                btn.clicked.connect(lambda _, s=stage["stage_id"]: self.approve_stage(s))
                self.stage_table.setCellWidget(i, 3, btn)
            else:
                self.stage_table.setItem(i, 3, QTableWidgetItem("N/A"))

    def approve_stage(self, stage_id):
        self.client.request("approve_stage", stage_id=stage_id)
        self.refresh_stages()

class HardwareTab(QWidget):
    def __init__(self, client: OrchestratorClient):
        super().__init__()
        self.client = client
        self.layout = QVBoxLayout(self)
        self.thermal_guardian = ThermalGuardian()
        self.telemetry = LevelZeroTelemetry()
        
        # Header with Tune Button
        self.header_layout = QHBoxLayout()
        self.thermal_label = QLabel("Thermal Status: Unknown")
        self.thermal_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        self.header_layout.addWidget(self.thermal_label)
        
        self.header_layout.addStretch()
        
        self.tune_btn = QPushButton("NPU Cache Tune")
        self.tune_btn.clicked.connect(self.trigger_uma_tune)
        self.header_layout.addWidget(self.tune_btn)
        
        # Add VPU Status Fetch Button
        self.fetch_vpu_btn = QPushButton("Fetch VPU Status")
        self.fetch_vpu_btn.clicked.connect(self.fetch_vpu_status)
        self.header_layout.addWidget(self.fetch_vpu_btn)

        self.layout.addLayout(self.header_layout)
        
        # Telemetry Status
        self.telemetry_label = QLabel("GPU/NPU Telemetry: Initializing...")
        self.telemetry_label.setStyleSheet("font-size: 14px; color: #8b949e;")
        self.layout.addWidget(self.telemetry_label)
        
        # VPU Telemetry Placeholders
        self.vpu_device_label = QLabel("VPU Device: N/A")
        self.vpu_device_label.setStyleSheet("font-size: 13px; color: #8b949e;")
        self.layout.addWidget(self.vpu_device_label)

        self.vpu_memory_label = QLabel("VPU Peak Memory: N/A")
        self.vpu_memory_label.setStyleSheet("font-size: 13px; color: #8b949e;")
        self.layout.addWidget(self.vpu_memory_label)
        
        # Charts
        pg.setConfigOptions(antialias=True)
        self.win = pg.GraphicsLayoutWidget()
        self.win.setBackground('#0d1117')
        self.layout.addWidget(self.win)
        
        # CPU Plot
        self.cpu_plot = self.win.addPlot(title="CPU Usage (%)")
        self.cpu_plot.showGrid(x=True, y=True, alpha=0.3)
        self.cpu_plot.addLegend()
        self.cpu_curve = self.cpu_plot.plot(pen='#58a6ff', name='CPU')
        self.cpu_data = []
        
        # iGPU Plot
        self.igpu_plot = self.win.addPlot(title="iGPU Usage (%)")
        self.igpu_plot.showGrid(x=True, y=True, alpha=0.3)
        self.igpu_plot.addLegend()
        self.igpu_curve = self.igpu_plot.plot(pen='#3fb950', name='iGPU') # Greenish
        self.igpu_data = []
        
        self.win.nextRow()
        
        # NPU Plot
        self.npu_plot = self.win.addPlot(title="NPU Usage (%)")
        self.npu_plot.showGrid(x=True, y=True, alpha=0.3)
        self.npu_plot.addLegend()
        self.npu_curve = self.npu_plot.plot(pen='#a5d6ff', name='NPU') # Light blue
        self.npu_data = []
        
        # Temp Plot
        self.temp_plot = self.win.addPlot(title="Temperature (°C)")
        self.temp_plot.showGrid(x=True, y=True, alpha=0.3)
        self.temp_plot.addLegend()
        self.temp_curve = self.temp_plot.plot(pen='#f85149', name='Temp') # Reddish
        self.temp_data = []
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_stats)
        self.timer.start(1000)

    def trigger_uma_tune(self):
        resp = self.client.request("optimize_uma")
        if resp.get("status") == "ok":
            self.telemetry_label.setText("SITREP: UMA Priority Pinning Refreshed successfully.")
            self.telemetry_label.setStyleSheet("font-size: 14px; color: #58a6ff;")
        else:
            self.telemetry_label.setText(f"SITREP: UMA Tune Failed: {resp.get('error', 'Unknown error')}")
            self.telemetry_label.setStyleSheet("font-size: 14px; color: #f85149;")

    def fetch_vpu_status(self):
        """
        Fetches VPU status from the orchestrator and updates the UI labels.
        This assumes an RPC endpoint 'get_worker_status' exists on the orchestrator
        that can return VPU-specific telemetry.
        """
        resp = self.client.request("get_worker_status", worker_type="vpu")
        
        if resp.get("status") == "ok" and "data" in resp:
            vpu_data = resp["data"]
            device = vpu_data.get("device", "N/A")
            peak_memory = vpu_data.get("peak_memory_mb", "N/A")
            
            self.vpu_device_label.setText(f"VPU Device: {device}")
            self.vpu_memory_label.setText(f"VPU Peak Memory: {peak_memory} MB")
            self.vpu_device_label.setStyleSheet("font-size: 13px; color: #58a6ff;")
            self.vpu_memory_label.setStyleSheet("font-size: 13px; color: #58a6ff;")
        else:
            # Handle error case or no VPU found
            error_msg = resp.get("error", "No VPU worker status found or orchestrator error.")
            self.vpu_device_label.setText("VPU Device: N/A (Error)")
            self.vpu_memory_label.setText(f"VPU Peak Memory: N/A ({error_msg})")
            self.vpu_device_label.setStyleSheet("font-size: 13px; color: #f85149;")
            self.vpu_memory_label.setStyleSheet("font-size: 13px; color: #f85149;")

    def update_stats(self):
        # CPU Usage
        cpu = psutil.cpu_percent()
        self.cpu_data.append(cpu)
        if len(self.cpu_data) > 60: self.cpu_data.pop(0)
        self.cpu_curve.setData(self.cpu_data)
        
        # Level Zero Telemetry (iGPU/NPU)
        metrics = self.telemetry.get_metrics()
        igpu_util = 0.0
        npu_util = 0.0
        details = []

        if len(metrics) > 0:
            # Assuming first device is iGPU, second is NPU (common on MTL)
            igpu_util = metrics[0].get("utilization", 0.0)
            details.append(f"GPU: {metrics[0]['freq_mhz']:.0f}MHz, {metrics[0]['power_w']:.1f}W")
            
            if len(metrics) > 1:
                npu_util = metrics[1].get("utilization", 0.0)
                details.append(f"NPU: {metrics[1]['freq_mhz']:.0f}MHz, {metrics[1]['power_w']:.1f}W")
            
            self.telemetry_label.setText(" | ".join(details))
            self.telemetry_label.setStyleSheet("font-size: 14px; color: #58a6ff;")
        else:
            # Enhanced Fallback Hints
            hint = "Level Zero Sysman not detected. Check /dev/dri permissions or install intel-level-zero-gpu."
            self.telemetry_label.setText(f"GPU/NPU Telemetry: {hint}")
            self.telemetry_label.setStyleSheet("font-size: 14px; color: #8b949e;")
            
            # Still show some random data for visual feedback in dev
            import random
            igpu_util = random.uniform(5, 15) if cpu > 20 else random.uniform(0, 5)
            npu_util = random.uniform(10, 30) if "running" in self.thermal_label.text().lower() else random.uniform(0, 2)

        self.igpu_data.append(igpu_util)
        if len(self.igpu_data) > 60: self.igpu_data.pop(0)
        self.igpu_curve.setData(self.igpu_data)
        
        self.npu_data.append(npu_util)
        if len(self.npu_data) > 60: self.npu_data.pop(0)
        self.npu_curve.setData(self.npu_data)
        
        # Thermal
        status = self.thermal_guardian.get_status()
        temp = status["temperature"]
        self.temp_data.append(temp)
        if len(self.temp_data) > 60: self.temp_data.pop(0)
        self.temp_curve.setData(self.temp_data)
        
        self.thermal_label.setText(f"Thermal Status: {status['level']} ({temp:.1f}°C)")
        if status['level'] == 'CRITICAL':
            self.thermal_label.setStyleSheet("font-size: 16px; color: #f85149; font-weight: bold;")
        elif status['level'] == 'HIGH':
            self.thermal_label.setStyleSheet("font-size: 16px; color: #d29922;")
        else:
            self.thermal_label.setStyleSheet("font-size: 16px; color: #3fb950;")

        # --- VPU Telemetry Update ---
        # Attempt to fetch VPU status periodically.
        self.fetch_vpu_status()

class ArtifactTab(QWidget):
    def __init__(self, storage_root):
        super().__init__()
        self.storage_root = storage_root
        self.layout = QVBoxLayout(self)
        self.label = QLabel(f"Artifact Store: {storage_root}")
        self.label.setStyleSheet("color: #8b949e;")
        self.layout.addWidget(self.label)
        
        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Hash", "Path"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setAlternatingRowColors(True)
        self.layout.addWidget(self.table)
        
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self.refresh_artifacts)
        self.layout.addWidget(self.refresh_btn)

    def refresh_artifacts(self):
        # This is a simple explorer of the content-addressed store
        import os
        artifacts = []
        if os.path.exists(self.storage_root):
            for root, dirs, files in os.walk(self.storage_root):
                for file in files:
                    if len(file) == 64: # SHA256 length
                        artifacts.append((file, os.path.join(root, file)))
        
        self.table.setRowCount(len(artifacts))
        for i, (h, p) in enumerate(artifacts):
            self.table.setItem(i, 0, QTableWidgetItem(h))
            self.table.setItem(i, 1, QTableWidgetItem(p))

class DiffViewTab(QWidget):
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout(self)
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        
        self.before_text = QTextEdit()
        self.before_text.setPlaceholderText("Before Ablation")
        self.before_text.setReadOnly(True)
        
        self.after_text = QTextEdit()
        self.after_text.setPlaceholderText("After Ablation")
        self.after_text.setReadOnly(True)
        
        self.splitter.addWidget(self.before_text)
        self.splitter.addWidget(self.after_text)
        self.layout.addWidget(self.splitter)
        
        self.load_btn = QPushButton("Load Comparison")
        self.layout.addWidget(self.load_btn)


class SystemCard(QFrame):
    def __init__(self, title: str, summary: str, action_label: str = None, action=None):
        super().__init__()
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setStyleSheet("""
            QFrame {
                background-color: #0f1b2f;
                border: 1px solid #22324a;
                border-radius: 14px;
            }
            QLabel#title {
                color: #eef6ff;
                font-size: 16px;
                font-weight: bold;
            }
            QLabel#summary {
                color: #9db2cf;
                font-size: 12px;
            }
            QLabel#status {
                color: #7dd3fc;
                font-size: 11px;
                font-weight: bold;
                text-transform: uppercase;
                letter-spacing: 0.08em;
            }
        """)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        self.title_label = QLabel(title)
        self.title_label.setObjectName("title")
        layout.addWidget(self.title_label)

        self.summary_label = QLabel(summary)
        self.summary_label.setObjectName("summary")
        self.summary_label.setWordWrap(True)
        layout.addWidget(self.summary_label)

        self.status_label = QLabel("")
        self.status_label.setObjectName("status")
        layout.addWidget(self.status_label)

        layout.addStretch()

        self.action_btn = None
        if action_label:
            self.action_btn = QPushButton(action_label)
            if action:
                self.action_btn.clicked.connect(action)
            layout.addWidget(self.action_btn)

    def update_content(self, summary: str, status: str, accent: str):
        self.summary_label.setText(summary)
        self.status_label.setText(status)
        self.status_label.setStyleSheet(
            f"color: {accent}; font-size: 11px; font-weight: bold; text-transform: uppercase; letter-spacing: 0.08em;"
        )


class SystemsTab(QWidget):
    def __init__(self, main_window: "MainWindow"):
        super().__init__()
        self.main_window = main_window
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(18, 18, 18, 18)
        self.layout.setSpacing(14)

        header = QFrame()
        header.setStyleSheet("QFrame { background-color: #0f1b2f; border: 1px solid #22324a; border-radius: 16px; }")
        header_layout = QVBoxLayout(header)
        header_layout.setContentsMargins(20, 20, 20, 20)

        title = QLabel("Systems Command Deck")
        title.setStyleSheet("font-size: 26px; font-weight: bold; color: #eef6ff;")
        header_layout.addWidget(title)

        subtitle = QLabel("Linked access to orchestration, hardware telemetry, artifacts, rankings, interrogation, and diff inspection.")
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet("font-size: 13px; color: #9db2cf;")
        header_layout.addWidget(subtitle)

        self.snapshot_label = QLabel("Refreshing subsystem snapshot...")
        self.snapshot_label.setStyleSheet("font-size: 12px; color: #7dd3fc; font-weight: bold;")
        header_layout.addWidget(self.snapshot_label)

        header_buttons = QHBoxLayout()
        refresh_btn = QPushButton("Refresh Systems")
        refresh_btn.clicked.connect(self.refresh_cards)
        header_buttons.addWidget(refresh_btn)

        jobs_btn = QPushButton("Open Dashboard")
        jobs_btn.clicked.connect(lambda: self.main_window.switch_to_tab(self.main_window.dashboard))
        header_buttons.addWidget(jobs_btn)
        header_buttons.addStretch()
        header_layout.addLayout(header_buttons)
        self.layout.addWidget(header)

        self.grid = QGridLayout()
        self.grid.setHorizontalSpacing(14)
        self.grid.setVerticalSpacing(14)
        self.layout.addLayout(self.grid)

        self.cards = {}
        descriptors = [
            ("orchestrator", "Orchestrator", "Control plane for jobs, workers, and stage approvals.", "Open Dashboard",
             lambda: self.main_window.switch_to_tab(self.main_window.dashboard)),
            ("hardware", "Hardware Fabric", "CPU, iGPU, NPU, thermal state, and acceleration readiness.", "Open Hardware",
             lambda: self.main_window.switch_to_tab(self.main_window.hardware)),
            ("artifacts", "Artifact Store", "Content-addressed artifacts and promotion outputs.", "Open Artifacts",
             lambda: self.main_window.switch_to_tab(self.main_window.artifacts)),
            ("leaderboard", "Evaluation", "Leaderboard, scoring, robustness, and efficiency summaries.", "Open Leaderboard",
             lambda: self.main_window.switch_to_tab(self.main_window.leaderboard)),
            ("chat", "RAG Interrogation", "Operator chat, atom retrieval, and behavioral context review.", "Open Interrogation",
             lambda: self.main_window.switch_to_tab(self.main_window.chat)),
            ("diff", "Diff Review", "Before/after inspection surface for ablation comparisons.", "Open Diff View",
             lambda: self.main_window.switch_to_tab(self.main_window.diff_view)),
        ]

        for index, (key, title_text, summary, label, action) in enumerate(descriptors):
            card = SystemCard(title_text, summary, label, action)
            self.cards[key] = card
            self.grid.addWidget(card, index // 2, index % 2)

        self.layout.addStretch()
        self.refresh_cards()

    def refresh_cards(self):
        snapshot = self.main_window.collect_system_snapshot()
        self.snapshot_label.setText(snapshot["headline"])

        for key, payload in snapshot["systems"].items():
            card = self.cards.get(key)
            if card:
                card.update_content(payload["summary"], payload["status"], payload["accent"])

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AEGIS-LAB Dashboard")
        self.resize(1280, 850)
        self.setStyleSheet(DARK_STYLESHEET)

        self.client = OrchestratorClient()
        self.telemetry = LevelZeroTelemetry()
        self.thermal_guardian = ThermalGuardian()

        # Main Layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)

        # SITREP Top-Bar
        self.sitrep_bar = QFrame()
        self.sitrep_bar.setFrameShape(QFrame.Shape.StyledPanel)
        self.sitrep_bar.setStyleSheet("background-color: #0f1b2f; border-bottom: 1px solid #22324a; min-height: 48px; border-radius: 14px;")
        self.sitrep_layout = QHBoxLayout(self.sitrep_bar)
        self.sitrep_layout.setContentsMargins(15, 5, 15, 5)

        self.jobs_sitrep = QLabel("Active Jobs: 0")
        self.jobs_sitrep.setStyleSheet("color: #58a6ff; font-weight: bold;")
        self.sitrep_layout.addWidget(self.jobs_sitrep)

        self.sitrep_layout.addSpacing(20)

        self.thermal_sitrep = QLabel("Thermal: NOMINAL")
        self.thermal_sitrep.setStyleSheet("color: #3fb950; font-weight: bold;")
        self.sitrep_layout.addWidget(self.thermal_sitrep)

        self.sitrep_layout.addSpacing(20)

        self.npu_sitrep = QLabel("Acceleration: CHECKING")
        self.npu_sitrep.setStyleSheet("color: #58a6ff; font-weight: bold;")
        self.sitrep_layout.addWidget(self.npu_sitrep)

        self.sitrep_layout.addStretch()
        
        self.status_label = QLabel("SYSTEM READY")
        self.status_label.setStyleSheet("color: #8b949e; font-size: 10px; font-family: monospace;")
        self.sitrep_layout.addWidget(self.status_label)

        self.fullscreen_btn = QPushButton("⛶")
        self.fullscreen_btn.setToolTip("Toggle Fullscreen (F11)")
        self.fullscreen_btn.setFixedWidth(30)
        self.fullscreen_btn.setStyleSheet("background-color: transparent; color: #8b949e; border: none; font-size: 16px;")
        self.fullscreen_btn.clicked.connect(self.toggle_fullscreen)
        self.sitrep_layout.addWidget(self.fullscreen_btn)

        self.main_layout.addWidget(self.sitrep_bar)

        # Tabs
        self.tabs = QTabWidget()
        self.main_layout.addWidget(self.tabs)

        from framewerx.aegis_lab.gui.widgets.graph_view import GraphView
        from framewerx.aegis_lab.gui.widgets.chat_view import ChatView
        from framewerx.aegis_lab.gui.widgets.leaderboard_view import LeaderboardTab

        self.systems = SystemsTab(self)
        self.dashboard = DashboardTab(self.client)
        self.hardware = HardwareTab(self.client)
        self.artifacts = ArtifactTab(os.path.expanduser("~/.aegis_lab/artifacts"))
        self.leaderboard = LeaderboardTab(self.client)
        self.chat = ChatView()
        self.diff_view = DiffViewTab()

        self.tabs.addTab(self.systems, "Systems")
        self.tabs.addTab(self.dashboard, "Dashboard")
        self.tabs.addTab(self.hardware, "Hardware")
        self.tabs.addTab(self.artifacts, "Artifacts")
        self.tabs.addTab(self.leaderboard, "Leaderboard")
        self.tabs.addTab(self.chat, "Interrogation")
        self.tabs.addTab(self.diff_view, "Diff View")

        self.chat.send_message.connect(self.handle_chat)

        # SITREP Update Timer
        self.sitrep_timer = QTimer()
        self.sitrep_timer.timeout.connect(self.update_sitrep)
        self.sitrep_timer.start(2000)

    def toggle_fullscreen(self):
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    def switch_to_tab(self, widget):
        index = self.tabs.indexOf(widget)
        if index >= 0:
            self.tabs.setCurrentIndex(index)

    def collect_system_snapshot(self):
        discovery = HardwareDiscovery.discover()
        jobs_response = self.client.request("list_jobs")
        jobs = jobs_response if isinstance(jobs_response, list) else []
        running_jobs = sum(1 for job in jobs if job.get("status") == "running")
        total_jobs = len(jobs)
        telemetry_metrics = self.telemetry.get_metrics()
        thermal = self.thermal_guardian.get_status()
        artifact_root = os.path.expanduser("~/.aegis_lab/artifacts")
        artifact_count = 0

        if os.path.exists(artifact_root):
            for root, _, files in os.walk(artifact_root):
                artifact_count += sum(1 for file in files if len(file) == 64)

        orchestrator_ok = not isinstance(jobs_response, dict) or "error" not in jobs_response
        hardware_parts = []
        if discovery["cpu_avx512"]:
            hardware_parts.append("AVX-512")
        if discovery["cpu_amx"]:
            hardware_parts.append("AMX")
        if discovery["igpu_present"]:
            hardware_parts.append(f"iGPU {discovery['igpu_type']}")
        if discovery["npu_present"]:
            hardware_parts.append(f"NPU {discovery['npu_type']}")
        if discovery["cuda_compat"]:
            hardware_parts.append("CUDA-compat bridge")
        if not hardware_parts:
            hardware_parts.append("CPU-only fallback")

        telemetry_summary = f"{len(telemetry_metrics)} telemetry device(s) online" if telemetry_metrics else "Telemetry offline"
        headline = (
            f"{running_jobs} running job(s), thermal {thermal['level'].lower()}, "
            f"{telemetry_summary}, {len(hardware_parts)} linked acceleration path(s)."
        )

        return {
            "headline": headline,
            "systems": {
                "orchestrator": {
                    "summary": (
                        f"{total_jobs} tracked job(s), {running_jobs} actively running. "
                        f"{'IPC responding.' if orchestrator_ok else 'IPC unavailable.'}"
                    ),
                    "status": "Online" if orchestrator_ok else "Offline",
                    "accent": "#34d399" if orchestrator_ok else "#f87171",
                },
                "hardware": {
                    "summary": f"{', '.join(hardware_parts)}. Thermal {thermal['level']} at {thermal['temperature']:.1f}C.",
                    "status": "Accelerated" if discovery["accel_available"] else "Fallback",
                    "accent": "#7dd3fc" if discovery["accel_available"] else "#fbbf24",
                },
                "artifacts": {
                    "summary": f"{artifact_count} stored artifact blob(s) under {artifact_root}.",
                    "status": "Mounted" if os.path.exists(artifact_root) else "Missing",
                    "accent": "#c084fc" if os.path.exists(artifact_root) else "#f87171",
                },
                "leaderboard": {
                    "summary": "Ranking view wired to orchestrator leaderboard RPC for integrity and efficiency metrics.",
                    "status": "Ready",
                    "accent": "#fbbf24",
                },
                "chat": {
                    "summary": "Interrogation panel linked to operator chat history and atom context sidebar.",
                    "status": "Interactive",
                    "accent": "#38bdf8",
                },
                "diff": {
                    "summary": "Side-by-side before/after review surface available for ablation comparisons.",
                    "status": "Review",
                    "accent": "#fb7185",
                },
            },
        }

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_F11:
            self.toggle_fullscreen()
        else:
            super().keyPressEvent(event)

    def update_sitrep(self):
        # Active Jobs
        resp = self.client.request("list_jobs")
        if isinstance(resp, list):
            active_count = sum(1 for j in resp if j["status"] == "running")
            self.jobs_sitrep.setText(f"Active Jobs: {active_count}")
        else:
            self.jobs_sitrep.setText("Active Jobs: unavailable")
        
        # Thermal
        status = self.thermal_guardian.get_status()
        self.thermal_sitrep.setText(f"Thermal: {status['level']}")
        if status['level'] == 'NOMINAL':
            self.thermal_sitrep.setStyleSheet("color: #3fb950; font-weight: bold;")
        elif status['level'] in ['HIGH', 'CRITICAL']:
            self.thermal_sitrep.setStyleSheet("color: #f85149; font-weight: bold;")
        else:
            self.thermal_sitrep.setStyleSheet("color: #d29922; font-weight: bold;")

        # Acceleration Availability
        discovery = HardwareDiscovery.discover()
        metrics = self.telemetry.get_metrics()
        if discovery["npu_present"] and len(metrics) > 1:
            self.npu_sitrep.setText("Acceleration: NPU ACTIVE")
            self.npu_sitrep.setStyleSheet("color: #7dd3fc; font-weight: bold;")
        elif discovery["igpu_present"] or discovery["cuda_compat"]:
            self.npu_sitrep.setText("Acceleration: GPU LINKED")
            self.npu_sitrep.setStyleSheet("color: #7dd3fc; font-weight: bold;")
        elif discovery["accel_available"]:
            self.npu_sitrep.setText("Acceleration: PARTIAL")
            self.npu_sitrep.setStyleSheet("color: #fbbf24; font-weight: bold;")
        else:
            self.npu_sitrep.setText("Acceleration: CPU FALLBACK")
            self.npu_sitrep.setStyleSheet("color: #94a3b8; font-weight: bold;")

        self.status_label.setText(time.strftime("LAST REFRESH %H:%M:%S"))
        self.systems.refresh_cards()

    def handle_chat(self, text):
        # In a real impl, this would call the RAG engine via IPC
        self.chat.update_context([
            {"atom_id": "coding_core", "score": 0.92},
            {"atom_id": "refusal_circuit", "score": 0.15}
        ])
        self.chat.add_response(f"Simulated ablation-aware response to: {text}")


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
