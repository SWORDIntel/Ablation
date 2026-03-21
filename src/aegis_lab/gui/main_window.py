import sys
import time
import zmq
import json
import psutil
import os
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QTabWidget, QLabel, QPushButton, QTableWidget, QTableWidgetItem,
    QHeaderView, QProgressBar, QDialog, QTextEdit, QSplitter, QFrame
)
from PyQt6.QtCore import QTimer, Qt, QThread, pyqtSignal
import pyqtgraph as pg

from aegis_lab.hardware.thermal import ThermalGuardian
from aegis_lab.hardware.discovery import HardwareDiscovery
from aegis_lab.hardware.telemetry import LevelZeroTelemetry
from aegis_lab.gui.widgets.graph_view import GraphView
from aegis_lab.gui.widgets.chat_view import ChatView

DARK_STYLESHEET = """
QMainWindow, QWidget {
    background-color: #0d1117;
    color: #c9d1d9;
}
QTabWidget::pane {
    border: 1px solid #30363d;
}
QTabBar::tab {
    background-color: #161b22;
    padding: 8px 12px;
    border: 1px solid #30363d;
    margin-right: 2px;
}
QTabBar::tab:selected {
    background-color: #0d1117;
    border-bottom: 2px solid #58a6ff;
    color: #58a6ff;
}
QPushButton {
    background-color: #21262d;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 5px 15px;
    color: #58a6ff;
    font-weight: bold;
}
QPushButton:hover {
    background-color: #30363d;
}
QTableWidget {
    gridline-color: #30363d;
    border: 1px solid #30363d;
}
QHeaderView::section {
    background-color: #161b22;
    color: #8b949e;
    padding: 4px;
    border: 1px solid #30363d;
}
QProgressBar {
    border: 1px solid #30363d;
    border-radius: 4px;
    text-align: center;
    background-color: #161b22;
}
QProgressBar::chunk {
    background-color: #58a6ff;
}
QTextEdit {
    background-color: #0d1117;
    border: 1px solid #30363d;
    color: #c9d1d9;
}
"""

class OrchestratorClient:
    def __init__(self, url="tcp://localhost:5555"):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(url)
        self.socket.setsockopt(zmq.RCVTIMEO, 2000)

    def request(self, msg_type, **kwargs):
        try:
            kwargs["type"] = msg_type
            self.socket.send_json(kwargs)
            return self.socket.recv_json()
        except Exception as e:
            return {"status": "error", "error": str(e)}

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
        self.stage_layout.addWidget(self.stage_table)
        self.splitter.addWidget(self.stage_widget)
        
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
        
        self.layout.addLayout(self.header_layout)
        
        # Telemetry Status
        self.telemetry_label = QLabel("GPU/NPU Telemetry: Initializing...")
        self.telemetry_label.setStyleSheet("font-size: 14px; color: #8b949e;")
        self.layout.addWidget(self.telemetry_label)
        
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
        self.sitrep_bar.setStyleSheet("background-color: #161b22; border-bottom: 1px solid #30363d; min-height: 40px;")
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

        self.npu_sitrep = QLabel("NPU: DISCOVERED")
        self.npu_sitrep.setStyleSheet("color: #58a6ff; font-weight: bold;")
        self.sitrep_layout.addWidget(self.npu_sitrep)

        self.sitrep_layout.addStretch()
        
        self.status_label = QLabel("SYSTEM READY")
        self.status_label.setStyleSheet("color: #8b949e; font-size: 10px; font-family: monospace;")
        self.sitrep_layout.addWidget(self.status_label)

        self.main_layout.addWidget(self.sitrep_bar)

        # Tabs
        self.tabs = QTabWidget()
        self.main_layout.addWidget(self.tabs)

        from aegis_lab.gui.widgets.graph_view import GraphView
        from aegis_lab.gui.widgets.chat_view import ChatView
        from aegis_lab.gui.widgets.leaderboard_view import LeaderboardTab

        self.dashboard = DashboardTab(self.client)
        self.hardware = HardwareTab(self.client)
        self.artifacts = ArtifactTab(os.path.expanduser("~/.aegis_lab/artifacts"))
        self.leaderboard = LeaderboardTab(self.client)
        self.chat = ChatView()
        self.diff_view = DiffViewTab()

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

    def update_sitrep(self):
        # Active Jobs
        resp = self.client.request("list_jobs")
        if isinstance(resp, list):
            active_count = sum(1 for j in resp if j["status"] == "running")
            self.jobs_sitrep.setText(f"Active Jobs: {active_count}")
        
        # Thermal
        status = self.thermal_guardian.get_status()
        self.thermal_sitrep.setText(f"Thermal: {status['level']}")
        if status['level'] == 'NOMINAL':
            self.thermal_sitrep.setStyleSheet("color: #3fb950; font-weight: bold;")
        elif status['level'] in ['HIGH', 'CRITICAL']:
            self.thermal_sitrep.setStyleSheet("color: #f85149; font-weight: bold;")
        else:
            self.thermal_sitrep.setStyleSheet("color: #d29922; font-weight: bold;")

        # NPU Availability
        metrics = self.telemetry.get_metrics()
        # On MTL-P, device 0 is usually iGPU, device 1 is NPU
        if len(metrics) > 1:
            self.npu_sitrep.setText("NPU: ACTIVE")
            self.npu_sitrep.setStyleSheet("color: #58a6ff; font-weight: bold;")
        elif len(metrics) > 0:
             self.npu_sitrep.setText("NPU: NOT DETECTED")
             self.npu_sitrep.setStyleSheet("color: #8b949e; font-weight: bold;")
        else:
            self.npu_sitrep.setText("NPU: NO TELEMETRY")
            self.npu_sitrep.setStyleSheet("color: #8b949e; font-weight: bold;")

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
