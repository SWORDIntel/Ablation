from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QTableWidget, QTableWidgetItem, 
    QHeaderView, QLabel, QPushButton, QHBoxLayout
)
from PyQt6.QtCore import Qt, QTimer

class LeaderboardTab(QWidget):
    """
    AEGIS-LAB Ablation Leaderboard.
    Displays ranked results of model edits based on Method 1-5 metrics.
    """
    
    def __init__(self, client):
        super().__init__()
        self.client = client
        self.layout = QVBoxLayout(self)
        
        self.header = QLabel("🏆 Ablation Integrity Leaderboard")
        self.header.setStyleSheet("font-size: 24px; font-weight: bold; color: #58a6ff; margin-bottom: 10px;")
        self.layout.addWidget(self.header)
        
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels([
            "Rank", "Job ID", "Integrity Score", "Elo", "Robustness", "Perf/Watt"
        ])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setStyleSheet("background-color: #161b22; color: #c9d1d9; gridline-color: #30363d;")
        self.layout.addWidget(self.table)
        
        self.refresh_btn = QPushButton("Refresh Rankings")
        self.refresh_btn.clicked.connect(self.refresh_data)
        self.layout.addWidget(self.refresh_btn)
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.refresh_data)
        self.timer.start(10000) # Refresh every 10 seconds

    def refresh_data(self):
        # We need to add a 'get_leaderboard' handler to the Orchestrator IPC
        resp = self.client.request("get_leaderboard")
        if "error" in resp: return
        
        rankings = resp if isinstance(resp, list) else []
        self.table.setRowCount(len(rankings))
        
        for i, entry in enumerate(rankings):
            self.table.setItem(i, 0, QTableWidgetItem(f"#{i+1}"))
            self.table.setItem(i, 1, QTableWidgetItem(entry["job_id"]))
            self.table.setItem(i, 2, QTableWidgetItem(f"{entry['total_score']:.4f}"))
            
            metrics = entry.get("metrics", {})
            self.table.setItem(i, 3, QTableWidgetItem(f"{metrics.get('elo_rating', 0):.0f}"))
            self.table.setItem(i, 4, QTableWidgetItem(f"{metrics.get('adversarial_robustness', 0):.2f}"))
            self.table.setItem(i, 5, QTableWidgetItem(f"{metrics.get('perf_per_watt', 0):.2f}"))
            
            # Highlight top rank
            if i == 0:
                for col in range(6):
                    self.table.item(i, col).setForeground(Qt.GlobalColor.yellow)
