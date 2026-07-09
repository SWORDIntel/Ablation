import sys
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLineEdit, QPushButton, QListWidget, QLabel, QListWidgetItem
from framewerx.aegis_lab.intake.hf_browser import HFModelBrowser

class HFSelectorWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.browser = HFModelBrowser()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search HF models...")
        self.search_btn = QPushButton("Search")
        self.model_list = QListWidget()
        self.download_btn = QPushButton("Download Selection")

        layout.addWidget(QLabel("Search HuggingFace"))
        layout.addWidget(self.search_input)
        layout.addWidget(self.search_btn)
        layout.addWidget(self.model_list)
        layout.addWidget(self.download_btn)

        self.setLayout(layout)

        self.search_btn.clicked.connect(self.perform_search)
        self.download_btn.clicked.connect(self.download_selected)

    def perform_search(self):
        query = self.search_input.text()
        models = self.browser.search_models(query)
        self.model_list.clear()
        for model in models:
            item = QListWidgetItem(model.modelId)
            self.model_list.addItem(item)

    def download_selected(self):
        selected = self.model_list.currentItem()
        if selected:
            self.browser.download_model(selected.text())
