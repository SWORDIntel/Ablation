from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, 
    QPushButton, QLabel, QListWidget, QSplitter,
    QScrollArea, QFrame, QMessageBox, QListWidgetItem
)
from PyQt6.QtCore import Qt, pyqtSignal

class ChatBubble(QFrame):
    def __init__(self, text, is_operator=True):
        super().__init__()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        self.bubble = QLabel(text)
        self.bubble.setWordWrap(True)
        self.bubble.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        
        # Modern bubble styling for Dark Mode
        bg_color = "#58a6ff" if is_operator else "#21262d"
        text_color = "#ffffff" if is_operator else "#c9d1d9"
        border_radius = "12px"
        
        self.bubble.setStyleSheet(f"""
            QLabel {{
                background-color: {bg_color};
                color: {text_color};
                border-radius: {border_radius};
                padding: 10px 14px;
                font-size: 13px;
                border: 1px solid {bg_color if is_operator else "#30363d"};
            }}
        """)
        
        if is_operator:
            layout.addStretch()
            layout.addWidget(self.bubble)
        else:
            layout.addWidget(self.bubble)
            layout.addStretch()

class ChatView(QWidget):
    """
    Refined QIHSE-Accelerated RAG Operator Chat interface with Bubble Layout.
    Updated for Global Dark Mode.
    """
    send_message = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.all_atoms = [] # To store full atom metadata
        
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(10, 10, 10, 10)
        
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.main_layout.addWidget(self.splitter)
        
        # --- Main Chat Area ---
        self.chat_container = QWidget()
        self.chat_layout = QVBoxLayout(self.chat_container)
        self.chat_layout.setContentsMargins(0, 0, 0, 0)
        
        # History Scroll Area
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll_area.setStyleSheet("background-color: #0d1117; border: none;")
        
        self.history_widget = QWidget()
        self.history_layout = QVBoxLayout(self.history_widget)
        self.history_layout.addStretch()
        self.scroll_area.setWidget(self.history_widget)
        
        self.header_label = QLabel("Interrogation History (Ablation-Aware)")
        self.header_label.setStyleSheet("color: #58a6ff; font-weight: bold; font-size: 14px;")
        self.chat_layout.addWidget(self.header_label)
        self.chat_layout.addWidget(self.scroll_area)
        
        # Loading Indicator
        self.loading_label = QLabel("<i>Retrieving behavioral context...</i>")
        self.loading_label.setStyleSheet("color: #8b949e; margin-left: 10px;")
        self.loading_label.hide()
        self.chat_layout.addWidget(self.loading_label)
        
        # Input Area
        self.input_layout = QHBoxLayout()
        self.prompt_input = QLineEdit()
        self.prompt_input.setPlaceholderText("Enter probe query (e.g., 'Analyze refusal triggers')...")
        self.prompt_input.setMinimumHeight(40)
        self.prompt_input.setStyleSheet("""
            QLineEdit {
                background-color: #161b22;
                border: 1px solid #30363d;
                border-radius: 6px;
                padding: 0 10px;
                color: #c9d1d9;
            }
        """)
        self.prompt_input.returnPressed.connect(self._on_send)
        
        self.send_btn = QPushButton("Send")
        self.send_btn.setMinimumHeight(40)
        # Inherit global QPushButton styles, but can override if needed
        self.send_btn.clicked.connect(self._on_send)
        
        self.input_layout.addWidget(self.prompt_input)
        self.input_layout.addWidget(self.send_btn)
        self.chat_layout.addLayout(self.input_layout)
        
        self.splitter.addWidget(self.chat_container)
        
        # --- Context Sidebar (RAG Atoms) ---
        self.context_container = QWidget()
        self.context_layout = QVBoxLayout(self.context_container)
        self.context_layout.setContentsMargins(10, 0, 0, 0)
        
        self.sidebar_label = QLabel("Retrieved Behavioral Atoms")
        self.sidebar_label.setStyleSheet("color: #58a6ff; font-weight: bold;")
        self.context_layout.addWidget(self.sidebar_label)
        
        # Search Bar for Atoms
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Filter atoms...")
        self.search_input.setStyleSheet("background-color: #161b22; border: 1px solid #30363d; color: #c9d1d9;")
        self.search_input.textChanged.connect(self._filter_atoms)
        self.context_layout.addWidget(self.search_input)
        
        self.atom_list = QListWidget()
        self.atom_list.setStyleSheet("""
            QListWidget {
                background-color: #0d1117;
                border: 1px solid #30363d;
                color: #c9d1d9;
            }
            QListWidget::item:selected {
                background-color: #21262d;
                color: #58a6ff;
            }
        """)
        self.atom_list.itemClicked.connect(self._on_atom_clicked)
        self.context_layout.addWidget(self.atom_list)
        
        self.splitter.addWidget(self.context_container)
        self.splitter.setStretchFactor(0, 3)
        self.splitter.setStretchFactor(1, 1)

    def _on_send(self):
        text = self.prompt_input.text().strip()
        if text:
            self.add_bubble(text, is_operator=True)
            self.prompt_input.clear()
            self.loading_label.show()
            self.send_message.emit(text)

    def add_bubble(self, text, is_operator=False):
        bubble = ChatBubble(text, is_operator)
        # Add before the stretch
        self.history_layout.insertWidget(self.history_layout.count() - 1, bubble)
        # Scroll to bottom
        self.scroll_area.verticalScrollBar().setValue(
            self.scroll_area.verticalScrollBar().maximum()
        )

    def add_response(self, text: str):
        self.loading_label.hide()
        self.add_bubble(text, is_operator=False)

    def update_context(self, atoms: list):
        self.all_atoms = atoms
        self._filter_atoms()

    def _filter_atoms(self):
        search_text = self.search_input.text().lower()
        self.atom_list.clear()
        for atom in self.all_atoms:
            atom_id = str(atom.get('atom_id', 'Unknown'))
            score = atom.get('score', 0.0)
            if search_text in atom_id.lower():
                item = QListWidgetItem(f"{atom_id} (Sim: {score:.4f})")
                item.setData(Qt.ItemDataRole.UserRole, atom)
                self.atom_list.addItem(item)

    def _on_atom_clicked(self, item):
        atom_data = item.data(Qt.ItemDataRole.UserRole)
        if atom_data:
            # Enhanced interactive feedback for selection
            details = "\n".join([f"{k}: {v}" for k, v in atom_data.items()])
            msg = QMessageBox(self)
            msg.setWindowTitle("Atom Selection")
            msg.setText(f"You have selected: {atom_data.get('atom_id')}")
            msg.setInformativeText("Would you like to stage this atom for ablation?")
            msg.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if msg.exec() == QMessageBox.StandardButton.Yes:
                self.add_response(f"Atom '{atom_data.get('atom_id')}' has been staged for your ablation mission.")
