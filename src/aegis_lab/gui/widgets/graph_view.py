from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import Qt, QPointF, QRectF, QPropertyAnimation, pyqtProperty, QEasingCurve
from PyQt6.QtGui import QPainter, QColor, QPen, QBrush, QFont, QPainterPath

class GraphView(QWidget):
    STAGES = ["intake", "probe", "extract", "clean", "verify", "quant", "promote"]
    
    STATUS_COLORS = {
        "pending": QColor(48, 54, 61),      # Darker grey for pending
        "running": QColor(88, 166, 255),   # Blue accent for running
        "succeeded": QColor(63, 185, 80),  # Success green
        "failed": QColor(248, 81, 73)      # Error red
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(200)
        self.node_status = {stage: "pending" for stage in self.STAGES}
        
        # Property for QPropertyAnimation
        self._pulse_factor = 0.0
        
        # Setup Animation
        self.pulse_anim = QPropertyAnimation(self, b"pulse_factor")
        self.pulse_anim.setDuration(1500)
        self.pulse_anim.setStartValue(0.0)
        self.pulse_anim.setEndValue(1.0)
        self.pulse_anim.setLoopCount(-1)
        self.pulse_anim.setEasingCurve(QEasingCurve.Type.InOutSine)
        self.pulse_anim.start()

    @pyqtProperty(float)
    def pulse_factor(self):
        return self._pulse_factor

    @pulse_factor.setter
    def pulse_factor(self, value):
        self._pulse_factor = value
        # Only repaint if there is a running stage to save CPU
        if "running" in self.node_status.values():
            self.update()

    def set_stage_status(self, stage_name, status):
        if stage_name in self.node_status:
            self.node_status[stage_name] = status.lower()
            self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        
        width = self.width()
        height = self.height()
        
        # Background fill (matches global theme if not inherited)
        painter.fillRect(self.rect(), QColor("#0d1117"))
        
        node_radius = 20
        margin = 60
        spacing = (width - 2 * margin) / (len(self.STAGES) - 1)
        y = height / 2 - 10
        
        # Draw links with smooth curves
        for i in range(len(self.STAGES) - 1):
            x1 = margin + i * spacing
            x2 = margin + (i + 1) * spacing
            
            path = QPainterPath()
            path.moveTo(x1, y)
            
            # Create a slight S-curve for better aesthetics
            cp1 = QPointF(x1 + spacing / 2, y)
            cp2 = QPointF(x2 - spacing / 2, y)
            path.cubicTo(cp1, cp2, QPointF(x2, y))
            
            pen = QPen(QColor(48, 54, 61), 3) # Link color
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(pen)
            painter.drawPath(path)

        # Draw nodes
        for i, stage in enumerate(self.STAGES):
            x = margin + i * spacing
            status = self.node_status.get(stage, "pending")
            color = self.STATUS_COLORS.get(status, QColor(48, 54, 61))
            
            current_radius = node_radius
            
            # Pulsing effect for running stages
            if status == "running":
                pulse_val = self._pulse_factor
                
                # Outer glow pulse
                glow_color = QColor(color)
                glow_color.setAlpha(int(100 * (1 - pulse_val)))
                painter.setBrush(QBrush(glow_color))
                painter.setPen(Qt.PenStyle.NoPen)
                glow_radius = node_radius + (12 * pulse_val)
                painter.drawEllipse(QPointF(x, y), glow_radius, glow_radius)
                
                # Slight radius variation
                current_radius += 2 * pulse_val

            # Draw node circle
            painter.setBrush(QBrush(color))
            node_pen = QPen(QColor("#30363d"), 2)
            if status == "running":
                node_pen.setColor(QColor("#58a6ff"))
                node_pen.setWidth(2)
            elif status == "succeeded":
                node_pen.setColor(QColor("#3fb950"))
            painter.setPen(node_pen)
            painter.drawEllipse(QPointF(x, y), current_radius, current_radius)
            
            # Draw label
            label_color = QColor("#c9d1d9")
            if status == "running":
                label_color = QColor("#58a6ff")
            elif status == "succeeded":
                label_color = QColor("#3fb950")
            
            painter.setPen(label_color)
            font = QFont("Inter" if "win32" in Qt.GlobalColor.__module__ else "Arial", 9, QFont.Weight.Bold)
            painter.setFont(font)
            
            label_rect = QRectF(x - spacing/2, y + node_radius + 15, spacing, 20)
            painter.drawText(label_rect, Qt.AlignmentFlag.AlignCenter, stage.capitalize())

    def update_from_stages(self, stages):
        # Reset all to pending before updating
        for s in self.STAGES:
            self.node_status[s] = "pending"
            
        # Map stage_name to status
        for stage in stages:
            name = stage.get("stage_name", "").lower()
            status = stage.get("status", "pending").lower()
            # Handle mapping substrings like 'intake' in 'job-s0'
            for s in self.STAGES:
                if s in name:
                    self.node_status[s] = status
                    break
        self.update()
