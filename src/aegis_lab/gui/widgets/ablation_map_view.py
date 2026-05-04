import numpy as np
import pyqtgraph.opengl as gl
from PyQt6.QtCore import pyqtSlot, QTimer, Qt
from PyQt6.QtGui import QColor, QVector3D
import logging

logger = logging.getLogger(__name__)

class AblationMapView(gl.GLViewWidget):
    """
    3D Visualization of model layers and behavioral atoms.
    Layers are stacked along the Z-axis. Atoms are points within layers.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setCameraPosition(distance=30, elevation=30, azimuth=45)
        self.setBackgroundColor('#09111f')

        # Add a base grid
        self.grid = gl.GLGridItem()
        self.grid.scale(2, 2, 1)
        self.grid.setDepthValue(10) # Constant draw order
        self.addItem(self.grid)

        # Storage for atoms
        # atom_id -> {pos: (x, y, z), color: (r, g, b, a), size: s, score: f}
        self.atoms = {}
        
        # Scatter plot for all atoms
        self.scatter = gl.GLScatterPlotItem()
        self.addItem(self.scatter)

        # Mock data timer for testing if needed
        self.test_timer = QTimer()
        self.test_timer.timeout.connect(self._add_mock_atom)
        
        self.layer_count = 12
        self.atoms_per_layer = 64
        self._init_base_layers()

    def _init_base_layers(self):
        """Initialize a faint grid of possible atom locations."""
        pos = []
        colors = []
        sizes = []
        
        for l in range(self.layer_count):
            z = l * 2.0
            side = int(np.sqrt(self.atoms_per_layer))
            for i in range(side):
                for j in range(side):
                    x = (i - side/2) * 1.5
                    y = (j - side/2) * 1.5
                    pos.append([x, y, z])
                    colors.append([0.2, 0.3, 0.5, 0.1]) # Faint blue
                    sizes.append(3)
        
        self.base_pos = np.array(pos)
        self.base_colors = np.array(colors)
        self.base_sizes = np.array(sizes)
        self._update_scatter()

    def _update_scatter(self):
        """Merges base layers and active atoms for rendering."""
        if not self.atoms:
            self.scatter.setData(pos=self.base_pos, color=self.base_colors, size=self.base_sizes, pxMode=True)
            return

        active_pos = []
        active_colors = []
        active_sizes = []

        for atom_id, data in self.atoms.items():
            active_pos.append(data['pos'])
            # Map score to color (0.0 -> blue, 1.0 -> red/gold)
            score = data.get('score', 0.5)
            r = min(1.0, score * 1.5)
            g = min(1.0, (1.0 - abs(score - 0.5) * 2))
            b = min(1.0, (1.0 - score) * 1.5)
            active_colors.append([r, g, b, 0.9])
            active_sizes.append(8 + score * 10)

        all_pos = np.vstack([self.base_pos, np.array(active_pos)])
        all_colors = np.vstack([self.base_colors, np.array(active_colors)])
        all_sizes = np.concatenate([self.base_sizes, np.array(active_sizes)])

        self.scatter.setData(pos=all_pos, color=all_colors, size=all_sizes, pxMode=True)

    @pyqtSlot(dict)
    def add_atom(self, atom_data):
        """
        Adds or updates an atom in the visualization.
        Expected keys: atom_id, layer_idx, x, y, score
        """
        atom_id = atom_data.get('atom_id')
        layer = atom_data.get('layer_idx', 0)
        x_idx = atom_data.get('x', 0)
        y_idx = atom_data.get('y', 0)
        score = atom_data.get('score', 0.0)

        # Map indices to 3D space
        side = int(np.sqrt(self.atoms_per_layer))
        x = (x_idx - side/2) * 1.5
        y = (y_idx - side/2) * 1.5
        z = layer * 2.0

        self.atoms[atom_id] = {
            'pos': [x, y, z],
            'score': score
        }
        self._update_scatter()

    def clear(self):
        self.atoms = {}
        self._update_scatter()

    def _add_mock_atom(self):
        import random
        mock_id = f"atom_{random.randint(0, 1000)}"
        self.add_atom({
            'atom_id': mock_id,
            'layer_idx': random.randint(0, self.layer_count - 1),
            'x': random.randint(0, 7),
            'y': random.randint(0, 7),
            'score': random.random()
        })

    def start_mock_stream(self):
        self.test_timer.start(200)

    def stop_mock_stream(self):
        self.test_timer.stop()
