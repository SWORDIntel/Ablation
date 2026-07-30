from textual.app import App, ComposeResult
from textual.widgets import Header, Footer
from aegis_lab.gui.widgets.mission_monitor import MissionMonitor

class AegisTUI(App):
    BINDINGS = [("q", "quit", "Quit the application")]
    CSS = """
    #cockpit-grid {
        grid-size: 2 2;
        grid-columns: 1fr 1fr;
        grid-rows: 1fr 1fr;
    }
    #message-log {
        column-span: 2;
        background: $surface-darken-1;
    }
    """

    def compose(self) -> ComposeResult:
        yield Header()
        yield MissionMonitor()
        yield Footer()

if __name__ == "__main__":
    AegisTUI().run()
