"""Minimal tests for TUI screens/widgets -- imports and class attributes only.

These tests verify that the modules import correctly and that class-level
attributes are defined, without instantiating Textual widgets (which requires
an active event loop and causes test isolation issues).
"""
from __future__ import annotations


class TestHistorySearchImport:
    def test_import(self):
        from koboi.tui.screens.history_search import HistorySearchScreen
        assert HistorySearchScreen is not None

    def test_is_modal_screen(self):
        from textual.screen import ModalScreen
        from koboi.tui.screens.history_search import HistorySearchScreen
        assert issubclass(HistorySearchScreen, ModalScreen)

    def test_has_bindings(self):
        from koboi.tui.screens.history_search import HistorySearchScreen
        assert hasattr(HistorySearchScreen, 'BINDINGS')


class TestSessionManagerImport:
    def test_import(self):
        from koboi.tui.screens.session_manager import SessionManagerScreen
        assert SessionManagerScreen is not None

    def test_is_modal_screen(self):
        from textual.screen import ModalScreen
        from koboi.tui.screens.session_manager import SessionManagerScreen
        assert issubclass(SessionManagerScreen, ModalScreen)

    def test_has_bindings(self):
        from koboi.tui.screens.session_manager import SessionManagerScreen
        assert hasattr(SessionManagerScreen, 'BINDINGS')


class TestTranscriptViewerImport:
    def test_import(self):
        from koboi.tui.screens.transcript_viewer import TranscriptViewerScreen
        assert TranscriptViewerScreen is not None

    def test_is_modal_screen(self):
        from textual.screen import ModalScreen
        from koboi.tui.screens.transcript_viewer import TranscriptViewerScreen
        assert issubclass(TranscriptViewerScreen, ModalScreen)


class TestWelcomeScreenImport:
    def test_import(self):
        from koboi.tui.screens.welcome_screen import WelcomeScreen
        assert WelcomeScreen is not None

    def test_is_modal_screen(self):
        from textual.screen import ModalScreen
        from koboi.tui.screens.welcome_screen import WelcomeScreen
        assert issubclass(WelcomeScreen, ModalScreen)


class TestPlanViewImport:
    def test_import_plan_step(self):
        from koboi.tui.widgets.plan_view import PlanStep
        assert PlanStep is not None

    def test_import_plan_step_widget(self):
        from koboi.tui.widgets.plan_view import PlanStepWidget
        assert PlanStepWidget is not None

    def test_import_plan_view(self):
        from koboi.tui.widgets.plan_view import PlanView
        assert PlanView is not None

    def test_plan_step_dataclass(self):
        from koboi.tui.widgets.plan_view import PlanStep
        step = PlanStep(index=1, description="Test")
        assert step.index == 1
        assert step.description == "Test"
        assert step.completed is False
        assert step.skipped is False

    def test_plan_step_defaults(self):
        from koboi.tui.widgets.plan_view import PlanStep
        step = PlanStep(index=5, description="Do something")
        assert step.completed is False
        assert step.skipped is False

    def test_plan_step_completed(self):
        from koboi.tui.widgets.plan_view import PlanStep
        step = PlanStep(index=1, description="Test", completed=True)
        assert step.completed is True

    def test_plan_step_skipped(self):
        from koboi.tui.widgets.plan_view import PlanStep
        step = PlanStep(index=1, description="Test", skipped=True)
        assert step.skipped is True
