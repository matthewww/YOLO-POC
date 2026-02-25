"""Tests for the Pydantic settings layer."""

from fpv_vision.config.settings import ModelSettings, TrainingSettings, DataSettings, Settings


def test_model_settings_defaults():
    s = ModelSettings()
    assert s.confidence == 0.5
    assert s.iou_threshold == 0.45
    assert s.imgsz == 640


def test_training_settings_defaults():
    s = TrainingSettings()
    assert s.epochs == 100
    assert s.batch_size == 16
    assert s.optimizer == "auto"


def test_data_settings_defaults():
    s = DataSettings()
    assert s.dir == "data/fpv"
    assert s.yaml == "data/fpv/data.yaml"


def test_settings_classes_loaded():
    s = Settings()
    assert len(s.classes) > 0
    assert "motor" in s.classes
    assert "camera" in s.classes
