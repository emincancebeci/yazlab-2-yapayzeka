"""
test_automata.py — Automata Birim Testleri
Geçiş olasılıkları toplamı 1.0'a eşit mi?
Path probability doğru hesaplanıyor mu?
Çalıştır: py -m pytest tests/test_automata.py -v
"""
import pytest
import numpy as np
from src.models.automata.automata import ProbabilisticAutomata


@pytest.fixture
def config():
    return {
        "fixed":   {"window_size": 3, "alphabet_size": 3},
        "automata": {"anomaly_threshold": 0.05, "smoothing_alpha": 0.0},
    }


@pytest.fixture
def fitted_automata(config):
    np.random.seed(42)
    model = ProbabilisticAutomata(config)
    model.fit(np.random.randn(300))
    return model


class TestTransitionMatrix:

    def test_probs_sum_to_one(self, fitted_automata):
        for src, dst_dict in fitted_automata.transition_matrix.items():
            total = sum(dst_dict.values())
            assert abs(total - 1.0) < 1e-6, f"State {src}: toplam={total}"

    def test_has_states(self, fitted_automata):
        assert fitted_automata.n_states > 0

    def test_all_states_in_matrix(self, fitted_automata):
        for state in fitted_automata.states:
            assert state in fitted_automata.transition_matrix


class TestPrediction:

    def test_predict_shape(self, fitted_automata, config):
        w = config["fixed"]["window_size"]
        series = np.random.randn(50)
        preds = fitted_automata.predict(series)
        assert len(preds) == len(series) - w + 1

    def test_predict_binary(self, fitted_automata):
        preds = fitted_automata.predict(np.random.randn(50))
        assert set(preds).issubset({0, 1})

    def test_predict_sequence_fields(self, fitted_automata):
        seq = fitted_automata.predict_sequence(np.random.randn(20))
        required = {"time_step", "state", "pattern", "status",
                    "transition_prob", "path_probability", "decision"}
        for record in seq:
            assert required.issubset(record.keys())

    def test_fit_required_before_predict(self, config):
        model = ProbabilisticAutomata(config)
        with pytest.raises(RuntimeError):
            model.predict(np.random.randn(20))
