"""Unit tests for Phase 4 signal generation."""

import numpy as np

from rnn_lstm_sinusoid_demixing.data.signal_generator import (
    build_signals,
    generate_clean_sinusoid,
    generate_noisy_composite,
    generate_time_axis,
)
from rnn_lstm_sinusoid_demixing.shared.config import SignalConfig

_SR = 100
_DUR = 1.0
_N = _SR  # int(100 * 1.0) = 100 samples
_TIME = generate_time_axis(_SR, _DUR)
_CLEAN = [generate_clean_sinusoid(_TIME, f) for f in [1.0, 3.0, 5.0, 7.0]]
_CFG = SignalConfig(sampling_rate=_SR, duration_seconds=_DUR)


class TestGenerateTimeAxis:
    def test_shape(self) -> None:
        assert generate_time_axis(_SR, _DUR).shape == (_N,)

    def test_dtype(self) -> None:
        assert generate_time_axis(_SR, _DUR).dtype == np.float32

    def test_starts_at_zero(self) -> None:
        assert generate_time_axis(_SR, _DUR)[0] == 0.0

    def test_step_equals_one_over_sr(self) -> None:
        t = generate_time_axis(_SR, _DUR)
        assert abs(float(t[1] - t[0]) - 1.0 / _SR) < 1e-6

    def test_no_nan_inf(self) -> None:
        assert np.all(np.isfinite(generate_time_axis(_SR, _DUR)))


class TestGenerateCleanSinusoid:
    def test_shape(self) -> None:
        assert generate_clean_sinusoid(_TIME, 1.0).shape == (_N,)

    def test_dtype(self) -> None:
        assert generate_clean_sinusoid(_TIME, 1.0).dtype == np.float32

    def test_amplitude_bounds(self) -> None:
        amp = 2.0
        s = generate_clean_sinusoid(_TIME, 1.0, amplitude=amp)
        assert np.all(np.abs(s) <= amp + 1e-5)

    def test_unit_amplitude_bounds(self) -> None:
        s = generate_clean_sinusoid(_TIME, 1.0)
        assert np.all(np.abs(s) <= 1.0 + 1e-5)

    def test_no_nan_inf(self) -> None:
        assert np.all(np.isfinite(generate_clean_sinusoid(_TIME, 7.0)))


class TestGenerateNoisyComposite:
    def test_noisy_component_shapes(self) -> None:
        noisy, _ = generate_noisy_composite(_CLEAN, 0.1)
        assert all(n.shape == (_N,) for n in noisy)

    def test_composite_shape(self) -> None:
        _, c = generate_noisy_composite(_CLEAN, 0.1)
        assert c.shape == (_N,)

    def test_composite_dtype(self) -> None:
        _, c = generate_noisy_composite(_CLEAN, 0.1)
        assert c.dtype == np.float32

    def test_composite_equals_sum_of_noisy(self) -> None:
        noisy, composite = generate_noisy_composite(_CLEAN, 0.1)
        expected = np.stack(noisy).sum(axis=0).astype(np.float32)
        np.testing.assert_array_almost_equal(composite, expected)

    def test_no_nan_inf_in_components(self) -> None:
        noisy, _ = generate_noisy_composite(_CLEAN, 0.1)
        assert all(np.all(np.isfinite(n)) for n in noisy)

    def test_no_nan_inf_in_composite(self) -> None:
        _, composite = generate_noisy_composite(_CLEAN, 0.1)
        assert np.all(np.isfinite(composite))

    def test_deterministic_with_same_seed(self) -> None:
        _, c1 = generate_noisy_composite(_CLEAN, 0.1, random_seed=7)
        _, c2 = generate_noisy_composite(_CLEAN, 0.1, random_seed=7)
        np.testing.assert_array_equal(c1, c2)

    def test_different_seeds_produce_different_results(self) -> None:
        _, c1 = generate_noisy_composite(_CLEAN, 0.1, random_seed=0)
        _, c2 = generate_noisy_composite(_CLEAN, 0.1, random_seed=99)
        assert not np.array_equal(c1, c2)

    def test_zero_noise_level_leaves_clean(self) -> None:
        noisy, _ = generate_noisy_composite(_CLEAN, 0.0)
        for clean, n in zip(_CLEAN, noisy):
            np.testing.assert_array_almost_equal(n, clean)

    def test_num_noisy_components_matches_input(self) -> None:
        noisy, _ = generate_noisy_composite(_CLEAN, 0.1)
        assert len(noisy) == len(_CLEAN)


class TestBuildSignals:
    def test_output_shapes(self) -> None:
        t, clean, noisy, composite = build_signals(_CFG)
        assert t.shape == (_N,)
        assert len(clean) == 4
        assert len(noisy) == 4
        assert composite.shape == (_N,)

    def test_composite_is_sum_of_noisy(self) -> None:
        _, _, noisy, composite = build_signals(_CFG)
        expected = np.stack(noisy).sum(axis=0).astype(np.float32)
        np.testing.assert_array_almost_equal(composite, expected)

    def test_all_outputs_finite(self) -> None:
        t, clean, noisy, composite = build_signals(_CFG)
        assert np.all(np.isfinite(t))
        assert all(np.all(np.isfinite(s)) for s in clean)
        assert all(np.all(np.isfinite(s)) for s in noisy)
        assert np.all(np.isfinite(composite))

    def test_all_dtypes_float32(self) -> None:
        t, clean, noisy, composite = build_signals(_CFG)
        assert t.dtype == np.float32
        assert all(s.dtype == np.float32 for s in clean)
        assert all(s.dtype == np.float32 for s in noisy)
        assert composite.dtype == np.float32

    def test_custom_amplitudes_and_phases(self) -> None:
        amps = [2.0, 0.5, 1.5, 1.0]
        phases = [0.0, np.pi / 4, np.pi / 2, np.pi]
        _, clean, _, _ = build_signals(_CFG, amplitudes=amps, phases=phases)
        assert all(np.all(np.abs(s) <= a + 1e-5) for s, a in zip(clean, amps))
