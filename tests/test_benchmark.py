"""Level 3's harness, as tests.

Everything except "a GPU answers" is provable on the Linux dev box, and this
file is the argument that it is: the chunking, the RTF arithmetic, the
first-minute/last-minute split, the p95, the incremental writes, the OOM path
and the table rendering all run here with no GPU, no weights and no network.

Two seams make that possible (D58):

* **An injected clock.** `Harness` takes `clock` and `sleep`, so a ten-minute
  run is ten simulated minutes and the wall-clock minute windows land on exactly
  the chunks the test says they should. Nothing here sleeps.
* **An injected engine.** `StubEngine` and the `ScriptedEngine` below stand in
  for CTranslate2. The one test that must exercise the real `faster-whisper`
  call path does it with a fake `WhisperModel`, which is how the lazy-generator
  trap is caught without downloading a model.

What is *not* here, and cannot be: `get_cuda_device_count() == 1` against a real
device, VRAM polling (there is no `nvidia-smi` on this box), thermal throttling,
and a genuine CUDA OOM. Those are owed to the Windows machine, along with the
ten-minute recording that is the harness's actual input.
"""

from __future__ import annotations

import json

import pytest

from speech_translator import config as cfg
from speech_translator.tools import benchmark as bm
from speech_translator.tools import dump_utterances

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


class FakeClock:
    """Simulated wall clock. `sleep` advances it; nothing waits."""

    def __init__(self, t: float = 0.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.t += float(seconds)


class ScriptedEngine:
    """An engine whose per-call cost is a function of the call index."""

    def __init__(self, clock: FakeClock, proc_fn, load_s: float = 5.0) -> None:
        self.clock = clock
        self.proc_fn = proc_fn
        self.load_s = load_s
        self.calls = 0
        self.seen: list[bytes] = []

    def load(self) -> None:
        self.clock.sleep(self.load_s)

    def transcribe(self, pcm: bytes) -> str:
        self.seen.append(pcm)
        self.clock.sleep(self.proc_fn(self.calls))
        self.calls += 1
        return "texto"

    def unload(self) -> None:
        pass


def chunks_of(count: int, duration_ms: int = 1000) -> list[bm.Chunk]:
    return [
        bm.Chunk(i, b"\x00" * (duration_ms * bm.cfg.SAMPLE_RATE * 2 // 1000), duration_ms, "silence")
        for i in range(count)
    ]


MEASURED_MT = {"p95_ms": 300.0, "source": "measured", "n": 10, "samples_ms": [300.0]}
NO_MT = {"p95_ms": None, "source": "unmeasured", "n": 0, "samples_ms": [],
         "error": "GOOGLE_TRANSLATE_API_KEY unset"}


def harness(tmp_path, chunks, clock, *, minutes=1.0, engine_factory=None, mt=None, **kw):
    return bm.Harness(
        chunks=chunks,
        minutes=minutes,
        device="cpu",
        language="es",
        config=cfg.Config(),  # the defaults, not this machine's .env
        out_dir=tmp_path,
        engine_factory=engine_factory,
        poller=None,
        mt=mt or MEASURED_MT,
        clock=clock,
        sleep=clock.sleep,
        settle_s=0.0,
        **kw,
    )


def results_shell() -> dict:
    return {
        "harness": {"minutes": 1.0, "device": "cpu", "engine": "stub", "language": "es"},
        "chunks": {"mode": "segmenter", "count": 3, "audio_s": 3.0,
                   "duration_ms": {"p50": 1000, "p95": 1000}},
        "host": {"platform": "linux", "python": "3.12"},
        "mt": MEASURED_MT,
        "rows": [],
    }


# --------------------------------------------------------------------------
# Chunking (D57)
# --------------------------------------------------------------------------


def test_segmenter_chunking_matches_dump_utterances():
    """The harness must cut the audio the way the app's own tool does.

    `dump_utterances` is the independent reference: it drives the same Segmenter
    through the same `open_source()` seam and reports boundaries. If the harness
    ever grew its own chunking, RTF would be measured over audio the app never
    produces.
    """
    chunks, summary = bm.build_chunks(cfg.SPEECH_FIXTURE_PATH, mode="segmenter")
    reference = dump_utterances.dump(input_wav=str(cfg.SPEECH_FIXTURE_PATH))

    assert summary["mode"] == "segmenter"
    assert len(chunks) == reference["emitted"] == len(reference["utterances"])
    assert chunks, "the committed speech fixture must produce utterances (D56)"

    for chunk, row in zip(chunks, reference["utterances"]):
        assert chunk.duration_ms == row["duration_ms"]
        assert chunk.closed_by == row["closed_by"]
        # Byte-identical, not merely the same length: this is the audio Whisper
        # is handed, padding and trimming included (D55).
        assert len(chunk.pcm) == row["bytes"]
        assert len(chunk.pcm) == chunk.duration_ms * bm.cfg.SAMPLE_WIDTH_BYTES * 16


def test_segmenter_chunk_summary_reports_the_distribution():
    """The distribution is the finding D57 is about — it has to be recorded."""
    _chunks, summary = bm.build_chunks(cfg.SPEECH_FIXTURE_PATH, mode="segmenter")
    assert summary["duration_ms"]["max"] <= cfg.Config().max_utterance_ms
    assert summary["audio_s"] < summary["source_audio_s"]  # silence was dropped
    assert set(summary["closed_by"]) == {"silence", "max_length"}
    assert summary["speech_s"] is not None


@pytest.mark.parametrize("chunk_ms,expected_ms", [(4000, 4000), (1500, 1472)])
def test_fixed_chunking_is_whole_frames(chunk_ms, expected_ms):
    """D52's bracket. 1500 ms is not a whole number of 32 ms frames, so the
    harness reports what it actually cut — 1472 — rather than the number asked
    for."""
    chunks, summary = bm.build_chunks(
        cfg.SPEECH_FIXTURE_PATH, mode="fixed", chunk_ms=chunk_ms
    )
    assert summary["mode"] == "fixed"
    assert summary["chunk_ms"] == chunk_ms
    assert all(ch.duration_ms == expected_ms for ch in chunks[:-1])
    assert all(len(ch.pcm) == ch.duration_ms * 32 for ch in chunks)
    assert chunks[-1].duration_ms <= expected_ms  # the tail is kept, not padded
    assert all(ch.closed_by is None for ch in chunks)


def test_chunks_are_built_once_and_shared_by_every_configuration(tmp_path):
    """Otherwise an RTF difference between rows could be a difference in what
    was measured rather than in the model."""
    clock = FakeClock()
    chunks = chunks_of(3)
    engines: list[ScriptedEngine] = []

    def factory(_rc):
        engine = ScriptedEngine(clock, lambda _k: 0.25)
        engines.append(engine)
        return engine

    h = harness(tmp_path, chunks, clock, minutes=0.05, engine_factory=factory)
    h.run_matrix([bm.RunConfig("a", "float16"), bm.RunConfig("b", "float16")], results_shell())

    assert len(engines) == 2
    # Identity, not equality: the same byte objects reached both configurations.
    assert [id(p) for p in engines[0].seen] == [id(p) for p in engines[1].seen]


# --------------------------------------------------------------------------
# RTF arithmetic (BENCHMARK.md steps 5 and 6)
# --------------------------------------------------------------------------


def test_rtf_is_over_the_whole_run_and_excludes_load_and_warmup(tmp_path):
    clock = FakeClock()
    # 1 s chunks. Warm-up costs 10 s; every other call costs 0.4 s.
    engine = ScriptedEngine(clock, lambda k: 10.0 if k == 0 else 0.4, load_s=30.0)
    h = harness(tmp_path, chunks_of(5), clock, minutes=1.0, engine_factory=lambda _rc: engine)
    row = h.run_one(bm.RunConfig("small", "float16"))

    assert row["status"] == "ok"
    assert row["load_s"] == 30.0            # timed...
    assert row["warmup_s"] == 10.0          # ...and so is the warm-up
    # 60 s of wall clock: 10 s of warm-up then 0.4 s calls. 1 + 125 calls.
    assert row["chunks_processed"] == 126
    assert row["audio_s"] == 125.0          # warm-up's second is not counted
    assert row["rtf"] == pytest.approx(0.4)  # neither load nor warm-up leaks in
    assert row["rtf_p95_chunk"] == pytest.approx(0.4)


def test_warmup_is_excluded_from_the_first_minute_too(tmp_path):
    """It lands inside the first minute and would dominate the comparison the
    throttling penalty is made of."""
    clock = FakeClock()
    engine = ScriptedEngine(clock, lambda k: 10.0 if k == 0 else 0.4, load_s=0.0)
    h = harness(tmp_path, chunks_of(5), clock, minutes=1.0, engine_factory=lambda _rc: engine)
    row = h.run_one(bm.RunConfig("small", "float16"))

    assert row["rtf_first_min"] == pytest.approx(0.4)  # not (10 + n*0.4)/(n+1)


def test_first_and_last_minute_split_and_throttling_penalty(tmp_path):
    """A run that is fast for two minutes and then half as fast."""
    clock = FakeClock()
    engine = ScriptedEngine(clock, lambda k: 0.5 if k < 240 else 1.0, load_s=0.0)
    h = harness(tmp_path, chunks_of(4), clock, minutes=3.0, engine_factory=lambda _rc: engine)
    row = h.run_one(bm.RunConfig("small", "float16"))

    assert row["wall_clock_s"] == pytest.approx(180.0)
    assert row["chunks_processed"] == 300           # 240 x 0.5 s + 60 x 1.0 s
    assert row["rtf_first_min"] == pytest.approx(0.5)
    assert row["rtf_last_min"] == pytest.approx(1.0)
    assert row["throttling_penalty"] == pytest.approx(1.0)   # +100 %
    # Whole run, not a slice: 239 warm chunks at 0.5 s plus 60 at 1.0 s.
    assert row["rtf"] == pytest.approx((239 * 0.5 + 60 * 1.0) / 299, abs=1e-4)
    assert row["minute_windows_overlap"] is False


def test_a_short_run_flags_that_its_minute_windows_overlap(tmp_path):
    clock = FakeClock()
    engine = ScriptedEngine(clock, lambda _k: 0.5, load_s=0.0)
    h = harness(tmp_path, chunks_of(4), clock, minutes=0.5, engine_factory=lambda _rc: engine)
    row = h.run_one(bm.RunConfig("small", "float16"))
    assert row["minute_windows_overlap"] is True


def test_the_audio_loops_and_laps_are_recorded(tmp_path):
    """D59: ten minutes is wall clock, so a shorter recording is looped."""
    clock = FakeClock()
    engine = ScriptedEngine(clock, lambda _k: 1.0, load_s=0.0)
    h = harness(tmp_path, chunks_of(4), clock, minutes=1.0, engine_factory=lambda _rc: engine)
    row = h.run_one(bm.RunConfig("small", "float16"))

    assert row["chunks_processed"] == 60
    assert row["laps"] == pytest.approx(15.0)
    records = [json.loads(line) for line in
               (tmp_path / "chunks-small-float16.jsonl").read_text(encoding="utf-8").splitlines()]
    assert records[0]["lap"] == 0 and records[4]["lap"] == 1
    assert records[4]["chunk"] == 0  # back to the first chunk on the second lap


def test_window_membership_is_start_based():
    records = [
        {"wall_start_s": 0.0, "audio_s": 1.0, "proc_s": 0.5},
        {"wall_start_s": 59.9, "audio_s": 1.0, "proc_s": 0.5},
        {"wall_start_s": 60.0, "audio_s": 1.0, "proc_s": 0.5},
    ]
    assert len(bm.window(records, 0.0, 60.0)) == 2  # a chunk belongs to one window


# --------------------------------------------------------------------------
# Percentiles and gate 3 (D25, D51)
# --------------------------------------------------------------------------


def test_percentile_is_linear_interpolation():
    assert bm.percentile([1, 2, 3, 4], 95) == pytest.approx(3.85)
    assert bm.percentile([10], 95) == 10.0
    assert bm.percentile([1, 2, 3, 4, 5], 50) == 3.0
    assert bm.percentile([5, 1, 3], 100) == 5.0
    with pytest.raises(ValueError):
        bm.percentile([], 95)


def test_word_age_is_d25s_arithmetic():
    # D25's own worked example: silence 600, MT 300, D 4000, RTF 0.5 -> 6.9 s.
    assert bm.word_age_ms(600, 300, 4000, 0.5) == pytest.approx(6900.0)


def test_gate_three_is_a_p95_over_the_real_chunks_with_the_cap_bound_beside_it(tmp_path):
    """The selection column is a percentile over the distribution the Segmenter
    actually produces; the cap bound is reported next to it because that is the
    arithmetic D25 used to derive `max_utterance_ms`."""
    clock = FakeClock()
    # 2 s chunks at RTF 0.25 — well short of the 4 s cap, as real utterances are.
    chunks = chunks_of(4, duration_ms=2000)
    engine = ScriptedEngine(clock, lambda _k: 0.5, load_s=0.0)
    h = harness(tmp_path, chunks, clock, minutes=1.0, engine_factory=lambda _rc: engine)
    row = h.run_one(bm.RunConfig("small", "float16"))

    # 600 + 300 + 2000 x 1.25
    assert row["word_age_p95_ms"] == pytest.approx(3400.0)
    # 600 + 300 + 4000 x 1.25 — the bound, larger, and reported separately
    assert row["word_age_at_cap_ms"] == pytest.approx(5900.0)
    assert row["gate_word_age"] is True
    assert row["word_age_p95_ms"] < row["word_age_at_cap_ms"]


def test_gate_rtf_is_sustained_not_just_the_average(tmp_path):
    """A configuration that only fails after it heats up still fails gate 1."""
    clock = FakeClock()
    engine = ScriptedEngine(clock, lambda k: 0.1 if k < 400 else 0.9, load_s=0.0)
    h = harness(tmp_path, chunks_of(4), clock, minutes=2.0, engine_factory=lambda _rc: engine)
    row = h.run_one(bm.RunConfig("small", "float16"))

    assert row["rtf"] < bm.GATE_RTF_MAX          # the average passes...
    assert row["rtf_last_min"] > bm.GATE_RTF_MAX  # ...the last minute does not
    assert row["gate_rtf"] is False


# --------------------------------------------------------------------------
# The measured MT round trip (D51)
# --------------------------------------------------------------------------


def test_mt_roundtrip_without_a_key_is_missing_not_300():
    c = cfg.Config(google_translate_api_key=None)
    mt = bm.measure_mt_roundtrip(c)
    assert mt["p95_ms"] is None          # the term is missing, not defaulted
    assert mt["source"] == "unmeasured"
    assert mt["samples_ms"] == [] and mt["n"] == 0
    assert "D51" in mt["error"]


def test_mt_roundtrip_takes_the_p95_of_ten_calls():
    class Response:
        def raise_for_status(self): return None

    calls = []

    def post(url, **kwargs):
        calls.append((url, kwargs))
        return Response()

    c = cfg.Config(google_translate_api_key="test-key")
    mt = bm.measure_mt_roundtrip(c, samples=10, post=post)

    assert mt["source"] == "measured"
    assert mt["n"] == 10 and len(calls) == 10
    assert mt["p95_ms"] == pytest.approx(bm.percentile(mt["samples_ms"], 95), abs=0.05)
    assert mt["chars_billed"] == len(bm._MT_PROBE) * 10  # billed against D31's budget
    assert calls[0][1]["params"] == {"key": "test-key"}


def test_an_unmeasured_mt_term_leaves_gate_three_unknown(tmp_path):
    """The failure D51 named: a guessed 300 ms quietly entering a gate."""
    clock = FakeClock()
    engine = ScriptedEngine(clock, lambda _k: 0.25, load_s=0.0)
    h = harness(tmp_path, chunks_of(4), clock, minutes=0.2,
                engine_factory=lambda _rc: engine, mt=NO_MT)
    row = h.run_one(bm.RunConfig("small", "float16"))

    assert row["mt_source"] == "unmeasured"
    assert row["word_age_p95_ms"] is None
    assert row["word_age_at_cap_ms"] is None
    assert row["gate_word_age"] is None                  # unknown, not "pass"
    assert row["rtf"] is not None                        # the rest still measured

    results = results_shell() | {"rows": [row], "mt": NO_MT}
    markdown = bm.render_markdown(results)
    assert "MT unmeasured" in markdown
    assert "unknown" in markdown


def test_an_assumed_mt_term_is_labelled_wherever_it_appears(tmp_path):
    clock = FakeClock()
    engine = ScriptedEngine(clock, lambda _k: 0.25, load_s=0.0)
    assumed = {"p95_ms": 300.0, "source": "assumed", "n": 0, "samples_ms": []}
    h = harness(tmp_path, chunks_of(4), clock, minutes=0.2,
                engine_factory=lambda _rc: engine, mt=assumed)
    row = h.run_one(bm.RunConfig("small", "float16"))

    assert row["mt_source"] == "assumed"
    assert row["word_age_p95_ms"] is not None
    markdown = bm.render_markdown(results_shell() | {"rows": [row], "mt": assumed})
    assert "MT assumed" in markdown
    assert "not evidence" in markdown


# --------------------------------------------------------------------------
# Failure containment — the reason D51 wrote this file
# --------------------------------------------------------------------------


def test_an_oom_is_a_failed_row_and_the_matrix_continues(tmp_path):
    """Losing rows 4-7 because row 3 died is the failure being prevented."""
    clock = FakeClock()
    matrix = [bm.RunConfig(name, "float16") for name in ("small", "medium", "turbo", "large")]

    def factory(rc):
        if rc.model_size == "turbo":
            return bm.StubEngine(rc, rtf=0.3, load_s=1.0, sleep=clock.sleep, fail_at_chunk=2,
                                 error=RuntimeError("CUDA failed with error out of memory"))
        return bm.StubEngine(rc, rtf=0.3, load_s=1.0, sleep=clock.sleep)

    h = harness(tmp_path, chunks_of(4), clock, minutes=0.2, engine_factory=factory)
    results = h.run_matrix(matrix, results_shell())

    statuses = [(r["model_size"], r["status"], r["failure"]) for r in results["rows"]]
    assert statuses == [
        ("small", "ok", None),
        ("medium", "ok", None),
        ("turbo", "failed", "oom"),
        ("large", "ok", None),          # the run did not stop
    ]
    failed = results["rows"][2]
    assert failed["stage"] == "transcribe"
    assert failed["chunks_processed"] == 2   # the partial work is kept
    assert failed["gate_rtf"] is False and failed["gate_vram"] is False

    on_disk = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))
    assert len(on_disk["rows"]) == 4


def test_a_non_oom_failure_is_classified_as_an_error(tmp_path):
    """The control: the OOM classifier must be able to say no."""
    clock = FakeClock()
    rc = bm.RunConfig("small", "float16")
    engine = bm.StubEngine(rc, sleep=clock.sleep, fail_at_chunk=1,
                           error=ValueError("unsupported compute type"))
    h = harness(tmp_path, chunks_of(4), clock, minutes=0.2, engine_factory=lambda _rc: engine)
    row = h.run_one(rc)

    assert row["status"] == "failed"
    assert row["failure"] == "error"
    assert row["gate_vram"] is None      # not an OOM, so VRAM is simply unknown


def test_a_load_failure_is_a_failed_row_not_an_exception(tmp_path):
    clock = FakeClock()
    rc = bm.RunConfig("large-v3", "float16")
    engine = bm.StubEngine(rc, sleep=clock.sleep, fail_at_load=True,
                           error=RuntimeError("CUDA out of memory"))
    h = harness(tmp_path, chunks_of(4), clock, minutes=0.2, engine_factory=lambda _rc: engine)
    row = h.run_one(rc)

    assert row["status"] == "failed" and row["failure"] == "oom"
    assert row["stage"] == "load"
    assert row["chunks_processed"] == 0


@pytest.mark.parametrize("message,expected", [
    ("CUDA failed with error out of memory", "oom"),
    ("RuntimeError: CUBLAS_STATUS_ALLOC_FAILED", "oom"),
    ("std::bad_alloc", "oom"),
    ("cudaErrorMemoryAllocation", "oom"),
    ("Invalid compute type int8_float16 for device cpu", "error"),
    ("connection reset", "error"),
])
def test_failure_classification(message, expected):
    assert bm.classify_failure(RuntimeError(message)) == expected


def test_results_are_written_after_every_row(tmp_path):
    """A kill at minute 55 must keep minutes 0-54."""
    clock = FakeClock()
    seen: list[int] = []

    def factory(rc):
        # Read the file the *previous* row wrote, at the moment the next row starts.
        path = tmp_path / "results.json"
        seen.append(len(json.loads(path.read_text(encoding="utf-8"))["rows"]) if path.exists() else -1)
        return bm.StubEngine(rc, sleep=clock.sleep)

    h = harness(tmp_path, chunks_of(4), clock, minutes=0.2, engine_factory=factory)
    matrix = [bm.RunConfig(n, "float16") for n in ("a", "b", "c")]
    h.run_matrix(matrix, results_shell())

    assert seen == [-1, 1, 2]  # row 1 was on disk before row 2 began, and so on
    for name in ("a", "b", "c"):
        chunks_file = tmp_path / f"chunks-{name}-float16.jsonl"
        assert chunks_file.exists() and chunks_file.read_text(encoding="utf-8").strip()


# --------------------------------------------------------------------------
# The CUDA guard (D36)
# --------------------------------------------------------------------------


def test_cuda_guard_refuses_to_time_anything_without_a_gpu():
    with pytest.raises(bm.BenchmarkAborted) as excinfo:
        bm.require_cuda("cuda", count_fn=lambda: 0)
    message = str(excinfo.value)
    assert "D36" in message                      # names the decision, not just the fact
    assert "No timing has been taken" in message


def test_cuda_guard_reports_an_import_failure_as_the_dll_trap():
    def boom() -> int:
        raise ImportError("DLL load failed while importing translator")

    with pytest.raises(bm.BenchmarkAborted) as excinfo:
        bm.require_cuda("cuda", count_fn=boom)
    assert "D41" in str(excinfo.value) and "doctor" in str(excinfo.value)


def test_cuda_guard_allows_an_explicit_cpu_dry_run():
    assert bm.require_cuda("cpu", count_fn=lambda: 0) == 0


def test_a_cpu_run_is_stamped_as_not_a_measurement(tmp_path):
    clock = FakeClock()
    rc = bm.RunConfig("tiny", "int8")
    h = harness(tmp_path, chunks_of(4), clock, minutes=0.2,
                engine_factory=lambda _rc: bm.StubEngine(_rc, sleep=clock.sleep))
    row = h.run_one(rc)
    assert row["not_a_gpu_measurement"] is True
    markdown = bm.render_markdown(results_shell() | {"rows": [row]})
    assert "NOT A GPU MEASUREMENT" in markdown


# --------------------------------------------------------------------------
# The faster-whisper call path, without weights
# --------------------------------------------------------------------------


class FakeSegment:
    def __init__(self, text: str) -> None:
        self.text = text


class FakeWhisperModel:
    """Stands in for `WhisperModel`, and returns a *lazy* generator like it does."""

    def __init__(self, model_size, device, compute_type, download_root):
        self.init_args = (model_size, device, compute_type, download_root)
        self.kwargs = None
        self.consumed = False

    def transcribe(self, audio, **kwargs):
        self.kwargs = kwargs
        self.audio = audio

        def segments():
            self.consumed = True
            yield FakeSegment("hola ")
            yield FakeSegment("qué tal")

        return segments(), object()


def _engine_with_fake_model(tmp_path):
    made: list[FakeWhisperModel] = []

    def factory(*args, **kwargs):
        model = FakeWhisperModel(args[0], kwargs["device"], kwargs["compute_type"],
                                 kwargs["download_root"])
        made.append(model)
        return model

    engine = bm.FasterWhisperEngine(
        bm.RunConfig("small", "float16"), device="cuda", language="es",
        cache_dir=tmp_path / "models", model_factory=factory,
    )
    return engine, made


def test_the_lazy_segments_generator_is_consumed_inside_the_timed_region(tmp_path):
    """`transcribe()` decodes nothing until the generator is drained. Timing the
    call without draining it reports an RTF near zero and means it."""
    engine, made = _engine_with_fake_model(tmp_path)
    engine.load()
    text = engine.transcribe(b"\x00\x01" * 16000)

    assert made[0].consumed is True
    assert text == "hola qué tal"


def test_the_benchmarked_configuration_is_the_one_the_app_will_use(tmp_path):
    """BENCHMARK.md step 4 — anything else measures software we are not shipping."""
    engine, made = _engine_with_fake_model(tmp_path)
    engine.load()
    engine.transcribe(b"\x00\x01" * 16000)

    assert made[0].kwargs == {
        "language": "es",
        "beam_size": 1,
        "condition_on_previous_text": False,
        "vad_filter": False,
    }
    assert made[0].init_args[1:3] == ("cuda", "float16")
    assert made[0].init_args[3] == str(tmp_path / "models")  # the pinned cache (D41)


def test_pcm_reaches_the_model_as_normalised_float32(tmp_path):
    engine, made = _engine_with_fake_model(tmp_path)
    engine.load()
    engine.transcribe(b"\x00\x80" + b"\xff\x7f")  # -32768, +32767
    audio = made[0].audio
    assert audio.dtype.name == "float32"
    assert audio[0] == pytest.approx(-1.0)
    assert audio[1] == pytest.approx(1.0, abs=1e-4)


# --------------------------------------------------------------------------
# The table
# --------------------------------------------------------------------------


def test_the_rendered_table_uses_benchmark_mds_own_columns(tmp_path):
    """The table is transcribed, not retyped — so its header has to be the one
    already in the document. This test fails if either side drifts."""
    doc = (cfg.REPO_ROOT / "docs" / "BENCHMARK.md").read_text(encoding="utf-8")
    assert bm.MARKDOWN_HEADER in doc


def test_rendered_table_labels_the_word_age_column_predicted(tmp_path):
    clock = FakeClock()
    h = harness(tmp_path, chunks_of(4), clock, minutes=0.2,
                engine_factory=lambda rc: bm.StubEngine(rc, sleep=clock.sleep))
    results = h.run_matrix([bm.RunConfig("small", "float16")], results_shell())
    markdown = (tmp_path / "results.md").read_text(encoding="utf-8")

    assert "p95 word age *(predicted)*" in markdown
    assert "predicted" in markdown.lower()
    assert "Level 7 measures it end to end" in markdown
    # It reports; it does not choose (D25/D26).
    assert "does not select a model" in markdown
    assert results["rows"][0]["status"] == "ok"


def test_a_failed_row_renders_as_a_failed_row(tmp_path):
    row = {
        "model_size": "large-v3", "compute_type": "float16", "status": "failed",
        "failure": "oom", "error": "CUDA out of memory", "peak_vram_mb": 3980.0,
        "gate_vram": False, "mt_source": "measured",
    }
    markdown = bm.render_markdown(results_shell() | {"rows": [row]})
    assert "FAILED (oom)" in markdown
    assert "CUDA out of memory" in markdown


def test_render_mode_rebuilds_the_table_from_a_results_file(tmp_path, capsys):
    path = tmp_path / "results.json"
    path.write_text(json.dumps(results_shell()), encoding="utf-8")
    assert bm.main(["--render", str(path)]) == 0
    assert bm.MARKDOWN_HEADER in capsys.readouterr().out


def test_non_ascii_survives_the_writers(tmp_path):
    """D41/D42: the whole point of this application's output is non-ASCII."""
    clock = FakeClock()
    engine = bm.StubEngine(bm.RunConfig("small", "float16"), sleep=clock.sleep,
                           text="¿Me escuchan bien? — señor, こんにちは")
    h = harness(tmp_path, chunks_of(2), clock, minutes=0.05,
                engine_factory=lambda _rc: engine)
    h.run_matrix([bm.RunConfig("small", "float16")], results_shell())

    payload = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))
    sample = payload["rows"][0]["samples"][0]["text"]
    assert "señor" in sample and "こんにちは" in sample


# --------------------------------------------------------------------------
# GPU polling — the parsing is testable here; the device is not
# --------------------------------------------------------------------------


def test_gpu_poller_parses_nvidia_smi_output():
    def runner(command):
        if "--query-compute-apps=pid,used_memory" in command[1]:
            return f"{__import__('os').getpid()}, 1580\n"
        if command[1].startswith("--query-gpu=memory.used"):
            return "2480, 4096, 71, 1350, 42.10\n"
        return "NVIDIA GeForce GTX 1650 Ti, 566.03, 4096\n"

    poller = bm.GpuPoller(runner=runner)
    sample = poller.sample()
    assert sample["memory_used_mb"] == 2480.0
    assert sample["memory_total_mb"] == 4096.0
    assert sample["temperature_c"] == 71.0
    assert sample["sm_clock_mhz"] == 1350.0
    assert sample["own_memory_mb"] == 1580.0        # our slice, not the browser's
    assert poller.describe()["name"].startswith("NVIDIA")


def test_no_nvidia_smi_gives_null_columns_rather_than_a_crash(tmp_path):
    """The Linux dev box. The VRAM gate is then `unknown`, which is honest."""
    poller = bm.GpuPoller(runner=lambda _cmd: None)
    assert poller.sample() is None

    clock = FakeClock()
    h = harness(tmp_path, chunks_of(4), clock, minutes=0.1,
                engine_factory=lambda rc: bm.StubEngine(rc, sleep=clock.sleep))
    h.poller = poller
    row = h.run_one(bm.RunConfig("small", "float16"))

    assert row["peak_vram_mb"] is None
    assert row["gate_vram"] is None
    assert row["status"] == "ok"


def test_vram_gate_uses_the_whole_card_with_a_headroom_margin(tmp_path):
    """Gate 2's budget is not 4 GB — the desktop and the browser are already in
    it, which is why the benchmark is run with a browser open (D18)."""
    clock = FakeClock()
    h = harness(tmp_path, chunks_of(4), clock, minutes=0.1,
                engine_factory=lambda rc: bm.StubEngine(rc, sleep=clock.sleep))
    row = {"status": "ok", "rtf": 0.3, "rtf_last_min": 0.3, "rtf_p95_chunk": 0.3,
           "peak_vram_mb": 3900.0, "total_vram_mb": 4096.0}
    gates = h._gate_metrics(row, [])
    assert gates["gate_vram"] is False          # 3900 > 4096 x 0.9

    row["peak_vram_mb"] = 3000.0
    assert h._gate_metrics(row, [])["gate_vram"] is True


# --------------------------------------------------------------------------
# CLI wiring
# --------------------------------------------------------------------------


def test_matrix_parsing():
    assert bm.parse_matrix("small:float16, medium:int8_float16") == [
        bm.RunConfig("small", "float16"),
        bm.RunConfig("medium", "int8_float16"),
    ]
    with pytest.raises(ValueError):
        bm.parse_matrix("small")
    with pytest.raises(ValueError):
        bm.parse_matrix("")


def test_the_default_matrix_is_benchmark_mds_seven_rows():
    doc = (cfg.REPO_ROOT / "docs" / "BENCHMARK.md").read_text(encoding="utf-8")
    assert len(bm.DEFAULT_MATRIX) == 7
    for run_config in bm.DEFAULT_MATRIX:
        assert f"| {run_config.model_size} | {run_config.compute_type} |" in doc
    # D19: English-only models cannot serve a source language the user does not speak.
    assert not any("distil" in rc.model_size for rc in bm.DEFAULT_MATRIX)


def test_missing_input_wav_explains_where_it_comes_from(capsys):
    assert bm.main(["--dry-run"]) == 2
    assert "record_loopback" in capsys.readouterr().err


def test_dry_run_forces_the_stub_and_cpu():
    args = bm.build_parser().parse_args(["--dry-run", "--input-wav", "x.wav"])
    assert args.dry_run is True
    # main() applies the overrides; assert the defaults it starts from.
    assert args.engine == "faster-whisper" and args.device == "cuda"


def test_the_harness_never_writes_the_model_choice_into_config():
    """Level 3 is a measurement; writing its result into config is a human step,
    and Level 4's problem. The harness must not have the seam at all."""
    source = (cfg.PACKAGE_DIR / "tools" / "benchmark.py").read_text(encoding="utf-8")
    assert "model_size=" not in source.replace("model_size=run_config.model_size", "")
    assert "ST_MODEL_SIZE" not in source
    assert "ST_COMPUTE_TYPE" not in source
