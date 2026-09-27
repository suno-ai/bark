import sys
import types
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _build_parser():
    try:
        from bark.cli import build_parser
        return build_parser
    except Exception:
        generation = types.ModuleType("bark.generation")
        generation.SAMPLE_RATE = 24000
        generation.preload_models = lambda *args, **kwargs: None
        api = types.ModuleType("bark.api")
        for name in (
            "generate_audio",
            "text_to_semantic",
            "semantic_to_waveform",
            "save_as_prompt",
        ):
            setattr(api, name, lambda *args, **kwargs: None)
        sys.modules["bark.generation"] = generation
        sys.modules["bark.api"] = api
        if "scipy.io.wavfile" not in sys.modules:
            wavfile = types.ModuleType("scipy.io.wavfile")
            wavfile.write = lambda *args, **kwargs: None
            sys.modules.setdefault("scipy", types.ModuleType("scipy"))
            sys.modules.setdefault("scipy.io", types.ModuleType("scipy.io"))
            sys.modules["scipy.io.wavfile"] = wavfile
        for key in list(sys.modules):
            if key == "bark" or (key.startswith("bark.") and key not in ("bark.generation", "bark.api")):
                del sys.modules[key]
        from bark.cli import build_parser
        return build_parser


build_parser = _build_parser()


class CliBoolTest(unittest.TestCase):
    def test_false_strings_stay_false(self):
        args = build_parser().parse_args(
            ["--silent", "false", "--output_full", "false", "--text", "hi"]
        )
        self.assertIs(args.silent, False)
        self.assertIs(args.output_full, False)

    def test_true_and_default(self):
        self.assertIs(build_parser().parse_args(["--silent", "true"]).silent, True)
        self.assertIs(build_parser().parse_args(["--text", "hi"]).silent, False)
        self.assertIs(build_parser().parse_args(["--text", "hi"]).output_full, False)


if __name__ == "__main__":
    unittest.main()
