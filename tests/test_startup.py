"""Exercise the launcher without loading a model or starting a server."""

import importlib.util
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import os
from pathlib import Path
import shutil
import ssl
import stat
import subprocess
import sys
import tempfile
import tarfile
import threading
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
BASH = shutil.which("bash")
OPENSSL = shutil.which("openssl")


@unittest.skipUnless(BASH and OPENSSL, "bash and openssl are required")
class StartupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="voice startup ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        shutil.copy2(ROOT / "run.sh", self.root / "run.sh")
        (self.root / "server.py").write_text(
            "import json, os, sys\n"
            "print(json.dumps({'args': sys.argv[1:], 'token': os.getenv('VOICE_TOKEN')}))\n"
        )
        self.model = self.root / "model files"
        self.model.mkdir()
        (self.model / "model.int8.onnx").touch()
        (self.model / "tokens.txt").touch()
        self.bin = self.root / "bin"
        self.bin.mkdir()
        for tool in ("dirname", "mkdir", "chmod", "mktemp", "mv", "rm", "cat"):
            (self.bin / tool).symlink_to(shutil.which(tool))
        (self.bin / "openssl").symlink_to(OPENSSL)
        (self.bin / "ffmpeg").symlink_to("/bin/true")
        self.env = os.environ.copy()
        for key in list(self.env):
            if key.startswith("VOICE_"):
                self.env.pop(key)
        self.env.update(
            PATH=str(self.bin),
            VOICE_PYTHON=sys.executable,
            VOICE_MODEL_DIR=str(self.model),
        )

    def run_launcher(self, *args):
        return subprocess.run(
            [BASH, str(self.root / "run.sh"), *args],
            cwd="/",
            env=self.env,
            text=True,
            capture_output=True,
            timeout=20,
        )

    def successful_launch(self, *args):
        result = self.run_launcher(*args)
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    def test_python_path_with_spaces_starts_and_preserves_arguments(self):
        interpreter = self.root / "python with spaces"
        interpreter.symlink_to(sys.executable)
        self.env["VOICE_PYTHON"] = str(interpreter)

        result = self.successful_launch("--dry-run")

        self.assertIn("--dry-run", result["args"])
        self.assertIn(str(self.model / "model.int8.onnx"), result["args"])

    def test_project_local_runtime_and_model_are_the_defaults(self):
        python = self.root / ".local/envs/speech/bin/python"
        python.parent.mkdir(parents=True)
        python.symlink_to(sys.executable)
        model = self.root / ".local/models/sensevoice"
        model.mkdir(parents=True)
        (model / "model.int8.onnx").touch()
        (model / "tokens.txt").touch()
        self.env.pop("VOICE_PYTHON")
        self.env.pop("VOICE_MODEL_DIR")
        result = self.successful_launch()
        self.assertIn(str(model / "model.int8.onnx"), result["args"])
        self.assertIn(str(self.root / ".local/models/qwen3-asr-0.6b"), result["args"])

    def test_early_launcher_errors_follow_environment_or_cli_language(self):
        (self.bin / "ffmpeg").unlink()
        self.env["VOICE_UI_LANGUAGE"] = "zh-CN"
        result = self.run_launcher()
        self.assertIn("缺少必要命令", result.stderr)
        result = self.run_launcher("--ui-language", "en")
        self.assertIn("Required command not found", result.stderr)
        result = self.run_launcher("--ui-language=zh-CN")
        self.assertIn("缺少必要命令", result.stderr)

    def test_streaming_wrapper_has_empty_optional_context_by_default(self):
        shutil.copy2(ROOT / "run_streaming.sh", self.root / "run_streaming.sh")
        (self.bin / "bash").symlink_to(BASH)
        (self.root / "run.sh").write_text(
            '#!/usr/bin/env bash\nexec "$VOICE_PYTHON" -c '\
            "'import json,os; print(json.dumps({\"context\":os.environ.get(\"VOICE_QWEN_CONTEXT\")}))'\n"
        )
        for configured, expected in ((None, ""), ("", ""), ("Meeting notes and product names", "Meeting notes and product names")):
            with self.subTest(context=configured):
                if configured is None:
                    self.env.pop("VOICE_QWEN_CONTEXT", None)
                else:
                    self.env["VOICE_QWEN_CONTEXT"] = configured
                result = subprocess.run([BASH, str(self.root / "run_streaming.sh")], env=self.env, capture_output=True, text=True, timeout=10)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(json.loads(result.stdout)["context"], expected)

    def test_native_backend_keeps_url_credentials_and_uses_official_model_path(self):
        original_token = self.successful_launch()["token"]
        native_model = self.root / "native model files"
        native_model.mkdir()
        for name in ("config.json", "model.safetensors.index.json"):
            (native_model / name).touch()
        self.env.update(
            VOICE_ASR_BACKEND="qwen3-streaming", VOICE_QWEN_NATIVE_MODEL_DIR=str(native_model)
        )
        result = self.successful_launch()
        self.assertEqual(result["token"], original_token)
        args = result["args"]
        self.assertEqual(args[args.index("--asr-backend") + 1], "qwen3-streaming")
        self.assertEqual(args[args.index("--qwen-native-model-dir") + 1], str(native_model))

    def test_missing_runtime_tool_fails_before_server_starts(self):
        for tool in ("ffmpeg", "openssl"):
            with self.subTest(tool=tool):
                (self.bin / tool).unlink()
                try:
                    result = self.run_launcher()
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn(tool, result.stderr)
                    self.assertEqual(result.stdout, "")
                finally:
                    (self.bin / tool).symlink_to(
                        OPENSSL if tool == "openssl" else "/bin/true"
                    )

    def test_private_token_is_stable_and_not_passed_as_argument(self):
        first = self.successful_launch()
        token_file = self.root / ".local" / "token"
        self.assertTrue(token_file.is_file())
        token = token_file.read_text().strip()
        self.assertGreaterEqual(len(token), 24)
        self.assertEqual(first["token"], token)
        self.assertNotIn(token, first["args"])
        self.assertEqual(self.successful_launch()["token"], token)
        for path in (
            token_file,
            self.root / ".local" / "tls" / "voice-input.key",
            self.root / ".local" / "tls" / "voice-input.crt",
        ):
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
        self.assertEqual(stat.S_IMODE(token_file.parent.stat().st_mode), 0o700)

    def test_environment_token_overrides_saved_token(self):
        self.successful_launch()
        token_file = self.root / ".local" / "token"
        self.assertTrue(token_file.is_file())
        original = token_file.read_text()
        self.env["VOICE_TOKEN"] = "explicit-private-test-token"

        result = self.successful_launch()

        self.assertEqual(result["token"], "explicit-private-test-token")
        self.assertNotIn("explicit-private-test-token", result["args"])
        self.assertEqual(token_file.read_text(), original)

    def test_expired_certificate_is_replaced(self):
        self.successful_launch()
        cert = self.root / ".local" / "tls" / "voice-input.crt"
        key = cert.with_suffix(".key")
        subprocess.run(
            [OPENSSL, "x509", "-in", str(cert), "-signkey", str(key),
             "-days", "-1", "-out", str(cert)],
            check=True,
            capture_output=True,
        )
        expired = cert.read_bytes()
        check = subprocess.run(
            [OPENSSL, "x509", "-checkend", "0", "-noout", "-in", str(cert)],
            capture_output=True,
        )
        self.assertNotEqual(check.returncode, 0)

        self.successful_launch()

        self.assertTrue(cert.read_bytes() != expired, "Expired certificate was reused")
        check = subprocess.run(
            [OPENSSL, "x509", "-checkend", "0", "-noout", "-in", str(cert)],
            capture_output=True,
        )
        self.assertEqual(check.returncode, 0)

    def test_certificate_generation_failure_does_not_leave_partial_credentials(self):
        openssl = self.bin / "openssl"
        openssl.unlink()
        openssl.write_text(
            f"#!{sys.executable}\n"
            "import pathlib, sys\n"
            "args = sys.argv[1:]\n"
            "if args[0] == 'req':\n"
            "    pathlib.Path(args[args.index('-keyout') + 1]).write_text('partial key')\n"
            "sys.exit(1)\n"
        )
        openssl.chmod(0o755)

        result = self.run_launcher()

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("certificate", result.stderr.lower())
        tls_dir = self.root / ".local" / "tls"
        self.assertEqual(list(tls_dir.iterdir()), [])

    def test_unusable_certificate_key_pair_is_replaced(self):
        self.successful_launch()
        cert = self.root / ".local" / "tls" / "voice-input.crt"
        key = cert.with_suffix(".key")
        for problem in ("corrupt", "mismatched"):
            with self.subTest(problem=problem):
                previous = cert.read_bytes()
                if problem == "corrupt":
                    key.write_text("not a private key")
                else:
                    subprocess.run(
                        [OPENSSL, "genpkey", "-algorithm", "RSA", "-pkeyopt",
                         "rsa_keygen_bits:2048", "-out", str(key)],
                        check=True,
                        capture_output=True,
                    )

                self.successful_launch()

                self.assertTrue(
                    cert.read_bytes() != previous, "Unusable TLS pair was reused"
                )
                ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER).load_cert_chain(cert, key)


class ModelSetupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("voice_model_setup", ROOT / "scripts/setup.py")
        cls.setup = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.setup)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="voice setup ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def archive(self, members):
        archive = self.root / "tiny-model.tar.bz2"
        with tarfile.open(archive, "w:bz2") as output:
            for name, content, kind in members:
                item = tarfile.TarInfo(name)
                item.type = kind
                if kind == tarfile.REGTYPE:
                    value = content.encode()
                    item.size = len(value)
                    output.addfile(item, io.BytesIO(value))
                else:
                    item.linkname = content
                    output.addfile(item)
        return archive

    def test_cpu_install_verifies_every_file_and_detects_later_corruption(self):
        archive = self.archive([
            ("model/model.int8.onnx", "tiny weights", tarfile.REGTYPE),
            ("model/tokens.txt", "test tokens", tarfile.REGTYPE),
        ])
        destination = self.root / "installed"
        with patch.object(self.setup, "download_archive", return_value=archive) as download:
            self.setup.install_model("sensevoice", destination, self.root / "cache", 1)
            self.setup.install_model("sensevoice", destination, self.root / "cache", 1, verify_only=True)
            download.assert_called_once()
            (destination / "tokens.txt").write_text("changed tokens")
            with self.assertRaisesRegex(ValueError, "failed verification"):
                self.setup.install_model("sensevoice", destination, self.root / "cache", 1, verify_only=True)
        self.assertTrue((destination / "voice-input-model.json").is_file())

    def test_verify_only_missing_model_never_downloads_or_installs(self):
        destination = self.root / "missing"
        with patch.object(self.setup, "urlopen") as network, patch.object(self.setup, "download_archive") as download:
            with self.assertRaises(FileNotFoundError):
                self.setup.install_model("sensevoice", destination, self.root / "cache", 1, verify_only=True)
            network.assert_not_called()
            download.assert_not_called()
        self.assertFalse(destination.exists())
        self.assertFalse((self.root / "cache").exists())

    def test_archive_traversal_and_links_are_rejected_before_extraction(self):
        for name, kind in (("../outside", tarfile.REGTYPE), ("/outside", tarfile.REGTYPE),
                           ("model/link", tarfile.SYMTYPE), ("model/link", tarfile.LNKTYPE)):
            with self.subTest(name=name, kind=kind):
                archive = self.archive([(name, "outside", kind)])
                staging = self.root / "staging"
                staging.mkdir(exist_ok=True)
                with self.assertRaises(ValueError):
                    self.setup.extract_archive(archive, staging, 1024)
                self.assertEqual(list(staging.iterdir()), [])
                self.assertFalse((self.root / "outside").exists())

    def test_download_rejects_bad_hash_and_removes_temporary_file(self):
        spec = {"name": "tiny.tar.bz2", "release": "models", "size": 4, "sha256": "0" * 64}
        cache = self.root / "cache"
        with patch.object(self.setup, "urlopen", return_value=io.BytesIO(b"test")):
            with self.assertRaisesRegex(ValueError, "SHA256"):
                self.setup.download_archive(spec, cache, 1)
        self.assertEqual(list(cache.iterdir()), [])

    def test_missing_required_file_never_publishes_partial_model(self):
        archive = self.archive([("model/readme.txt", "not weights", tarfile.REGTYPE)])
        destination = self.root / "incomplete"
        with patch.object(self.setup, "download_archive", return_value=archive):
            with self.assertRaisesRegex(ValueError, "missing model.int8.onnx"):
                self.setup.install_model("sensevoice", destination, self.root / "cache", 1)
        self.assertFalse(destination.exists())
        self.assertEqual(list(self.root.glob(".model-extract-*")), [])

    def test_environment_setup_is_separate_from_model_download(self):
        destination = self.root / "environment"
        with patch.object(self.setup.subprocess, "run") as run, patch.object(self.setup, "urlopen") as network:
            self.setup.setup_environment("cpu", destination)
        network.assert_not_called()
        self.assertEqual(run.call_count, 3)
        self.assertEqual(run.call_args_list[0].args[0], [sys.executable, "-m", "venv", str(destination)])
        self.assertEqual(run.call_args.args[0][-1], str(ROOT / "requirements.txt"))
        self.assertNotIn("PYTHONPATH", run.call_args.kwargs["env"])
        self.assertNotIn("PYTHONHOME", run.call_args.kwargs["env"])

    def test_native_model_uses_existing_verified_downloader_with_offline_flag(self):
        destination = self.root / "native"
        with patch.object(self.setup.subprocess, "run") as run:
            result = self.setup.main(["models", "--model", "qwen3-native", "--destination", str(destination), "--verify-only"])
        self.assertEqual(result, 0)
        command = run.call_args.args[0]
        self.assertIn(str(ROOT / "scripts/download_qwen_asr.py"), command)
        self.assertIn("--verify-only", command)
        self.assertIn(str(destination), command)


@unittest.skipUnless(BASH, "bash is required")
class CorrectionLauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="correction launcher ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        shutil.copy2(ROOT / "run_with_correction.sh", self.root / "run_with_correction.sh")
        python = self.root / "test-python"
        python.write_text(
            f"#!{sys.executable}\nimport os,sys\n"
            "if sys.argv[1:] == ['-c', 'import pypinyin']: raise SystemExit(0)\n"
            f"os.execv({sys.executable!r}, [{sys.executable!r}, *sys.argv[1:]])\n"
        )
        python.chmod(0o755)
        self.calls = self.root / "lms-calls.jsonl"
        lms = self.root / "lms"
        lms.write_text(
            f"#!{sys.executable}\nimport json,sys\n"
            f"with open({str(self.calls)!r}, 'a') as output: output.write(json.dumps(sys.argv[1:]) + '\\n')\n"
        )
        lms.chmod(0o755)
        (self.root / "run.sh").write_text(
            '#!/usr/bin/env bash\nexec "$VOICE_PYTHON" -c '
            "'import json,os; print(json.dumps({\"model\":os.environ[\"VOICE_CORRECTION_MODEL\"]}))'\n"
        )
        self.models = [{"id": "test-correction-model"}]
        launcher = self

        class ModelsHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                payload = json.dumps({"data": launcher.models}).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args):
                pass

        self.api = ThreadingHTTPServer(("127.0.0.1", 0), ModelsHandler)
        self.thread = threading.Thread(target=self.api.serve_forever, daemon=True)
        self.thread.start()
        self.env = {key: value for key, value in os.environ.items() if not key.startswith("VOICE_")}
        self.env.update(
            VOICE_PYTHON=str(python), VOICE_LMS=str(lms),
            VOICE_CORRECTION_MODEL="test-correction-model",
            VOICE_CORRECTION_URL=f"http://127.0.0.1:{self.api.server_port}/v1",
        )

    def tearDown(self):
        self.api.shutdown()
        self.api.server_close()
        self.thread.join(timeout=2)

    def run_launcher(self):
        return subprocess.run([BASH, str(self.root / "run_with_correction.sh")], env=self.env, capture_output=True, text=True, timeout=10)

    def test_existing_correction_model_is_reused_without_loading_other_models(self):
        result = self.run_launcher()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["model"], "test-correction-model")
        self.assertFalse(self.calls.exists())

    def test_only_explicit_model_is_loaded_without_unloading_other_models(self):
        self.models = [{"id": "another-application-model"}]
        self.env["VOICE_LM_PATH"] = "example-model-path"
        result = self.run_launcher()
        self.assertEqual(result.returncode, 0, result.stderr)
        commands = [json.loads(line) for line in self.calls.read_text().splitlines()]
        self.assertEqual(commands, [
            ["server", "start", "--bind", "127.0.0.1", "--port", str(self.api.server_port)],
            ["load", "example-model-path", "--context-length", "8192", "--identifier", "test-correction-model", "-y"],
        ])

    def test_unknown_model_requires_explicit_configuration(self):
        self.models = []
        result = self.run_launcher()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("VOICE_LM_PATH", result.stderr)
        self.assertFalse(self.calls.exists())


if __name__ == "__main__":
    unittest.main()
