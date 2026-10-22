"""Public packages must exclude local state and reject source credential leaks."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
import zipfile


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/package_release.py"
SPEC = importlib.util.spec_from_file_location("package_release", SCRIPT)
release = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(release)


class SourceReleaseTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        for name in ("README.md", "README.zh-CN.md", "LICENSE", "static/index.html"):
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("Public source\n")

    def test_runtime_state_is_not_traversed_and_manifest_matches_public_source(self):
        private = self.root / ".local"
        private.mkdir()
        (private / "token").write_text("private-runtime-token-for-this-test")
        (private / "voice.key").write_text("private key material")
        (self.root / "static/__pycache__").mkdir()
        (self.root / "static/__pycache__/cache.py").write_text("private cache")
        destination = self.root / "dist/source.zip"
        result = release.build_archive(self.root, destination)
        with zipfile.ZipFile(destination) as archive:
            self.assertEqual(len(archive.namelist()), 5)
            self.assertTrue(all(name.startswith("lan-voice/") for name in archive.namelist()))
            self.assertFalse(any(".local" in name or "__pycache__" in name for name in archive.namelist()))
            self.assertIn("lan-voice/SOURCE_MANIFEST.json", archive.namelist())
        self.assertEqual(result["files"], 4)
        self.assertTrue(destination.with_suffix(".zip.sha256").is_file())

    def test_runtime_token_accidentally_copied_into_readme_is_rejected(self):
        secret = "private-runtime-token-for-this-test"
        (self.root / ".local").mkdir()
        (self.root / ".local/token").write_text(secret)
        (self.root / "README.md").write_text("An accidentally pasted URL contains " + secret)
        with self.assertRaisesRegex(ValueError, "Runtime credential"):
            release.build_archive(self.root, self.root / "dist/source.zip")
        self.assertFalse((self.root / "dist/source.zip").exists())

    def test_public_symlink_cannot_include_an_external_file(self):
        (self.root / "static/link.txt").symlink_to(self.root / "README.md")
        with self.assertRaisesRegex(ValueError, "symlink"):
            release.build_archive(self.root, self.root / "dist/source.zip")


if __name__ == "__main__":
    unittest.main()
