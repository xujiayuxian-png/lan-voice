"""Opt-in paste test inside a hidden, independent Xephyr display."""

import os
from pathlib import Path
import select
import shutil
import subprocess
import sys
import tempfile
import time
import unittest

try:
    import tkinter as tk
except ImportError:
    tk = None


ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(
    os.environ.get("VOICE_TEST_X11") == "1"
    and os.environ.get("DISPLAY")
    and shutil.which("Xephyr")
    and tk,
    "Set VOICE_TEST_X11=1 with Xephyr, tkinter and an X11 display",
)
class ClipboardTest(unittest.TestCase):
    def test_unicode_multiline_paste_in_hidden_independent_display(self):
        host = tk.Tk()
        host.withdraw()
        host.geometry("800x600")
        host.update_idletasks()
        self.addCleanup(host.destroy)
        self.assertFalse(host.winfo_ismapped())
        read_fd, write_fd = os.pipe()
        nested = None
        xephyr = None
        helper = None
        with tempfile.TemporaryFile() as log:
            try:
                xephyr = subprocess.Popen(
                    [
                        "Xephyr", "-parent", str(host.winfo_id()), "-no-host-grab",
                        "-displayfd", str(write_fd), "-screen", "800x600",
                        "-nolisten", "tcp", "-noreset", "-ac",
                    ],
                    pass_fds=(write_fd,),
                    stdout=log,
                    stderr=log,
                )
                os.close(write_fd)
                write_fd = None
                self.assertTrue(
                    select.select([read_fd], [], [], 5)[0],
                    "Hidden Xephyr did not become ready",
                )
                number = os.read(read_fd, 64).decode().strip()
                self.assertTrue(number.isdecimal())
                display = ":" + number
                desktop_number = os.environ["DISPLAY"].rsplit(":", 1)[-1].split(".")[0]
                self.assertNotEqual(number, desktop_number)

                nested = tk.Tk(screenName=display)
                target = tk.Text(nested)
                target.pack()
                nested.update()
                target.focus_force()
                nested.update()
                self.assertEqual(nested.focus_get(), target)
                self.assertFalse(host.winfo_ismapped())

                message = "独立显示测试\n第二行中文和 emoji 🐱"
                helper = subprocess.Popen(
                    [sys.executable, str(ROOT / "clipboard_paste.py")],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    text=True,
                    env={**os.environ, "DISPLAY": display},
                )
                helper.stdin.write(message)
                helper.stdin.close()
                deadline = time.monotonic() + 5
                while helper.poll() is None and time.monotonic() < deadline:
                    nested.update()
                    time.sleep(0.01)
                self.assertIsNotNone(helper.poll(), "Paste helper timed out")
                nested.update()
                self.assertEqual(helper.returncode, 0, helper.stderr.read())
                self.assertEqual(target.get("1.0", "end-1c"), message)
                self.assertFalse(host.winfo_ismapped())
            finally:
                if helper:
                    if helper.poll() is None:
                        helper.kill()
                        helper.wait(timeout=3)
                    helper.stdin.close()
                    helper.stderr.close()
                if nested:
                    nested.destroy()
                if xephyr:
                    xephyr.terminate()
                    try:
                        xephyr.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        xephyr.kill()
                        xephyr.wait(timeout=3)
                os.close(read_fd)
                if write_fd is not None:
                    os.close(write_fd)


if __name__ == "__main__":
    unittest.main()
