#!/usr/bin/env python3
"""Set the X11 clipboard, paste into the current focus, then exit."""

from __future__ import annotations

import ctypes
import sys
import tkinter as tk


def send_ctrl_v() -> None:
    x11 = ctypes.CDLL("libX11.so.6")
    xtst = ctypes.CDLL("libXtst.so.6")
    display_type = ctypes.c_void_p
    x11.XOpenDisplay.argtypes = [ctypes.c_char_p]
    x11.XOpenDisplay.restype = display_type
    x11.XCloseDisplay.argtypes = [display_type]
    x11.XStringToKeysym.argtypes = [ctypes.c_char_p]
    x11.XStringToKeysym.restype = ctypes.c_ulong
    x11.XKeysymToKeycode.argtypes = [display_type, ctypes.c_ulong]
    x11.XKeysymToKeycode.restype = ctypes.c_ubyte
    x11.XFlush.argtypes = [display_type]
    xtst.XTestFakeKeyEvent.argtypes = [display_type, ctypes.c_uint, ctypes.c_int, ctypes.c_ulong]
    xtst.XTestFakeKeyEvent.restype = ctypes.c_int

    display = x11.XOpenDisplay(None)
    if not display:
        raise RuntimeError("could not open the X11 display")
    try:
        control = x11.XKeysymToKeycode(
            display, x11.XStringToKeysym(b"Control_L")
        )
        letter_v = x11.XKeysymToKeycode(display, x11.XStringToKeysym(b"v"))
        if not control or not letter_v:
            raise RuntimeError("could not resolve Ctrl+V keycodes")
        xtst.XTestFakeKeyEvent(display, control, 1, 0)
        xtst.XTestFakeKeyEvent(display, letter_v, 1, 0)
        xtst.XTestFakeKeyEvent(display, letter_v, 0, 0)
        xtst.XTestFakeKeyEvent(display, control, 0, 0)
        x11.XFlush(display)
    finally:
        x11.XCloseDisplay(display)


def main() -> int:
    text = sys.stdin.read()
    if not text:
        raise SystemExit("no text supplied")
    root = tk.Tk()
    root.withdraw()
    error: list[BaseException] = []
    root.clipboard_clear()
    root.clipboard_append(text)
    root.update()

    def paste() -> None:
        try:
            send_ctrl_v()
        except BaseException as exc:  # Tk callbacks do not propagate exceptions.
            error.append(exc)

    root.after(100, paste)
    root.after(850, root.destroy)
    root.mainloop()
    if error:
        raise RuntimeError(str(error[0]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

