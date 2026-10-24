import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class CaptureWindow:
    window_id: int
    owner: str
    title: str
    bounds: Tuple[int, int, int, int]

    @property
    def label(self) -> str:
        title = self.title or "(untitled)"
        return f"{self.owner} - {title}"


class WindowCapture:
    """Captures one selected application window instead of the whole desktop."""

    def __init__(self, selected_window: CaptureWindow):
        self.selected_window = selected_window

        if sys.platform == "win32":
            try:
                import mss
            except ImportError as exc:
                raise RuntimeError(
                    "Install Windows screen capture support with "
                    "'python -m pip install mss'."
                ) from exc

            self._mss = mss.mss()
            return

        if sys.platform != "darwin":
            raise RuntimeError(
                "Window capture is supported on Windows and macOS only."
            )

        try:
            import Quartz
        except ImportError as exc:
            raise RuntimeError(
                "Install macOS window capture support with "
                "'pip install pyobjc-framework-Quartz'."
            ) from exc

        self._quartz = Quartz

    @staticmethod
    def list_windows() -> List[CaptureWindow]:
        if sys.platform == "win32":
            return [
                CaptureWindow(
                    window_id=0,
                    owner="Windows",
                    title="Primary monitor",
                    bounds=(0, 0, 0, 0),
                )
            ]

        if sys.platform != "darwin":
            return []

        try:
            import Quartz
        except ImportError:
            return []

        window_info = Quartz.CGWindowListCopyWindowInfo(
            Quartz.kCGWindowListOptionAll
            | Quartz.kCGWindowListExcludeDesktopElements,
            Quartz.kCGNullWindowID,
        )
        windows = []

        for info in window_info or []:
            owner = str(info.get(Quartz.kCGWindowOwnerName, "") or "")
            title = str(info.get(Quartz.kCGWindowName, "") or "")
            window_id = int(info.get(Quartz.kCGWindowNumber, 0) or 0)
            bounds = info.get(Quartz.kCGWindowBounds, {})

            if (
                not owner
                or not window_id
                or int(info.get(Quartz.kCGWindowLayer, 0) or 0) != 0
                or bounds.get("Width", 0) < 100
                or bounds.get("Height", 0) < 100
            ):
                continue

            windows.append(
                CaptureWindow(
                    window_id=window_id,
                    owner=owner,
                    title=title,
                    bounds=(
                        int(bounds.get("X", 0)),
                        int(bounds.get("Y", 0)),
                        int(bounds.get("Width", 0)),
                        int(bounds.get("Height", 0)),
                    ),
                )
            )

        return windows

    @classmethod
    def select(cls) -> "WindowCapture":
        windows = cls.list_windows()
        if not windows:
            raise RuntimeError(
                "No visible windows were found. Open Google Meet and try again."
            )

        if sys.platform == "win32":
            return cls(windows[0])

        print("\nSelect the Google Meet window to analyze:")
        for index, window in enumerate(windows, start=1):
            print(f"  {index}. {window.label}")

        while True:
            choice = input("Window number (or 'q' to quit): ").strip()
            if choice.lower() == "q":
                raise KeyboardInterrupt
            try:
                selected = windows[int(choice) - 1]
            except (ValueError, IndexError):
                print("Please enter one of the listed window numbers.")
                continue
            return cls(selected)

    def read(self) -> Optional[np.ndarray]:
        if sys.platform == "win32":
            monitor = self._mss.monitors[1]
            screenshot = self._mss.grab(monitor)
            frame = np.asarray(screenshot)
            return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        image = self._quartz.CGWindowListCreateImage(
            self._quartz.CGRectNull,
            self._quartz.kCGWindowListOptionIncludingWindow,
            self.selected_window.window_id,
            self._quartz.kCGWindowImageBoundsIgnoreFraming
            | self._quartz.kCGWindowImageNominalResolution,
        )
        if image is None:
            return None

        width = self._quartz.CGImageGetWidth(image)
        height = self._quartz.CGImageGetHeight(image)
        bytes_per_row = self._quartz.CGImageGetBytesPerRow(image)
        provider = self._quartz.CGImageGetDataProvider(image)
        data = self._quartz.CGDataProviderCopyData(provider)

        frame = np.frombuffer(data, dtype=np.uint8)
        frame = frame.reshape((height, bytes_per_row // 4, 4))[:, :width]
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
