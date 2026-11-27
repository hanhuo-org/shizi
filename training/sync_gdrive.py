"""Async checkpoint sync to Google Drive via rclone (adapted from jiyi/sync.py).

Uploads run in background subprocesses so the DDP rank-0 training loop never blocks.
Also provides pull/resume helpers so multi-session Kaggle 12h runs pick up seamlessly
from the latest checkpoint stored on Google Drive (`wylggww:shizi_v12/...`).
"""
from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path


def ensure_rclone_conf(conf_text: str | None = None) -> Path:
    """Write rclone.conf if conf_text (or RCLONE_CONF_TEXT env) is provided; return config path."""
    conf_dir = Path(os.path.expanduser("~/.config/rclone"))
    conf_dir.mkdir(parents=True, exist_ok=True)
    conf_path = conf_dir / "rclone.conf"
    text = conf_text or os.environ.get("RCLONE_CONF_TEXT", "")
    if text.strip():
        conf_path.write_text(text.strip() + "\n", encoding="utf-8")
    return conf_path


class CheckpointSyncer:
    def __init__(
        self,
        remote: str = "wylggww:shizi_v12/ckpts",
        max_inflight: int = 2,
        log_path: str | Path = "sync.log",
        extra_flags: list[str] | None = None,
    ) -> None:
        self.remote = remote.rstrip("/")
        self.max_inflight = max_inflight
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log = open(self.log_path, "ab")
        self.flags = extra_flags or [
            "--transfers", "4",
            "--checkers", "8",
            "--drive-chunk-size", "64M",
            "--retries", "5",
            "--low-level-retries", "10",
        ]
        self._pending: list[tuple[str, subprocess.Popen[bytes]]] = []

    def _reap(self) -> None:
        self._pending = [(t, p) for t, p in self._pending if p.poll() is None]

    def _wait_for_slot(self, target: str | None = None, timeout: float = 600.0) -> None:
        t0 = time.time()
        while time.time() - t0 < timeout:
            self._reap()
            same_busy = target is not None and any(t == target for t, _ in self._pending)
            if len(self._pending) < self.max_inflight and not same_busy:
                return
            time.sleep(2.0)

    def upload_async(self, local_path: str | Path, remote_name: str | None = None) -> None:
        """Queue an async upload of a file (using copyto to avoid dir-wrapping) or dir."""
        local = str(local_path)
        name = remote_name or os.path.basename(local.rstrip("/"))
        target = f"{self.remote}/{name}"
        self._wait_for_slot(target=target)
        sub = "copyto" if os.path.isfile(local) else "copy"
        cmd = ["rclone", sub, local, target, *self.flags]
        self.log.write(f"[{time.strftime('%F %T')}] START {' '.join(cmd)}\n".encode())
        self.log.flush()
        proc = subprocess.Popen(cmd, stdout=self.log, stderr=subprocess.STDOUT)
        self._pending.append((target, proc))

    def upload_sync(self, local_path: str | Path, remote_name: str | None = None, timeout: float = 1200.0) -> bool:
        """Blocking upload (used for final session exit or teacher feature cache)."""
        local = str(local_path)
        name = remote_name or os.path.basename(local.rstrip("/"))
        target = f"{self.remote}/{name}"
        sub = "copyto" if os.path.isfile(local) else "copy"
        cmd = ["rclone", sub, local, target, *self.flags]
        self.log.write(f"[{time.strftime('%F %T')}] SYNC {' '.join(cmd)}\n".encode())
        self.log.flush()
        r = subprocess.run(cmd, stdout=self.log, stderr=subprocess.STDOUT, timeout=timeout)
        return r.returncode == 0

    def download_if_exists(self, remote_name: str, local_path: str | Path, timeout: float = 900.0) -> bool:
        """Download a single remote file if present on Google Drive (for seamless --resume)."""
        src = f"{self.remote}/{remote_name}"
        dst = Path(local_path)
        dst.parent.mkdir(parents=True, exist_ok=True)
        chk = subprocess.run(["rclone", "lsf", src], capture_output=True, text=True, timeout=60)
        if chk.returncode != 0 or not chk.stdout.strip():
            return False
        cmd = ["rclone", "copyto", src, str(dst), *self.flags]
        self.log.write(f"[{time.strftime('%F %T')}] PULL {' '.join(cmd)}\n".encode())
        self.log.flush()
        r = subprocess.run(cmd, stdout=self.log, stderr=subprocess.STDOUT, timeout=timeout)
        return r.returncode == 0 and dst.exists() and dst.stat().st_size > 0

    def wait_all(self, timeout: float = 1800.0) -> None:
        t0 = time.time()
        for _, p in list(self._pending):
            remaining = max(1.0, timeout - (time.time() - t0))
            try:
                p.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                self.log.write(f"[{time.strftime('%F %T')}] TIMEOUT pid={p.pid}\n".encode())
        self._reap()
        self.log.write(f"[{time.strftime('%F %T')}] wait_all done\n".encode())
        self.log.flush()

    def healthy(self) -> bool:
        remote_name = self.remote.split(":")[0]
        try:
            r = subprocess.run(
                ["rclone", "lsd", f"{remote_name}:", "--max-depth", "1"],
                capture_output=True, timeout=60,
            )
            return r.returncode == 0
        except Exception:
            return False
