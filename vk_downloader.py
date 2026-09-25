#!/usr/bin/env python3
"""
VK Video Downloader — GUI (tkinter) + CLI поверх yt-dlp.

Запуск GUI:   python vk_downloader.py
Запуск CLI:   python vk_downloader.py <url> [-q 720] [-o папка] [--cookies-from-browser chrome]

Требуется: pip install -U yt-dlp   и   ffmpeg в PATH (для склейки видео+аудио / HLS).
Скачивайте только то, на что у вас есть право (личное использование, собственный контент,
материалы с разрешением автора).
"""
import argparse
import os
import re
import sys
import threading

try:
    import yt_dlp
except ImportError:
    sys.exit("Не найден yt-dlp. Установите: pip install -U yt-dlp")

QUALITIES = ["best", "1080", "720", "480", "360", "240", "audio"]


def normalize_url(url: str) -> str:
    """live-123_456 / video123_456 / vk.com/video... -> https://vkvideo.ru/video-123_456"""
    url = url.strip()
    m = re.search(r"(?:live|video|clip)(-?\d+_\d+)", url)
    if m:
        return f"https://vkvideo.ru/video{m.group(1)}"
    return url


def format_selector(quality: str) -> str:
    if quality == "audio":
        return "bestaudio/best"
    if quality == "best":
        return "bv*+ba/b"
    h = int(quality)
    return f"bv*[height<={h}]+ba/b[height<={h}]/b"


def build_opts(outdir, quality, cookies_browser=None, cookies_file=None,
               progress_cb=None, log_cb=None):
    class _Logger:
        def debug(self, m):
            if log_cb and not m.startswith("[debug]"):
                log_cb(m)
        def info(self, m): log_cb and log_cb(m)
        def warning(self, m): log_cb and log_cb("WARN: " + m)
        def error(self, m): log_cb and log_cb("ERROR: " + m)

    opts = {
        "format": format_selector(quality),
        "outtmpl": os.path.join(outdir, "%(title).150B [%(id)s].%(ext)s"),
        "merge_output_format": "mp4",
        "concurrent_fragment_downloads": int(os.environ.get("VK_FRAGMENTS", "4")),
        "retries": 10,
        "fragment_retries": 10,
        "noplaylist": True,
        "restrictfilenames": False,
        "windowsfilenames": True,
        "quiet": log_cb is not None,
        "no_warnings": False,
    }
    if quality == "audio":
        opts["postprocessors"] = [{"key": "FFmpegExtractAudio", "preferredcodec": "mp3"}]
    if cookies_browser:
        opts["cookiesfrombrowser"] = (cookies_browser,)
    if cookies_file:
        opts["cookiefile"] = cookies_file
    if progress_cb:
        opts["progress_hooks"] = [progress_cb]
    if log_cb:
        opts["logger"] = _Logger()
    return opts


def list_formats(url, **kw):
    with yt_dlp.YoutubeDL(build_opts(".", "best", **kw)) as ydl:
        info = ydl.extract_info(normalize_url(url), download=False)
    heights = sorted({f.get("height") for f in info.get("formats", []) if f.get("height")}, reverse=True)
    return info.get("title"), info.get("duration"), heights


def download(url, outdir, quality, **kw):
    os.makedirs(outdir, exist_ok=True)
    with yt_dlp.YoutubeDL(build_opts(outdir, quality, **kw)) as ydl:
        return ydl.download([normalize_url(url)])


# ------------------------------------------------------------------ GUI
def run_gui():
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox

    root = tk.Tk()
    root.title("VK Video Downloader")
    root.geometry("640x420")

    url_var = tk.StringVar()
    dir_var = tk.StringVar(value=os.path.join(os.path.expanduser("~"), "Downloads"))
    q_var = tk.StringVar(value="best")
    ck_var = tk.StringVar(value="")

    pad = {"padx": 8, "pady": 4}
    frm = ttk.Frame(root); frm.pack(fill="x", **pad)
    frm.columnconfigure(1, weight=1)

    ttk.Label(frm, text="Ссылка:").grid(row=0, column=0, sticky="w")
    ttk.Entry(frm, textvariable=url_var).grid(row=0, column=1, columnspan=2, sticky="ew")

    ttk.Label(frm, text="Папка:").grid(row=1, column=0, sticky="w")
    ttk.Entry(frm, textvariable=dir_var).grid(row=1, column=1, sticky="ew")
    ttk.Button(frm, text="...", width=3,
               command=lambda: dir_var.set(filedialog.askdirectory() or dir_var.get())
               ).grid(row=1, column=2)

    ttk.Label(frm, text="Качество:").grid(row=2, column=0, sticky="w")
    ttk.Combobox(frm, textvariable=q_var, values=QUALITIES, state="readonly", width=10
                 ).grid(row=2, column=1, sticky="w")

    ttk.Label(frm, text="Cookies из браузера:").grid(row=3, column=0, sticky="w")
    ttk.Combobox(frm, textvariable=ck_var, values=["", "chrome", "firefox", "edge", "opera", "brave"],
                 width=10).grid(row=3, column=1, sticky="w")

    bar = ttk.Progressbar(root, maximum=100); bar.pack(fill="x", **pad)
    status = tk.StringVar(value="Готово к загрузке")
    ttk.Label(root, textvariable=status).pack(anchor="w", padx=8)
    log = tk.Text(root, height=12, state="disabled"); log.pack(fill="both", expand=True, **pad)

    def ui(fn):  # выполнить в потоке UI
        root.after(0, fn)

    def add_log(msg):
        def _():
            log.configure(state="normal"); log.insert("end", msg + "\n")
            log.see("end"); log.configure(state="disabled")
        ui(_)

    def on_progress(d):
        if d["status"] == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate")
            done = d.get("downloaded_bytes", 0)
            pct = done * 100 / total if total else 0
            speed = d.get("speed") or 0
            ui(lambda: (bar.configure(value=pct),
                        status.set(f"{pct:.1f}%  {speed/1e6:.2f} МБ/с")))
        elif d["status"] == "finished":
            ui(lambda: status.set("Обработка (склейка)..."))

    btn = ttk.Button(root, text="Скачать")

    def work():
        try:
            download(url_var.get(), dir_var.get(), q_var.get(),
                     cookies_browser=ck_var.get() or None,
                     progress_cb=on_progress, log_cb=add_log)
            ui(lambda: (status.set("Готово"), bar.configure(value=100)))
        except Exception as e:
            add_log(f"ERROR: {e}")
            ui(lambda: status.set("Ошибка"))
        finally:
            ui(lambda: btn.configure(state="normal"))

    def start():
        if not url_var.get().strip():
            messagebox.showwarning("VK Downloader", "Вставьте ссылку на видео")
            return
        btn.configure(state="disabled"); bar.configure(value=0)
        threading.Thread(target=work, daemon=True).start()

    btn.configure(command=start); btn.pack(pady=6)
    root.mainloop()


# ------------------------------------------------------------------ CLI
def main():
    if len(sys.argv) == 1:
        return run_gui()
    p = argparse.ArgumentParser(description="VK Video Downloader")
    p.add_argument("url")
    p.add_argument("-q", "--quality", default="best", choices=QUALITIES)
    p.add_argument("-o", "--output", default=".")
    p.add_argument("--list", action="store_true", help="показать доступные качества")
    p.add_argument("--cookies-from-browser")
    p.add_argument("--cookies", help="cookies.txt (Netscape)")
    a = p.parse_args()
    kw = dict(cookies_browser=a.cookies_from_browser, cookies_file=a.cookies)
    if a.list:
        title, dur, hs = list_formats(a.url, **kw)
        print(title, dur, "сек;", "качества:", ", ".join(f"{h}p" for h in hs))
        return
    download(a.url, a.output, a.quality, **kw)


if __name__ == "__main__":
    main()
