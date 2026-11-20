"""Small tkinter front-end showing how to use msadpcm.

    python examples/gui_demo.py

Pick a WAV file, choose a block size and quality, press "Convert".
All the real work is one call: ``msadpcm.cli.convert_file``.  The GUI only
runs it in a thread and shows the progress callback.
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Optional, Tuple

from msadpcm import QUALITY_LEVELS
from msadpcm.cli import ConversionResult, convert_file

#: label -> (block_align bytes, samples per block); exactly one of them is used
BLOCK_CHOICES = {
    "Auto (like Windows ACM)": (None, None),
    "256 bytes": (256, None),
    "512 bytes": (512, None),
    "1024 bytes": (1024, None),
    "2048 bytes": (2048, None),
    "128 samples (70/140-byte blocks)": (None, 128),
}


def parse_block_choice(label: str) -> Tuple[Optional[int], Optional[int]]:
    """Translate a combobox label into ``(block_align, samples_per_block)``."""
    return BLOCK_CHOICES[label]


def main() -> None:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    root = tk.Tk()
    root.title("msadpcm demo")
    root.geometry("640x420")

    mode = tk.StringVar(value="encode")
    src_var = tk.StringVar()
    dst_var = tk.StringVar()
    block_var = tk.StringVar(value="Auto (like Windows ACM)")
    quality_var = tk.StringVar(value="high")
    status_var = tk.StringVar(value="Choose a WAV file.")

    pad = dict(padx=8, pady=4)
    frm = ttk.Frame(root)
    frm.pack(fill="both", expand=True, padx=8, pady=8)
    frm.columnconfigure(1, weight=1)

    ttk.Label(frm, text="Mode:").grid(row=0, column=0, sticky="w", **pad)
    modes = ttk.Frame(frm)
    modes.grid(row=0, column=1, columnspan=2, sticky="w")
    ttk.Radiobutton(modes, text="Encode: WAV → MS ADPCM", variable=mode, value="encode").pack(side="left")
    ttk.Radiobutton(modes, text="Decode: MS ADPCM → PCM WAV", variable=mode, value="decode").pack(side="left", padx=12)

    def browse_src() -> None:
        path = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav"), ("All files", "*.*")])
        if path:
            src_var.set(path)
            suffix = ".adpcm.wav" if mode.get() == "encode" else ".pcm.wav"
            dst_var.set(str(Path(path).with_name(Path(path).stem + suffix)))

    def browse_dst() -> None:
        path = filedialog.asksaveasfilename(defaultextension=".wav", filetypes=[("WAV files", "*.wav")])
        if path:
            dst_var.set(path)

    ttk.Label(frm, text="Input:").grid(row=1, column=0, sticky="w", **pad)
    ttk.Entry(frm, textvariable=src_var).grid(row=1, column=1, sticky="ew", **pad)
    ttk.Button(frm, text="Browse…", command=browse_src).grid(row=1, column=2, **pad)
    ttk.Label(frm, text="Output:").grid(row=2, column=0, sticky="w", **pad)
    ttk.Entry(frm, textvariable=dst_var).grid(row=2, column=1, sticky="ew", **pad)
    ttk.Button(frm, text="Save as…", command=browse_dst).grid(row=2, column=2, **pad)

    ttk.Label(frm, text="Block size:").grid(row=3, column=0, sticky="w", **pad)
    ttk.Combobox(frm, textvariable=block_var, values=list(BLOCK_CHOICES),
                 state="readonly", width=34).grid(row=3, column=1, sticky="w", **pad)
    ttk.Label(frm, text="Quality:").grid(row=4, column=0, sticky="w", **pad)
    ttk.Combobox(frm, textvariable=quality_var, values=list(QUALITY_LEVELS),
                 state="readonly", width=12).grid(row=4, column=1, sticky="w", **pad)

    button = ttk.Button(frm, text="Convert")
    button.grid(row=5, column=1, sticky="w", **pad)
    bar = ttk.Progressbar(frm, mode="determinate", maximum=1000)
    bar.grid(row=6, column=0, columnspan=3, sticky="ew", **pad)
    ttk.Label(frm, textvariable=status_var).grid(row=7, column=0, columnspan=3, sticky="w", **pad)
    result = tk.Text(frm, height=8, state="disabled", wrap="word")
    result.grid(row=8, column=0, columnspan=3, sticky="nsew", **pad)
    frm.rowconfigure(8, weight=1)

    def show(text: str) -> None:
        result.configure(state="normal")
        result.delete("1.0", "end")
        result.insert("end", text)
        result.configure(state="disabled")

    def on_progress(done: int, total: int) -> None:  # called from the worker thread
        root.after(0, lambda: (bar.configure(value=1000 * done / max(1, total)),
                               status_var.set("Encoding… %d%%" % (100 * done // max(1, total)))))

    def finished(res: Optional[ConversionResult], error: Optional[Exception]) -> None:
        button.configure(state="normal")
        if error is not None:
            status_var.set("Failed.")
            messagebox.showerror("msadpcm", str(error))
            return
        bar.configure(value=1000)
        status_var.set("Done in %.2f s." % res.seconds)
        lines = [
            "%s  ->  %s" % (res.src.name, res.dst.name),
            "Audio: %.2f s, %d Hz, %s" % (res.frames / res.sample_rate, res.sample_rate,
                                          "stereo" if res.channels == 2 else "mono"),
            "Size: %d -> %d bytes (x%.2f)" % (res.src_bytes, res.dst_bytes, res.ratio),
        ]
        if res.mode == "encode":
            lines.append("Block: %d bytes = %d samples per channel" % (res.block_align, res.samples_per_block))
            if res.snr_db is not None:
                lines.append("SNR after decoding again: %.2f dB" % res.snr_db)
        show("\n".join(lines))

    def worker(src: str, dst: str, kwargs: dict) -> None:
        try:
            res = convert_file(src, dst, progress=on_progress, **kwargs)
            root.after(0, lambda: finished(res, None))
        except Exception as exc:  # shown to the user instead of dying silently
            root.after(0, lambda: finished(None, exc))

    def start() -> None:
        src = src_var.get().strip()
        if not Path(src).is_file():
            messagebox.showerror("msadpcm", "Input file does not exist.")
            return
        ba, spb = parse_block_choice(block_var.get())
        kwargs = dict(mode=mode.get(), block_align=ba, samples_per_block=spb, quality=quality_var.get())
        button.configure(state="disabled")
        bar.configure(value=0)
        status_var.set("Working…")
        threading.Thread(target=worker, args=(src, dst_var.get() or None, kwargs), daemon=True).start()

    button.configure(command=start)
    root.mainloop()


if __name__ == "__main__":
    main()
