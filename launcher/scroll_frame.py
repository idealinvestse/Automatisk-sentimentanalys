"""Vertically scrollable container for Tkinter launcher on small screens."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Any


class ScrollableFrame(ttk.Frame):
    """Canvas + scrollbar wrapping an inner frame for overflow content."""

    def __init__(self, master: tk.Misc, **kwargs: Any) -> None:
        super().__init__(master, **kwargs)
        self._canvas = tk.Canvas(self, highlightthickness=0, borderwidth=0)
        self._scrollbar = ttk.Scrollbar(self, orient=tk.VERTICAL, command=self._canvas.yview)
        self.inner = ttk.Frame(self._canvas)

        self._window_id = self._canvas.create_window((0, 0), window=self.inner, anchor=tk.NW)
        self._canvas.configure(yscrollcommand=self._scrollbar.set)

        self._canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.inner.bind("<Configure>", self._on_inner_configure)
        self._canvas.bind("<Configure>", self._on_canvas_configure)
        self._canvas.bind("<Enter>", self._bind_mousewheel)
        self._canvas.bind("<Leave>", self._unbind_mousewheel)

    def _on_inner_configure(self, _event: tk.Event[object]) -> None:
        self._canvas.configure(scrollregion=self._canvas.bbox("all"))

    def _on_canvas_configure(self, event: tk.Event[object]) -> None:
        self._canvas.itemconfigure(self._window_id, width=event.width)

    def _on_mousewheel(self, event: tk.Event[object]) -> None:
        if event.num == 5 or getattr(event, "delta", 0) < 0:
            self._canvas.yview_scroll(1, "units")
        elif event.num == 4 or getattr(event, "delta", 0) > 0:
            self._canvas.yview_scroll(-1, "units")

    def _bind_mousewheel(self, _event: tk.Event[object]) -> None:
        self._canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self._canvas.bind_all("<Button-4>", self._on_mousewheel)
        self._canvas.bind_all("<Button-5>", self._on_mousewheel)

    def _unbind_mousewheel(self, _event: tk.Event[object]) -> None:
        self._canvas.unbind_all("<MouseWheel>")
        self._canvas.unbind_all("<Button-4>")
        self._canvas.unbind_all("<Button-5>")