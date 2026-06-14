"""
Homomorphic image dehazing — interactive GUI.

Left panel: filter tabs, switches, and run controls.
Right panel: Original and Dehazed previews.

Run:
    python HomomorphicImageDehaze.py
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from homomorphic_algorithm import (
    build_filter,
    homomorphic_dehaze,
    plot_transfer_function,
    show_histogram,
)

# Default filter parameters (GausHP tab)
DEFAULT_GAUS = {"d0": 1.0, "c": 2.0, "g_l": 0.1, "g_h": 1.1}
DEFAULT_BUTTER = {"d0": 1.0, "n": 2}
DEFAULT_IDEAL = {"d0": 1}

LEFT_PANEL_WIDTH = 270
WINDOW_WIDTH = 1180
WINDOW_HEIGHT = 780


class StableFigureCanvas(FigureCanvasTkAgg):
    """Matplotlib canvas that ignores move-only Configure events and debounces resizes."""

    def __init__(self, figure, master):
        super().__init__(figure, master)
        self._last_size = (0, 0)
        self._resize_after_id = None
        widget = self.get_tk_widget()
        widget.unbind("<Configure>")
        widget.bind("<Configure>", self._on_configure)

    def _on_configure(self, event):
        widget = self.get_tk_widget()
        if event.widget is not widget:
            return
        size = (event.width, event.height)
        if size == self._last_size or event.width < 200 or event.height < 200:
            return
        self._last_size = size
        root = widget.winfo_toplevel()
        if self._resize_after_id is not None:
            root.after_cancel(self._resize_after_id)
        self._resize_after_id = root.after(120, self._debounced_resize, event)

    def _debounced_resize(self, event):
        self._resize_after_id = None
        if event.width < 200 or event.height < 200:
            return
        super().resize(event)


class HorizontalSwitch(tk.Frame):
    """Horizontal On/Off toggle."""

    def __init__(self, master, label, **kwargs):
        super().__init__(master, **kwargs)
        self.var = tk.BooleanVar(value=False)
        tk.Label(self, text=label, anchor="center").pack()
        row = tk.Frame(self)
        row.pack()
        tk.Label(row, text="Off", font=("Segoe UI", 7)).pack(side=tk.LEFT)
        self.switch = tk.Checkbutton(
            row, variable=self.var, indicatoron=False,
            width=3, selectcolor="#0078d4",
        )
        self.switch.pack(side=tk.LEFT, padx=2)
        tk.Label(row, text="On", font=("Segoe UI", 7)).pack(side=tk.LEFT)

    def get(self):
        return self.var.get()

    def set_state(self, state):
        self.switch.config(state=state)


class VerticalSwitch(tk.Frame):
    """Vertical On/Off toggle."""

    def __init__(self, master, label, **kwargs):
        super().__init__(master, **kwargs)
        self.var = tk.BooleanVar(value=False)
        tk.Label(self, text=label, anchor="center").pack()
        tk.Label(self, text="On", font=("Segoe UI", 7)).pack()
        self.switch = tk.Checkbutton(
            self, variable=self.var, indicatoron=False,
            width=2, height=2, selectcolor="#0078d4",
        )
        self.switch.pack()
        tk.Label(self, text="Off", font=("Segoe UI", 7)).pack()

    def get(self):
        return self.var.get()

    def set_state(self, state):
        self.switch.config(state=state)


class HomomorphicImageDehazeApp:
    """Homomorphic filtering dehazing application."""

    def __init__(self, master):
        self.master = master
        self.image_data = None
        self.gray_float = None
        self.dehazed_data = None
        self.filter_h = None
        self.filter_type = "gaushp"
        self._filter_after_id = None
        self._build_window()
        self._build_layout()

    def _build_window(self):
        self.master.title("Homomorphic Image Dehazing")
        sw = self.master.winfo_screenwidth()
        sh = self.master.winfo_screenheight()
        x = int((sw - WINDOW_WIDTH) / 2)
        y = int((sh - WINDOW_HEIGHT) / 2)
        self.master.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}+{x}+{y}")
        self.master.minsize(900, 600)

    def _build_layout(self):
        pane = tk.PanedWindow(self.master, orient=tk.HORIZONTAL, sashwidth=4, bd=0)
        pane.pack(fill=tk.BOTH, expand=True)

        self.left_panel = tk.Frame(pane, width=LEFT_PANEL_WIDTH, bg="#f5f5f5")
        self.right_panel = tk.Frame(pane, bg="#ffffff")
        pane.add(self.left_panel, minsize=LEFT_PANEL_WIDTH, width=LEFT_PANEL_WIDTH)
        pane.add(self.right_panel, minsize=640)

        self._build_left_panel()
        self._build_right_panel()

    def _build_left_panel(self):
        p = self.left_panel
        row = 0

        self.select_btn = tk.Button(p, text="Select Image", command=self.select_image, width=16)
        self.select_btn.grid(row=row, column=0, columnspan=2, pady=(12, 4), padx=12)
        row += 1

        self.hist_btn = tk.Button(
            p, text="Histogram", command=self.show_hist, state="disabled", width=16,
        )
        self.hist_btn.grid(row=row, column=0, columnspan=2, pady=(0, 8), padx=12)
        row += 1

        self.status_label = tk.Label(p, text="", anchor="w", fg="#555555", bg="#f5f5f5")
        self.status_label.grid(row=row, column=0, columnspan=2, sticky="ew", padx=12, pady=(0, 6))
        row += 1

        self.notebook = ttk.Notebook(p)
        self.notebook.grid(row=row, column=0, columnspan=2, sticky="ew", padx=12, pady=4)
        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)

        self._build_gaushp_tab()
        self._build_butter_tab()
        self._build_ideal_tab()
        row += 1

        self.filter_info = tk.Label(
            p, text="Filter: —", anchor="w", bg="#f5f5f5", fg="#333333",
            font=("Segoe UI", 9), wraplength=250, justify="left",
        )
        self.filter_info.grid(row=row, column=0, columnspan=2, sticky="ew", padx=12, pady=(0, 4))
        row += 1

        self.disp_filter_btn = tk.Button(
            p, text="Disp Selected Filter", command=self.display_filter,
            state="disabled", width=18,
        )
        self.disp_filter_btn.grid(row=row, column=0, columnspan=2, pady=8, padx=12)
        row += 1

        sw_row1 = tk.Frame(p, bg="#f5f5f5")
        sw_row1.grid(row=row, column=0, columnspan=2, pady=4)
        self.i_fft_sw = HorizontalSwitch(sw_row1, "I_FFT")
        self.i_fft_sw.pack(side=tk.LEFT, padx=8)
        self.i_fft_sw.set_state("disabled")
        self.g_fft_sw = HorizontalSwitch(sw_row1, "G*I_FFT")
        self.g_fft_sw.pack(side=tk.LEFT, padx=8)
        self.g_fft_sw.set_state("disabled")
        row += 1

        sw_row2 = tk.Frame(p, bg="#f5f5f5")
        sw_row2.grid(row=row, column=0, columnspan=2, pady=4)
        self.gray_sw = HorizontalSwitch(sw_row2, "Dehazed_Gray")
        self.gray_sw.pack(side=tk.LEFT, padx=8)
        self.gray_sw.set_state("disabled")
        self.rgb_sw = HorizontalSwitch(sw_row2, "Dehazed_RGB")
        self.rgb_sw.pack(side=tk.LEFT, padx=8)
        self.rgb_sw.set_state("disabled")
        row += 1

        action = tk.Frame(p, bg="#f5f5f5")
        action.grid(row=row, column=0, columnspan=2, pady=(8, 4))
        self.adapt_eq = VerticalSwitch(action, "Adapt_EQ")
        self.adapt_eq.pack(side=tk.LEFT, padx=(12, 8))
        self.adapt_eq.set_state("disabled")
        self.run_btn = tk.Button(
            action, text="Run", command=self.run, state="disabled", width=14, height=3,
        )
        self.run_btn.pack(side=tk.LEFT, padx=8)
        row += 1

        diagram = (
            "f(x,y) → ln → FFT → H(u,v) → IFFT → exp → f'(x,y)"
        )
        tk.Label(
            p, text=diagram, font=("Segoe UI", 8), wraplength=250,
            justify="left", bg="#f5f5f5", fg="#333333",
        ).grid(row=row, column=0, columnspan=2, sticky="sw", padx=12, pady=(12, 8))

    def _field(self, parent, label, default, row, col, command=None):
        tk.Label(parent, text=label).grid(row=row, column=col, padx=4, pady=4, sticky="e")
        var = tk.DoubleVar(value=default)
        entry = tk.Entry(parent, textvariable=var, width=6)
        entry.grid(row=row, column=col + 1, padx=4, pady=4, sticky="w")
        if command:
            var.trace_add("write", lambda *_: self._schedule_rebuild_filter())
        return var

    def _build_gaushp_tab(self):
        tab = tk.Frame(self.notebook)
        self.notebook.add(tab, text="GausHP")
        self.gaus_d0 = self._field(tab, "D0", DEFAULT_GAUS["d0"], 0, 0, self._rebuild_filter)
        self.gaus_c = self._field(tab, "C", DEFAULT_GAUS["c"], 1, 0, self._rebuild_filter)
        self.gaus_gl = self._field(tab, "gL", DEFAULT_GAUS["g_l"], 0, 2, self._rebuild_filter)
        self.gaus_gh = self._field(tab, "gH", DEFAULT_GAUS["g_h"], 1, 2, self._rebuild_filter)

    def _build_butter_tab(self):
        tab = tk.Frame(self.notebook)
        self.notebook.add(tab, text="ButterHP")
        self.butter_d0 = self._field(tab, "D0", DEFAULT_BUTTER["d0"], 0, 0, self._rebuild_filter)
        self.butter_n = self._field(tab, "N", DEFAULT_BUTTER["n"], 1, 0, self._rebuild_filter)

    def _build_ideal_tab(self):
        tab = tk.Frame(self.notebook)
        self.notebook.add(tab, text="IdealHP")
        self.ideal_d0 = self._field(tab, "D0", DEFAULT_IDEAL["d0"], 0, 0, self._rebuild_filter)

    def _build_right_panel(self):
        self.fig = Figure(figsize=(8.5, 10), dpi=100)
        self.fig.subplots_adjust(left=0.01, right=0.99, top=0.97, bottom=0.01, hspace=0.06)
        self.ax_original = self.fig.add_subplot(211)
        self.ax_dehazed = self.fig.add_subplot(212)
        self.ax_original.set_title("Original", fontsize=11, pad=6)
        self.ax_dehazed.set_title("Dehazed", fontsize=11, pad=6)
        for ax in (self.ax_original, self.ax_dehazed):
            ax.axis("off")

        self.canvas = StableFigureCanvas(self.fig, master=self.right_panel)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

    def _show_bgr_in_axes(self, ax, image_bgr, title, redraw=True):
        ax.clear()
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        ax.imshow(rgb, extent=(0, w, h, 0), aspect="equal", interpolation="bilinear")
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)
        ax.set_aspect("equal", adjustable="box")
        ax.set_anchor("C")
        ax.set_title(title, fontsize=11, pad=6)
        ax.axis("off")
        if redraw:
            self.canvas.draw_idle()

    def _on_tab_changed(self, _event=None):
        tab_id = self.notebook.index(self.notebook.select())
        self.filter_type = ("gaushp", "butterhp", "idealhp")[tab_id]
        self._rebuild_filter()

    def _get_filter_params(self):
        if self.filter_type == "gaushp":
            return {
                "g_l": float(self.gaus_gl.get()),
                "g_h": float(self.gaus_gh.get()),
                "d0": float(self.gaus_d0.get()),
                "c": float(self.gaus_c.get()),
            }
        if self.filter_type == "butterhp":
            return {"d0": float(self.butter_d0.get()), "n": int(float(self.butter_n.get()))}
        return {"d0": int(float(self.ideal_d0.get()))}

    def _schedule_rebuild_filter(self):
        if self._filter_after_id is not None:
            self.master.after_cancel(self._filter_after_id)
        self._filter_after_id = self.master.after(250, self._rebuild_filter)

    def _update_filter_info(self):
        if self.filter_h is None:
            self.filter_info.config(text="Filter: —")
            return
        params = self._get_filter_params()
        param_str = ", ".join(
            f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
            for k, v in params.items()
        )
        label = {"gaushp": "GausHP", "butterhp": "ButterHP", "idealhp": "IdealHP"}[
            self.filter_type
        ]
        h_min, h_max = float(self.filter_h.min()), float(self.filter_h.max())
        self.filter_info.config(
            text=f"Filter: {label}  ({param_str})\nH range: [{h_min:.4g}, {h_max:.4g}]",
        )

    def _rebuild_filter(self):
        self._filter_after_id = None
        if self.gray_float is None:
            return
        try:
            self.filter_h = build_filter(
                self.filter_type, self.gray_float.shape, **self._get_filter_params(),
            )
            self._update_filter_info()
            self._enable_filter_controls()
        except (ValueError, tk.TclError):
            pass

    def _enable_filter_controls(self):
        self.hist_btn.config(state="normal")
        self.disp_filter_btn.config(state="normal")
        self.run_btn.config(state="normal")
        for sw in (
            self.i_fft_sw, self.g_fft_sw, self.gray_sw,
            self.rgb_sw, self.adapt_eq,
        ):
            sw.set_state("normal")

    def select_image(self):
        file_path = filedialog.askopenfilename(
            title="File Selector",
            filetypes=[
                ("Image files", "*.jpg;*.jpeg;*.png;*.gif;*.tif;*.bmp"),
                ("All files", "*.*"),
            ],
        )
        if not file_path:
            messagebox.showinfo("Info", "There is NO image selected")
            return

        image = cv2.imread(file_path)
        if image is None:
            messagebox.showerror("Error", "Could not read the selected image.")
            return

        cv2.destroyAllWindows()
        plt.close("all")

        self.image_data = image
        self.gray_float = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0
        self.dehazed_data = None
        self._show_bgr_in_axes(self.ax_original, self.image_data, "Original")
        self.ax_dehazed.clear()
        self.ax_dehazed.set_title("Dehazed", fontsize=11, pad=6)
        self.ax_dehazed.axis("off")
        self.canvas.draw_idle()
        self._rebuild_filter()

    def show_hist(self):
        if self.gray_float is None:
            return
        show_histogram(self.gray_float)
        plt.show(block=False)

    def display_filter(self):
        if self.filter_h is None:
            messagebox.showinfo("Info", "Select an image and configure a filter first.")
            return
        cv2.destroyAllWindows()
        plot_transfer_function(
            self.filter_h,
            filter_type=self.filter_type,
            params=self._get_filter_params(),
        )
        plt.show(block=False)

    def run(self):
        if self.image_data is None:
            messagebox.showinfo("Info", "There is NO image selected")
            return
        if self.filter_h is None:
            messagebox.showinfo("Info", "Configure a filter before running.")
            return

        cv2.destroyAllWindows()
        self.run_btn.config(state="disabled")
        self.status_label.config(text="Processing...")
        self.master.update_idletasks()

        result = homomorphic_dehaze(
            self.image_data,
            self.filter_h,
            show_i_fft=self.i_fft_sw.get(),
            show_g_fft=self.g_fft_sw.get(),
            show_dehazed_gray=self.gray_sw.get(),
            show_dehazed_rgb=self.rgb_sw.get(),
            adapt_eq=self.adapt_eq.get(),
        )

        self.dehazed_data = result["I_defog"]
        self._show_bgr_in_axes(self.ax_dehazed, self.dehazed_data, "Dehazed")
        self.status_label.config(text="Done.")
        self.run_btn.config(state="normal")

        if any([
            self.i_fft_sw.get(), self.g_fft_sw.get(),
            self.gray_sw.get(), self.rgb_sw.get(), self.adapt_eq.get(),
        ]):
            cv2.waitKey(0)
            cv2.destroyAllWindows()


def main():
    root = tk.Tk()
    HomomorphicImageDehazeApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
