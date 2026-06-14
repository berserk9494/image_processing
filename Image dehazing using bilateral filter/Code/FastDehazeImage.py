"""
Fast image dehazing — interactive GUI.

Left panel: parameters and run controls.
Right panel: Original and Dehazed (J) previews.

Run:
    python FastDehazeImage.py
"""

import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from dehaze_algorithm import adaptive_median_window, dehaze

DEFAULT_KERNEL = 15
DEFAULT_OMEGA = 11
DEFAULT_SIGMA_R = 20.0
DEFAULT_SIGMA_T = 20.0
DEFAULT_W = 0.95
DEFAULT_P = 0.95
DEFAULT_T0 = 0.3

LEFT_PANEL_WIDTH = 260
WINDOW_WIDTH = 1180
WINDOW_HEIGHT = 780
SIGMA_TICKS = (0.01, 20, 40, 60, 80, 100)


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


class ParameterSlider(tk.Frame):
    """Horizontal parameter slider with tick labels and live value readout."""

    def __init__(self, master, label, from_, to, default, ticks=None, **kwargs):
        super().__init__(master, **kwargs)
        self._updating = False

        header = tk.Frame(self)
        header.pack(fill=tk.X, padx=4)
        self.name_label = tk.Label(header, text=label, anchor="w", width=10)
        self.name_label.pack(side=tk.LEFT)
        self.value_label = tk.Label(
            header, text=self._format_value(default), anchor="e", width=8,
        )
        self.value_label.pack(side=tk.RIGHT)

        scale_wrap = tk.Frame(self)
        scale_wrap.pack(fill=tk.X, padx=8, pady=(6, 2))

        self.var = tk.DoubleVar(value=default)
        self.scale = tk.Scale(
            scale_wrap,
            from_=from_,
            to=to,
            orient=tk.HORIZONTAL,
            variable=self.var,
            showvalue=False,
            resolution=0.01,
            sliderlength=18,
            length=190,
            troughcolor="#d6d6d6",
            activebackground="#0078d4",
            highlightthickness=0,
            bd=0,
            command=self._on_slide,
        )
        self.scale.pack(fill=tk.X)

        if ticks:
            tick_row = tk.Frame(self)
            tick_row.pack(fill=tk.X, padx=10)
            tick_row.columnconfigure(tuple(range(len(ticks))), weight=1)
            for col, tick in enumerate(ticks):
                text = f"{tick:g}" if tick >= 1 else f"{tick:.2f}"
                tk.Label(
                    tick_row, text=text, font=("Segoe UI", 8), fg="#444444",
                ).grid(row=0, column=col, sticky="ew")

    @staticmethod
    def _format_value(value):
        return f"{value:.2f}" if value < 1 else f"{value:g}"

    def _on_slide(self, _value):
        if self._updating:
            return
        self.value_label.config(text=self._format_value(self.var.get()))

    def get(self):
        return self.var.get()

    def set_state(self, state):
        self.name_label.config(state=state)
        self.value_label.config(state=state)
        self.scale.config(state=state)


class VerticalSwitch(tk.Frame):
    """Vertical On/Off toggle for intermediate map display."""

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


class FastDehazeImageApp:
    """Bilateral-filter dehazing application."""

    def __init__(self, master):
        self.master = master
        self.image_data = None
        self.dehazed_data = None
        self._build_window()
        self._build_layout()

    def _build_window(self):
        self.master.title("Fast Image Dehazing")
        sw = self.master.winfo_screenwidth()
        sh = self.master.winfo_screenheight()
        x = int((sw - WINDOW_WIDTH) / 2)
        y = int((sh - WINDOW_HEIGHT) / 2)
        self.master.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}+{x}+{y}")
        self.master.minsize(900, 600)

    def _build_layout(self):
        self.main_pane = tk.PanedWindow(
            self.master, orient=tk.HORIZONTAL, sashwidth=4, bd=0,
        )
        self.main_pane.pack(fill=tk.BOTH, expand=True)

        self.left_panel = tk.Frame(self.main_pane, width=LEFT_PANEL_WIDTH, bg="#f5f5f5")
        self.right_panel = tk.Frame(self.main_pane, bg="#ffffff")
        self.main_pane.add(self.left_panel, minsize=LEFT_PANEL_WIDTH, width=LEFT_PANEL_WIDTH)
        self.main_pane.add(self.right_panel, minsize=640)

        self._build_left_panel()
        self._build_right_panel()

    def _build_left_panel(self):
        p = self.left_panel
        p.columnconfigure(1, weight=1)

        row = 0
        self.select_btn = tk.Button(p, text="Select Image", command=self.select_image, width=16)
        self.select_btn.grid(row=row, column=0, columnspan=2, pady=(12, 8), padx=12)

        row += 1
        self.status_label = tk.Label(p, text="", anchor="w", fg="#555555", bg="#f5f5f5")
        self.status_label.grid(row=row, column=0, columnspan=2, sticky="ew", padx=12, pady=(0, 6))

        row += 1
        self.kernel_label = tk.Label(p, text="Kernel", anchor="e", state="disabled", bg="#f5f5f5")
        self.kernel_label.grid(row=row, column=0, sticky="e", padx=(12, 6), pady=4)
        self.kernel_var = tk.IntVar(value=DEFAULT_KERNEL)
        self.kernel_spin = tk.Spinbox(
            p, from_=1, to=50, increment=2, textvariable=self.kernel_var,
            width=10, state="disabled", command=self._round_spinner,
        )
        self.kernel_spin.grid(row=row, column=1, sticky="w", padx=(0, 12), pady=4)

        row += 1
        self.omega_label = tk.Label(p, text="Omega", anchor="e", state="disabled", bg="#f5f5f5")
        self.omega_label.grid(row=row, column=0, sticky="e", padx=(12, 6), pady=4)
        self.omega_var = tk.IntVar(value=DEFAULT_OMEGA)
        self.omega_spin = tk.Spinbox(
            p, from_=1, to=50, increment=2, textvariable=self.omega_var,
            width=10, state="disabled", command=self._round_spinner,
        )
        self.omega_spin.grid(row=row, column=1, sticky="w", padx=(0, 12), pady=4)

        row += 1
        self.sigma_r_slider = ParameterSlider(
            p, "Sigma_r", 0.01, 100, DEFAULT_SIGMA_R, ticks=SIGMA_TICKS,
        )
        self.sigma_r_slider.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(10, 4))
        self.sigma_r_slider.set_state("disabled")

        row += 1
        self.sigma_t_slider = ParameterSlider(
            p, "Sigma_t", 0.01, 100, DEFAULT_SIGMA_T, ticks=SIGMA_TICKS,
        )
        self.sigma_t_slider.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(4, 10))
        self.sigma_t_slider.set_state("disabled")

        row += 1
        wp_frame = tk.Frame(p, bg="#f5f5f5")
        wp_frame.grid(row=row, column=0, columnspan=2, pady=8)

        self.w_label = tk.Label(wp_frame, text="w", state="disabled", bg="#f5f5f5")
        self.w_label.pack(side=tk.LEFT, padx=(12, 4))
        self.w_var = tk.DoubleVar(value=DEFAULT_W)
        self.w_entry = tk.Entry(wp_frame, textvariable=self.w_var, width=6, state="disabled")
        self.w_entry.pack(side=tk.LEFT)

        self.p_label = tk.Label(wp_frame, text="p", state="disabled", bg="#f5f5f5")
        self.p_label.pack(side=tk.LEFT, padx=(24, 4))
        self.p_var = tk.DoubleVar(value=DEFAULT_P)
        self.p_entry = tk.Entry(wp_frame, textvariable=self.p_var, width=6, state="disabled")
        self.p_entry.pack(side=tk.LEFT)

        row += 1
        switch_frame = tk.Frame(p, bg="#f5f5f5")
        switch_frame.grid(row=row, column=0, columnspan=2, pady=(6, 4))
        switch_names = ["W", "V", "R", "V_R", "t", "J"]
        self.switches = {}
        for name in switch_names:
            sw = VerticalSwitch(switch_frame, name)
            sw.pack(side=tk.LEFT, padx=4)
            sw.set_state("disabled")
            self.switches[name] = sw

        row += 1
        action_frame = tk.Frame(p, bg="#f5f5f5")
        action_frame.grid(row=row, column=0, columnspan=2, pady=(8, 4), sticky="ew")

        self.adapt_eq = VerticalSwitch(action_frame, "Adapt_EQ")
        self.adapt_eq.pack(side=tk.LEFT, padx=(12, 8))
        self.adapt_eq.set_state("disabled")

        self.run_btn = tk.Button(
            action_frame, text="Run", command=self.run, state="disabled", width=14, height=3,
        )
        self.run_btn.pack(side=tk.LEFT, padx=8)

        row += 1
        diagram_text = (
            "W → median → V; W → bilateral → R; "
            "guided joint bilateral(V,R) → t → J"
        )
        tk.Label(
            p, text=diagram_text, font=("Segoe UI", 8), wraplength=240,
            justify="left", bg="#f5f5f5", fg="#333333",
        ).grid(row=row, column=0, columnspan=2, sticky="sw", padx=12, pady=(12, 8))

    def _build_right_panel(self):
        self.fig = Figure(figsize=(8.5, 10), dpi=100)
        self.fig.subplots_adjust(left=0.01, right=0.99, top=0.97, bottom=0.01, hspace=0.06)
        self.ax_original = self.fig.add_subplot(211)
        self.ax_dehazed = self.fig.add_subplot(212)
        self.ax_original.set_title("Original", fontsize=11, pad=6)
        self.ax_dehazed.set_title("Dehazed (J)", fontsize=11, pad=6)
        for ax in (self.ax_original, self.ax_dehazed):
            ax.axis("off")

        self.canvas = StableFigureCanvas(self.fig, master=self.right_panel)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

    def _round_spinner(self):
        for var, spin in ((self.kernel_var, self.kernel_spin), (self.omega_var, self.omega_spin)):
            try:
                val = int(float(spin.get()))
                val = max(1, min(50, val))
                if val % 2 == 0:
                    val = max(1, val - 1)
                var.set(val)
            except (tk.TclError, ValueError):
                pass

    def _enable_controls(self):
        for w in (
            self.kernel_label, self.kernel_spin, self.omega_label, self.omega_spin,
            self.w_label, self.w_entry, self.p_label, self.p_entry, self.run_btn,
        ):
            w.config(state="normal")
        self.sigma_r_slider.set_state("normal")
        self.sigma_t_slider.set_state("normal")
        for sw in self.switches.values():
            sw.set_state("normal")
        self.adapt_eq.set_state("normal")

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

        self.image_data = image
        self.dehazed_data = None
        h, w = image.shape[:2]
        self.omega_var.set(adaptive_median_window(h, w))
        self._show_bgr_in_axes(self.ax_original, self.image_data, "Original")
        self.ax_dehazed.clear()
        self.ax_dehazed.set_title("Dehazed (J)", fontsize=11, pad=6)
        self.ax_dehazed.axis("off")
        self.canvas.draw_idle()
        self._enable_controls()

    def _update_progress(self, fraction):
        self.status_label.config(text=f"Processing: {fraction * 100:.0f}%")
        self.master.update_idletasks()

    def run(self):
        if self.image_data is None:
            messagebox.showinfo("Info", "There is NO image selected")
            return

        cv2.destroyAllWindows()
        self._round_spinner()

        switch_names = ["W", "V", "R", "V_R", "t", "J"]
        show_flags = {name: self.switches[name].get() for name in switch_names}
        adapt_eq = self.adapt_eq.get()
        params = {
            "radius": int(self.kernel_var.get()),
            "omega": int(self.omega_var.get()),
            "sigma_r": float(self.sigma_r_slider.get()),
            "sigma_t": float(self.sigma_t_slider.get()),
            "p": float(self.p_var.get()),
            "w": float(self.w_var.get()),
            "t0": DEFAULT_T0,
        }

        self.run_btn.config(state="disabled")
        self.status_label.config(text="Processing: 0%")
        self.master.update_idletasks()

        def worker():
            result = dehaze(
                self.image_data,
                show_w=False, show_v=False, show_r=False,
                show_v_r=False, show_t=False, show_j=False,
                adapt_eq=False,
                progress_callback=lambda f: self.master.after(
                    0, lambda frac=f: self._update_progress(frac),
                ),
                **params,
            )
            self.master.after(0, lambda: self._finish_run(result, show_flags, adapt_eq))

        threading.Thread(target=worker, daemon=True).start()

    def _finish_run(self, result, show_flags, adapt_eq):
        from dehaze_algorithm import _to_uint8_gray, disp_img, lab_adapthisteq

        self.dehazed_data = result["J"]
        self._show_bgr_in_axes(self.ax_dehazed, self.dehazed_data, "Dehazed (J)")
        self.status_label.config(text="Done.")
        self.run_btn.config(state="normal")

        if show_flags["W"]:
            disp_img(_to_uint8_gray(result["W"]), title="W")
        if show_flags["V"]:
            disp_img(_to_uint8_gray(result["V"]), title="V")
        if show_flags["R"]:
            disp_img(_to_uint8_gray(result["R"]), title="R")
        if show_flags["V_R"]:
            disp_img(_to_uint8_gray(result["V_R"]), title="V_R")
        if show_flags["t"]:
            disp_img(result["t"], title="depth image t")
        if show_flags["J"]:
            disp_img(result["J"], title="J")
            disp_img(self.image_data, title="Original")
        if adapt_eq:
            disp_img(lab_adapthisteq(result["J"]), title="J_adaptEQ")

        if any(show_flags.values()) or adapt_eq:
            cv2.waitKey(0)
            cv2.destroyAllWindows()


def main():
    root = tk.Tk()
    FastDehazeImageApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
