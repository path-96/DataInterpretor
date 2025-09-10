import tkinter as tk
from typing import Callable, Optional


class AppUI(tk.Frame):
    """UI layer only: widgets and layout.

    Exposes callbacks for user actions and a method to display the
    selected file path. No business logic here.
    """

    def __init__(self, master: tk.Misc, on_select_file: Optional[Callable[[], None]] = None):
        super().__init__(master)

        # Callbacks (wired by main.py)
        self._on_select_file: Callable[[], None] = on_select_file or (lambda: None)
        self._on_test_connection: Callable[[], None] = lambda: None
        self._on_select_csv_dir: Callable[[], None] = lambda: None
        self._on_select_ready_image: Callable[[], None] = lambda: None
        self._on_capture_ready_image: Callable[[], None] = lambda: None
        self._on_clear_ready_region: Callable[[], None] = lambda: None
        self._on_test_ready_image: Callable[[], None] = lambda: None

        # Root layout: left content (col 0) + right log panel (col 1)
        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # --- File selector section ---
        file_frame = tk.LabelFrame(self, text="ﾌｧｲﾙ選択", padx=8, pady=8)
        file_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 6))
        file_frame.grid_columnconfigure(1, weight=1)

        self.file_label = tk.Label(file_frame, text="Path:")
        self.file_label.grid(row=0, column=0, sticky="w", padx=(0, 6))

        self._path_var = tk.StringVar(value="No file selected")
        self.path_entry = tk.Entry(file_frame, textvariable=self._path_var, state="readonly")
        self.path_entry.grid(row=0, column=1, sticky="ew", padx=(0, 6))

        self.select_btn = tk.Button(file_frame, text="Select", width=10, command=self._on_select_file)
        self.select_btn.grid(row=0, column=2, sticky="e")

        # CSV folder selector in File section (row 1)
        tk.Label(file_frame, text="CSV保存先:").grid(row=1, column=0, sticky="w", padx=(0, 6), pady=(8, 0))
        self._csv_dir_var = tk.StringVar(value="")
        self.csv_dir_entry = tk.Entry(file_frame, textvariable=self._csv_dir_var, state="readonly")
        self.csv_dir_entry.grid(row=1, column=1, sticky="ew", padx=(0, 6), pady=(8, 0))
        self.csv_dir_btn = tk.Button(file_frame, text="Select", width=10, command=lambda: self._on_select_csv_dir())
        self.csv_dir_btn.grid(row=1, column=2, sticky="e", pady=(8, 0))

        # Ready Image template selector (row 2)

        # --- Connection section ---
        conn_frame = tk.LabelFrame(self, text="Connection", padx=8, pady=8)
        conn_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 6))
        conn_frame.grid_columnconfigure(1, weight=1)

        # IP input with validation
        self.ip_label = tk.Label(conn_frame, text="IP:")
        self.ip_label.grid(row=0, column=0, sticky="w")

        self._ip_var = tk.StringVar(value="")
        vcmd_ip = (self.register(self._validate_ip_partial), "%P")
        self.ip_entry = tk.Entry(conn_frame, textvariable=self._ip_var, validate="key", validatecommand=vcmd_ip)
        self.ip_entry.grid(row=0, column=1, sticky="ew", padx=(6, 6))

        # Port input with light validation
        self.port_label = tk.Label(conn_frame, text="Port:")
        self.port_label.grid(row=0, column=2, sticky="w")

        self._port_var = tk.StringVar(value="")
        vcmd_port = (self.register(self._validate_port_partial), "%P")
        self.port_entry = tk.Entry(conn_frame, textvariable=self._port_var, width=8, validate="key", validatecommand=vcmd_port)
        self.port_entry.grid(row=0, column=3, sticky="w", padx=(6, 6))

        self.test_btn = tk.Button(conn_frame, text="Test Connection", width=16, command=lambda: self._on_test_connection())
        self.test_btn.grid(row=0, column=4, sticky="e")
        # Hide per request
        try:
            self.test_btn.grid_remove()
        except Exception:
            pass

        # --- Controls (Start/Stop) ---
        controls = tk.LabelFrame(self, text="Controls", padx=8, pady=8)
        controls.grid(row=2, column=0, sticky="ew", padx=10, pady=(0, 10))
        controls.grid_columnconfigure(0, weight=0)
        controls.grid_columnconfigure(1, weight=0)
        controls.grid_columnconfigure(2, weight=1)  # spacer
        controls.grid_columnconfigure(3, weight=0)
        controls.grid_columnconfigure(4, weight=0)
        self.start_btn = tk.Button(controls, text="Start", width=12)
        self.stop_btn = tk.Button(controls, text="Stop", width=12)
        self.settings_btn = tk.Button(controls, text="遅延設定", width=10)
        self.plc_settings_btn = tk.Button(controls, text="PLCｱﾄﾞﾚｽ設定", width=15)
        self.start_btn.grid(row=0, column=0, sticky="w")
        self.stop_btn.grid(row=0, column=1, sticky="w", padx=(8, 0))
        # Hide Start/Stop per request
        try:
            self.start_btn.grid_remove()
            self.stop_btn.grid_remove()
        except Exception:
            pass
        # Move to left side and swap order: PLC Addr then Settings
        self.plc_settings_btn.grid(row=0, column=0, sticky="w")
        # Move Settings button to the right side
        self.settings_btn.grid(row=0, column=4, sticky="e")

        # --- PLC Lamps (Inputs and Outputs) ---
        plc_frame = tk.LabelFrame(self, text="PLC 入力", padx=8, pady=8)
        plc_frame.grid(row=3, column=0, sticky="ew", padx=10, pady=(0, 6))
        plc_frame.grid_columnconfigure(0, weight=1)
        plc_frame.grid_columnconfigure(1, weight=1)

        # Inputs column
        in_frame = tk.Frame(plc_frame)
        in_frame.grid(row=0, column=0, sticky="w")
        tk.Label(in_frame, text="PLC 入力").grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 4))
        self._plc_in_lamps = {}
        in_items = [
            ("input1", "自動運転中"),
            ("input2", "測定開始"),
            ("input3", "予備"),
            ("input4", "予備"),
            ("input5", "予備"),
            ("input6", "予備"),
            ("input7", "予備"),
        ]
        # Ensure Input 8: ﾊｰﾄﾋﾞｰﾄ　is present
        try:
            in_items.append(("input8", "ﾊｰﾄﾋﾞｰﾄ"))
        except Exception:
            pass
        for r, (key, desc) in enumerate(in_items, start=1):
            tk.Label(in_frame, text=f"[{key[-1]}] {desc}").grid(row=r, column=0, sticky="w", padx=(0, 8), pady=2)
            canvas = tk.Canvas(in_frame, width=24, height=24, highlightthickness=0)
            cid = canvas.create_oval(2, 2, 22, 22, outline="#666", fill="#aaa")
            canvas.grid(row=r, column=1, sticky="w")
            self._plc_in_lamps[key] = (canvas, cid)

        # Outputs column
        out_frame = tk.Frame(plc_frame)
        out_frame.grid(row=0, column=1, sticky="w", padx=(20, 0))
        tk.Label(out_frame, text="PC 出力").grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 4))
        self._plc_out_lamps = {}
        out_items = [
            ("output1", "準備OK"),
            ("output2", "測定中"),
            ("output3", "ﾃﾞｰﾀ転送完了"),
            ("output5", "予備"),
            ("output6", "予備"),
            ("output4", "予備"),
            ("output7", "予備"),
        ]
        # Ensure Output 8: ﾊｰﾄﾋﾞｰﾄ　is present
        try:
            out_items.append(("output8", "ﾊｰﾄﾋﾞｰﾄ"))
        except Exception:
            pass
        for r, (key, desc) in enumerate(out_items, start=1):
            tk.Label(out_frame, text=f"[{key[-1]}] {desc}").grid(row=r, column=0, sticky="w", padx=(0, 8), pady=2)
            canvas = tk.Canvas(out_frame, width=24, height=24, highlightthickness=0)
            cid = canvas.create_oval(2, 2, 22, 22, outline="#666", fill="#aaa")
            canvas.grid(row=r, column=1, sticky="w")
            self._plc_out_lamps[key] = (canvas, cid)

        # Program status lamp (non-PLC): 貂ｬ螳壼勣襍ｷ蜍穂ｸｭ
        prog_frame = tk.LabelFrame(plc_frame, text="MarcomProfessional", padx=8, pady=8)
        prog_frame.grid(row=1, column=0, columnspan=2, sticky="w")
        tk.Label(prog_frame, text="測定ｱﾌﾟﾘ起動中").grid(row=0, column=0, sticky="w", padx=(0, 8))
        self._prog_canvas = tk.Canvas(prog_frame, width=24, height=24, highlightthickness=0)
        self._prog_cid = self._prog_canvas.create_oval(2, 2, 22, 22, outline="#666", fill="#aaa")
        self._prog_canvas.grid(row=0, column=1, sticky="w")
        # Right-side Log viewer (top-to-bottom)
        log_frame = tk.LabelFrame(self, text="ﾛｸﾞ表示", padx=8, pady=8)
        log_frame.grid(row=0, column=1, rowspan=5, sticky="nsew", padx=(6, 10), pady=(10, 10))
        log_frame.grid_columnconfigure(0, weight=1)
        log_frame.grid_rowconfigure(1, weight=1)
        # Top row: Clear button aligned right
        self._on_clear_log: Callable[[], None] = lambda: None
        self.clear_log_btn = tk.Button(log_frame, text="ｸﾘｱ", width=10, command=lambda: self._on_clear_log())
        self.clear_log_btn.grid(row=0, column=1, sticky="e")
        # Log text area + scrollbar
        self._log = tk.Text(log_frame, height=10, wrap="none", state="disabled")
        yscroll = tk.Scrollbar(log_frame, orient="vertical", command=self._log.yview)
        self._log.configure(yscrollcommand=yscroll.set)
        self._log.grid(row=1, column=0, sticky="nsew")
        yscroll.grid(row=1, column=1, sticky="ns")

    # --- UI API exposed to main.py ---
    def set_on_select_file(self, callback: Callable[[], None]) -> None:
        self._on_select_file = callback or (lambda: None)
        self.select_btn.configure(command=self._on_select_file)

    def set_selected_file(self, path: str) -> None:
        self._path_var.set(path if path else "ﾌｧｲﾙが選択されていません")

    def get_selected_file(self) -> str:
        return self._path_var.get().strip()

    def get_ip(self) -> str:
        return self._ip_var.get().strip()

    def set_ip(self, ip: str) -> None:
        self._ip_var.set(ip)

    def get_port(self) -> str:
        return self._port_var.get().strip()

    def set_port(self, port: str) -> None:
        self._port_var.set(str(port))

    def set_on_test_connection(self, callback: Callable[[], None]) -> None:
        self._on_test_connection = callback or (lambda: None)

    def set_on_start(self, callback: Callable[[], None]) -> None:
        self.start_btn.configure(command=callback or (lambda: None))

    def set_on_stop(self, callback: Callable[[], None]) -> None:
        self.stop_btn.configure(command=callback or (lambda: None))

    def set_on_open_settings(self, callback: Callable[[], None]) -> None:
        self.settings_btn.configure(command=callback or (lambda: None))

    def set_on_open_plc_settings(self, callback: Callable[[], None]) -> None:
        self.plc_settings_btn.configure(command=callback or (lambda: None))

    def set_on_clear_log(self, callback: Callable[[], None]) -> None:
        self._on_clear_log = callback or (lambda: None)

    # CSV helpers
    def set_on_select_csv_dir(self, callback: Callable[[], None]) -> None:
        self._on_select_csv_dir = callback or (lambda: None)
        try:
            self.csv_dir_btn.configure(command=self._on_select_csv_dir)
        except Exception:
            pass

    def set_csv_dir(self, path: str) -> None:
        self._csv_dir_var.set(path or "")

    def get_csv_dir(self) -> str:
        return self._csv_dir_var.get().strip()

    # Ready image helpers
    def set_on_select_ready_image(self, callback: Callable[[], None]) -> None:
        self._on_select_ready_image = callback or (lambda: None)
        try:
            self.img_btn.configure(command=self._on_select_ready_image)
        except Exception:
            pass
            pass

    # --- Validation helpers (internal) ---
    def _validate_ip_partial(self, value: str) -> bool:
        """Validate IPv4 input while typing.

        Rules:
        - Only digits and up to three dots.
        - At most 4 octets.
        - Each octet 0-255 and max length 3 (when present).
        - Empty is allowed during typing.
        """
        if value == "":
            return True
        if len(value) > 15:  # 255.255.255.255
            return False
        if any(c not in "0123456789." for c in value):
            return False
        parts = value.split(".")
        if len(parts) > 4:
            return False
        for part in parts:
            if part == "":
                continue  # allow incomplete octets like '192.'
            if len(part) > 3:
                return False
            # Leading zeros are allowed; check numeric range
            try:
                num = int(part)
            except ValueError:
                return False
            if num < 0 or num > 255:
                return False
        return True

    def _validate_port_partial(self, value: str) -> bool:
        """Accept only digits and keep range within 1..65535 when present."""
        if value == "":
            return True
        if not value.isdigit():
            return False
        if len(value) > 5:
            return False
        try:
            num = int(value)
        except ValueError:
            return False
        return 0 <= num <= 65535

    # --- Log helpers ---
    def clear_log(self) -> None:
        try:
            self._log.configure(state="normal")
            self._log.delete("1.0", tk.END)
            self._log.configure(state="disabled")
        except Exception:
            pass

    def append_log(self, text: str) -> None:
        try:
            self._log.configure(state="normal")
            self._log.insert(tk.END, (text or "") + "\n")
            self._log.see(tk.END)
            self._log.configure(state="disabled")
        except Exception:
            pass

    def get_log_text(self) -> str:
        try:
            return self._log.get("1.0", tk.END)
        except Exception:
            return ""

    def set_log_text(self, text: str) -> None:
        try:
            self._log.configure(state="normal")
            self._log.delete("1.0", tk.END)
            if text:
                self._log.insert(tk.END, text)
            self._log.configure(state="disabled")
        except Exception:
            pass

    # --- PLC lamps API ---
    def set_plc_input_state(self, key: str, on: bool) -> None:
        w = self._plc_in_lamps.get(key)
        if not w:
            return
        canvas, cid = w
        canvas.itemconfig(cid, fill=("#00ff0d" if on else "#aaaaaa"))

    def set_plc_output_state(self, key: str, on: bool) -> None:
        w = self._plc_out_lamps.get(key)
        if not w:
            return
        canvas, cid = w
        canvas.itemconfig(cid, fill=("#00ff0d" if on else "#aaaaaa"))

    # --- Program status lamp API ---
    def set_instrument_running(self, running: bool) -> None:
        try:
            self._prog_canvas.itemconfig(self._prog_cid, fill=("#00ff0d" if running else "#aaaaaa"))
        except Exception:
            pass

    # Debug binding APIs removed per request
