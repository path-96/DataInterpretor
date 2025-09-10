import tkinter as tk
from tkinter import filedialog, messagebox
import socket
import sys
import ctypes
import os
import json
import csv
import datetime
import struct
import time
import re
import socket as _sock

try:
    import pymcprotocol  # Mitsubishi MC protocol
except Exception:
    pymcprotocol = None  # type: ignore

from ui import AppUI


def main() -> None:
    # Make the process DPI-aware on Windows so Tk scales correctly
    try:
        if sys.platform.startswith("win"):
            try:
                ctypes.windll.shcore.SetProcessDpiAwareness(2)  # Per-monitor DPI aware
            except Exception:
                try:
                    ctypes.windll.user32.SetProcessDPIAware()
                except Exception:
                    pass
    except Exception:
        pass

    root = tk.Tk()
    # Align Tk internal scaling with actual DPI (prevents shrink on high-DPI)
    try:
        ppi = float(root.winfo_fpixels("1i"))  # pixels per inch
        root.tk.call("tk", "scaling", ppi / 72.0)
    except Exception:
        pass
    root.title("測定ﾃﾞｰﾀ転送ｱﾌﾟﾘ")
    root.geometry("900x700")
    root.resizable(True, True)

    # Keep socket operations snappy to avoid UI freeze on disconnects
    try:
        # Keep network operations snappy so UI stays responsive
        _sock.setdefaulttimeout(0.8)
    except Exception:
        pass

    # Build UI (no logic bound yet)
    ui = AppUI(root)
    ui.pack(fill="both", expand=True)
    # Lock in a minimum size after layout so the window does not shrink
    try:
        root.update_idletasks()
        root.minsize(700, 700)
        ui.configure(width=900, height=700)
        ui.pack_propagate(False)
        root.geometry("900x700")
        root.after(50, lambda: (root.minsize(700, 700), root.geometry("900x700")))
    # Settings persistence
    except Exception:
        pass
    # Resolve application directory (works for frozen EXE and source)
    try:
        if getattr(sys, "frozen", False):
            app_dir = os.path.dirname(sys.executable)
        else:
            app_dir = os.path.dirname(os.path.abspath(__file__))
    except Exception:
        app_dir = os.getcwd()
    settings_path = os.path.join(app_dir, "app_settings.json")
    READY_IMAGE_PATH = os.path.join(app_dir, "image.png")
    config = {
        "f3_delay_ms": 300,
        "clear_delay_ms": 0,
        "ready_image_path": "image.png",
        "ready_search_region": [],  # [x, y, w, h] in screen coords
        "ready_image_scale": 1.0,
        # PLC connection options
        "plc_type": "Q",            # One of: "Q", "L", "QnA", "iQ-L", "iQ-R"
        "plc_timeout_sec": 3,        # MC protocol timeout (sec); socket = timeout+1
        "plc_addresses": {
            "inputs": {f"input{i}": "" for i in range(1, 9)},
            "outputs": {f"output{i}": "" for i in range(1, 9)},
            "value_output": "",
        },
    }

    def save_settings() -> None:
        data = {
            "ip": ui.get_ip(),
            "port": ui.get_port(),
            "file_path": ("" if ui.get_selected_file().lower() == "no file selected" else ui.get_selected_file()),
            "csv_dir": ui.get_csv_dir(),
            "f3_delay_ms": int(config.get("f3_delay_ms", 300)),
            "clear_delay_ms": int(config.get("clear_delay_ms", 0)),
            "ready_image_path": config.get("ready_image_path", "image.png"),
            "ready_image_scale": float(config.get("ready_image_scale", 1.0)),
            "ready_search_region": list(config.get("ready_search_region", [])),
            "plc_type": str(config.get("plc_type", "Q")),
            "plc_timeout_sec": int(config.get("plc_timeout_sec", 3)),
            "plc_addresses": config.get("plc_addresses", {}),
        }
        try:
            with open(settings_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception:
            pass  # silent

    def load_settings() -> None:
        try:
            with open(settings_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return
        ui.set_ip(data.get("ip", ""))
        ui.set_port(str(data.get("port", "")))
        file_path = data.get("file_path", "")
        if file_path:
            ui.set_selected_file(file_path)
        
        csv_dir = data.get("csv_dir", "")
        if isinstance(csv_dir, str):
            ui.set_csv_dir(csv_dir)
        # load config
        try:
            config["f3_delay_ms"] = int(data.get("f3_delay_ms", config["f3_delay_ms"]))
        except Exception:
            pass
        try:
            config["clear_delay_ms"] = int(data.get("clear_delay_ms", config["clear_delay_ms"]))
        except Exception:
            pass
        rip = data.get("ready_image_path")
        if isinstance(rip, str) and rip:
            config["ready_image_path"] = rip
        try:
            rscale = float(data.get("ready_image_scale", 1.0))
            if rscale <= 0:
                rscale = 1.0
            config["ready_image_scale"] = rscale
        except Exception:
            config["ready_image_scale"] = 1.0
        try:
            reg = data.get("ready_search_region")
            if isinstance(reg, (list, tuple)) and len(reg) == 4:
                x, y, w, h = reg
                if all(isinstance(v, (int, float)) for v in (x, y, w, h)) and w > 0 and h > 0:
                    config["ready_search_region"] = [int(x), int(y), int(w), int(h)]
        except Exception:
            pass
        pt = data.get("plc_type")
        if isinstance(pt, str) and pt:
            config["plc_type"] = pt
        try:
            pto = int(data.get("plc_timeout_sec", config.get("plc_timeout_sec", 3)))
            config["plc_timeout_sec"] = max(1, min(30, pto))
        except Exception:
            pass
        plc = data.get("plc_addresses")
        if isinstance(plc, dict):
            config["plc_addresses"].get("inputs", {}).update(plc.get("inputs", {}))
            config["plc_addresses"].get("outputs", {}).update(plc.get("outputs", {}))
            if "value_output" in plc:
                config["plc_addresses"]["value_output"] = plc.get("value_output", "")

    # Logic: handle file selection and update UI
    def on_select_file() -> None:
        path = filedialog.askopenfilename(title="Select a file", filetypes=[("All files", "*.*")])
        if path:
            ui.set_selected_file(path)
            save_settings()
    
    # Ready image selection: store path in settings
    def on_select_ready_image() -> None:
        path = filedialog.askopenfilename(title="Select Ready Image", filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif"), ("All files", "*.*")])
        if path:
            try:
                config["ready_image_path"] = path
            except Exception:
                pass
            save_settings()

    # CSV directory select
    def on_select_csv_dir() -> None:
        path = filedialog.askdirectory(title="Select CSV folder")
        if path:
            ui.set_csv_dir(path)
            save_settings()

    # Connection test logic
    def on_test_connection() -> None:
        ip = ui.get_ip()
        port_str = ui.get_port()

        if not ip or not port_str:
            messagebox.showwarning("Missing Info", "Please enter IP and Port.")
            return

        try:
            port = int(port_str)
            if not (1 <= port <= 65535):
                raise ValueError
        except ValueError:
            messagebox.showerror("Invalid Port", "Port must be an integer 1-65535.")
            return

        try:
            with socket.create_connection((ip, port), timeout=3):
                pass
        except Exception as e:
            messagebox.showerror("Connection Failed", f"Could not connect to {ip}:{port}\n{e}")
        else:
            messagebox.showinfo("Connection OK", f"Connected to {ip}:{port}")

    # Start/Stop logic
    def on_test_ready_image() -> None:
        try:
            found = _is_ready_image_present()
            try:
                if hasattr(ui, "set_instrument_running"):
                    ui.set_instrument_running(found)
            except Exception:
                pass
            if found:
                messagebox.showinfo("Ready Image", f"Found: {_get_ready_image_path()} (scale={config.get('ready_image_scale', 1.0)})")
            else:
                # Attempt auto-calibration by scanning scales
                try:
                    import cv2  # type: ignore
                    import numpy as np  # type: ignore
                    from PIL import ImageGrab  # type: ignore
                except Exception:
                    messagebox.showinfo("Ready Image", "Not found. Install dependencies for calibration: opencv-python, pillow, numpy")
                    return

                path = _get_ready_image_path()
                if not os.path.exists(path):
                    messagebox.showinfo("Ready Image", f"Not found. Missing file: {path}")
                    return
                try:
                    # Respect region if set
                    bbox = None
                    reg = config.get("ready_search_region", [])
                    if isinstance(reg, (list, tuple)) and len(reg) == 4 and reg[2] > 0 and reg[3] > 0:
                        bbox = (int(reg[0]), int(reg[1]), int(reg[0]+reg[2]), int(reg[1]+reg[3]))
                    screenshot = ImageGrab.grab(bbox=bbox) if bbox else ImageGrab.grab()
                    scr = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
                    tmpl0 = cv2.imread(path, cv2.IMREAD_COLOR)
                    if tmpl0 is None:
                        messagebox.showinfo("Ready Image", f"Not found. Could not read template: {path}")
                        return
                    best = (0.0, 1.0)  # (score, scale)
                    for scale in [x/100.0 for x in range(60, 141, 5)]:  # 0.60 .. 1.40
                        h, w = tmpl0.shape[:2]
                        nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
                        if nw < 2 or nh < 2:
                            continue
                        tmpl = cv2.resize(tmpl0, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR)
                        try:
                            res = cv2.matchTemplate(scr, tmpl, cv2.TM_CCOEFF_NORMED)
                            _, score, _, _ = cv2.minMaxLoc(res)
                            if score > best[0]:
                                best = (score, scale)
                        except Exception:
                            pass
                    score, scale = best
                    if score >= 0.70:
                        config["ready_image_scale"] = float(scale)
                        save_settings()
                        # Re-check using the new scale
                        found2 = _is_ready_image_present()
                        try:
                            if hasattr(ui, "set_instrument_running"):
                                ui.set_instrument_running(found2)
                        except Exception:
                            pass
                        if found2:
                            messagebox.showinfo("Ready Image", f"Found after calibration (scale={scale:.2f}, score={score:.2f}).")
                            return
                    messagebox.showinfo("Ready Image", f"Not found. Best score={score:.2f} at scale={scale:.2f}. Try re-capturing the template at this size.")
                    return
                except Exception as e:
                    messagebox.showerror("Ready Image", f"Calibration error:\n{e}")
                    return
        except Exception as e:
            messagebox.showerror("Ready Image", f"Error while checking image:\n{e}")

    def on_capture_ready_image() -> None:
        # Fullscreen overlay to drag a capture region; saves template and region
        try:
            from PIL import ImageGrab  # type: ignore
        except Exception as e:
            messagebox.showerror("Capture Ready", f"Pillow not available: {e}")
            return

        overlay = tk.Toplevel(root)
        overlay.attributes("-topmost", True)
        try:
            overlay.attributes("-fullscreen", True)
        except Exception:
            sw = root.winfo_screenwidth()
            sh = root.winfo_screenheight()
            overlay.geometry(f"{sw}x{sh}+0+0")
        overlay.attributes("-alpha", 0.25)
        overlay.configure(bg="#000000")
        overlay.overrideredirect(True)
        overlay.grab_set()

        canvas = tk.Canvas(overlay, bg="", highlightthickness=0, cursor="cross")
        canvas.pack(fill="both", expand=True)

        state = {"x0": 0, "y0": 0, "rect": None}

        def done_with_bbox(x1, y1, x2, y2):
            overlay.destroy()
            if x2 - x1 < 5 or y2 - y1 < 5:
                messagebox.showwarning("Capture Ready", "Selection too small. Try again.")
                return
            try:
                img = ImageGrab.grab(bbox=(x1, y1, x2, y2))
                # Save template to configured path
                path = _get_ready_image_path()
                os.makedirs(os.path.dirname(path), exist_ok=True)
                img.save(path)
                # Save region
                config["ready_search_region"] = [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]
                config["ready_image_scale"] = 1.0
                save_settings()
                # Immediate test
                found = _is_ready_image_present()
                if hasattr(ui, "set_instrument_running"):
                    ui.set_instrument_running(found)
                if found:
                    messagebox.showinfo("Capture Ready", f"Saved and detected at {path}.")
                else:
                    messagebox.showinfo("Capture Ready", f"Saved. Not detected yet. Try Test Ready.")
            except Exception as e:
                messagebox.showerror("Capture Ready", f"Failed to capture/save:\n{e}")

        def on_press(e):
            state["x0"], state["y0"] = e.x, e.y
            if state["rect"]:
                canvas.delete(state["rect"])
            state["rect"] = canvas.create_rectangle(e.x, e.y, e.x, e.y, outline="#ffff00", width=2)

        def on_motion(e):
            if state["rect"]:
                canvas.coords(state["rect"], state["x0"], state["y0"], e.x, e.y)

        def on_release(e):
            x1, y1 = state["x0"], state["y0"]
            x2, y2 = e.x, e.y
            if x2 < x1:
                x1, x2 = x2, x1
            if y2 < y1:
                y1, y2 = y2, y1
            done_with_bbox(int(x1), int(y1), int(x2), int(y2))

        def on_escape(e):
            try:
                overlay.destroy()
            except Exception:
                pass

        canvas.bind("<ButtonPress-1>", on_press)
        canvas.bind("<B1-Motion>", on_motion)
        canvas.bind("<ButtonRelease-1>", on_release)
        overlay.bind("<Escape>", on_escape)
    def press_f3_windows() -> None:
        if not sys.platform.startswith("win"):
            messagebox.showerror("Unsupported", "F3 simulation works on Windows only.")
            return
        try:
            user32 = ctypes.windll.user32
            KEYEVENTF_KEYUP = 0x0002
            VK_F3 = 0x72
            user32.keybd_event(VK_F3, 0, 0, 0)  # key down
            user32.keybd_event(VK_F3, 0, KEYEVENTF_KEYUP, 0)  # key up
        except Exception as e:
            messagebox.showerror("Key Press Failed", f"Could not send F3: {e}")

    # PLC monitoring state and helpers
    plc = None
    plc_job = None
    plc_err_suppressed = False
    last_input_states = {f"input{i}": None for i in range(1, 9)}  # track changes
    last_output_states = {f"output{i}": None for i in range(1, 9)}
    trigger_busy = False
    instrument_ready = False
    measure_start_t = None  # time when input1 turned ON
    _img_check_cache = {"t": 0.0, "val": False}

    def _get_ready_image_path() -> str:
        p = str(config.get("ready_image_path", "image.png") or "image.png")
        if os.path.isabs(p):
            return p
        # Prefer app folder
        p1 = os.path.join(app_dir, p)
        if os.path.exists(p1):
            return p1
        # Fallback to current working directory
        p2 = os.path.join(os.getcwd(), p)
        if os.path.exists(p2):
            return p2
        # Last resort: default in app folder
        return os.path.join(app_dir, "image.png")

    def _is_ready_image_present() -> bool:
        path = _get_ready_image_path()
        now = time.monotonic()
        if now - _img_check_cache.get("t", 0.0) < 1.0:
            return bool(_img_check_cache.get("val", False))
        ok = False
        try:
            # Quick path sanity
            if not os.path.exists(path):
                raise FileNotFoundError(path)
            try:
                import cv2  # type: ignore
                import numpy as np  # type: ignore
                from PIL import ImageGrab  # type: ignore
                # Limit capture to configured region if available
                bbox = None
                try:
                    reg = config.get("ready_search_region", [])
                    if isinstance(reg, (list, tuple)) and len(reg) == 4:
                        x, y, w, h = reg
                        if w > 0 and h > 0:
                            bbox = (int(x), int(y), int(x + w), int(y + h))
                except Exception:
                    bbox = None
                screenshot = ImageGrab.grab(bbox=bbox) if bbox else ImageGrab.grab()
                scr = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
                tmpl = cv2.imread(path, cv2.IMREAD_COLOR)
                if tmpl is not None and scr is not None:
                    # Apply configured scale if needed
                    try:
                        scale = float(config.get("ready_image_scale", 1.0))
                    except Exception:
                        scale = 1.0
                    if scale and abs(scale - 1.0) > 1e-3:
                        h, w = tmpl.shape[:2]
                        nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
                        if nw > 1 and nh > 1:
                            tmpl = cv2.resize(tmpl, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR)
                    # If region size equals template, try direct compare first
                    if scr.shape[:2] == tmpl.shape[:2]:
                        try:
                            diff = cv2.absdiff(scr, tmpl)
                            score = 1.0 - float(cv2.mean(diff)[0] / 255.0)
                            if score >= 0.98:
                                ok = True
                        except Exception:
                            pass
                    if not ok:
                        res = cv2.matchTemplate(scr, tmpl, cv2.TM_CCOEFF_NORMED)
                        _, max_val, _, _ = cv2.minMaxLoc(res)
                        ok = max_val >= 0.70
                    if not ok:
                        # Grayscale fallback for minor color/AA differences
                        try:
                            scr_g = cv2.cvtColor(scr, cv2.COLOR_BGR2GRAY)
                            tmpl_g = cv2.cvtColor(tmpl, cv2.COLOR_BGR2GRAY)
                            res2 = cv2.matchTemplate(scr_g, tmpl_g, cv2.TM_CCOEFF_NORMED)
                            _, max_val2, _, _ = cv2.minMaxLoc(res2)
                            ok = max_val2 >= 0.70
                        except Exception:
                            pass
            except Exception:
                pass
            if not ok:
                try:
                    import pyautogui  # type: ignore
                    box = pyautogui.locateOnScreen(path, confidence=5.0)
                    ok = bool(box)
                except Exception:
                    ok = False
        except Exception:
            ok = False
        _img_check_cache["t"] = now
        _img_check_cache["val"] = ok
        return ok

    def _disconnect_plc() -> None:
        nonlocal plc
        if plc is not None:
            try:
                plc.close()
            except Exception:
                pass
            plc = None

    def _ensure_plc_connected() -> bool:
        nonlocal plc, plc_err_suppressed
        if pymcprotocol is None:
            # Library missing; don't block UI with modal
            plc_err_suppressed = True
            return False
        if plc is not None:
            return True
        # Read connection params on UI thread only
        ip = ui.get_ip().strip()
        port_s = ui.get_port().strip()
        try:
            port_i = int(port_s)
        except Exception:
            plc_err_suppressed = True
            return False
        # Fast pre-check to avoid long hangs; if it fails, still try MC connect once
        precheck_ok = True
        try:
            with socket.create_connection((ip, port_i), timeout=1.5):
                pass
        except Exception:
            precheck_ok = False
        # Use a more generous timeout for the actual MC protocol connect
        old_to = _sock.getdefaulttimeout()
        try:
            _sock.setdefaulttimeout(float(max(1, int(config.get("plc_timeout_sec", 3))) + 1))
            # Instantiate with the configured PLC type
            client = pymcprotocol.Type3E(plctype=str(config.get("plc_type", "Q")))
            # Configure MC protocol timer/socket timeouts via library API
            try:
                client.setaccessopt(timer_sec=int(config.get("plc_timeout_sec", 3)))
            except Exception:
                pass
            # If the quick precheck failed, this connect may still succeed on slow networks
            client.connect(ip, port_i)
            plc = client
            plc_err_suppressed = False
            return True
        except Exception:
            _disconnect_plc()
            plc_err_suppressed = True
            return False
        finally:
            try:
                _sock.setdefaulttimeout(old_to)
            except Exception:
                pass

    plc_poll_inflight = False

    def _poll_plc_inputs() -> None:
        nonlocal plc_job, plc, trigger_busy, plc_poll_inflight
        if plc_poll_inflight:
            plc_job = root.after(500, _poll_plc_inputs)
            return
        plc_poll_inflight = True

        def _bit_on(v) -> bool:
            try:
                if isinstance(v, (list, tuple)) and len(v) > 0:
                    v = v[0]
                if isinstance(v, (int, bool)):
                    return bool(int(v))
                if isinstance(v, str):
                    s = v.strip().lower()
                    if s in ("1", "on", "true", "t"):
                        return True
                    if s in ("0", "off", "false", "f"):
                        return False
                    return bool(int(float(s)))
            except Exception:
                pass
            return False

        def worker():
            res_inputs = {}
            res_outputs = {}
            ok = _ensure_plc_connected()
            if ok:
                try:
                    inputs = config.get("plc_addresses", {}).get("inputs", {})
                    for i in range(1, 9):
                        key = f"input{i}"
                        addr = (inputs.get(key) or "").strip()
                        if not addr:
                            res_inputs[key] = False
                            continue
                        try:
                            vals = plc.batchread_bitunits(headdevice=addr, readsize=1)
                            res_inputs[key] = _bit_on(vals)
                        except Exception:
                            res_inputs[key] = False
                            _disconnect_plc()
                            break
                    outputs_cfg = config.get("plc_addresses", {}).get("outputs", {})
                    for okey, addr in outputs_cfg.items():
                        a = (addr or "").strip()
                        if not a:
                            res_outputs[okey] = False
                            continue
                        try:
                            vals = plc.batchread_bitunits(headdevice=a, readsize=1)
                            res_outputs[okey] = _bit_on(vals)
                        except Exception:
                            res_outputs[okey] = False
                            _disconnect_plc()
                            break
                except Exception:
                    pass
            # Apply results on UI thread
            # Perform image detection off the UI thread to avoid freezes
            ready_now = None
            try:
                ready_now = _is_ready_image_present()
            except Exception:
                ready_now = None

            def apply():
                nonlocal plc_poll_inflight, trigger_busy, instrument_ready, measure_start_t
                # Update lamps and trigger logic
                for k, v in res_inputs.items():
                    ui.set_plc_input_state(k, v)
                    prev = last_input_states.get(k)
                    last_input_states[k] = v
                    # Start measurement timing on Input2 rising edge (trigger)
                    if k == "input2" and v and not prev:
                        try:
                            import time as _t
                            measure_start_t = _t.perf_counter()
                            try:
                                print("[MEASURE] Input2 rising: timer started", flush=True)
                            except Exception:
                                pass
                        except Exception:
                            measure_start_t = None
                    if k == "input2" and v and not prev and not trigger_busy:
                        sel = ui.get_selected_file()
                        if not sel or sel.lower() == "no file selected" or not os.path.exists(sel):
                            pass  # avoid modal on timer
                        else:
                            trigger_busy = True

                            _update_outputs_for_processing(True)
                            press_f3_windows()
                            root.after(int(config.get("f3_delay_ms", 300)), lambda p=sel: read_and_clear_file(p, on_done=_on_processing_done))
                for k, v in res_outputs.items():
                    ui.set_plc_output_state(k, v)
                    last_output_states[k] = v
                # Heartbeat: mirror input8 to output8
                try:
                    hb_in = res_inputs.get("input8")
                    if hb_in is not None:
                        outputs_cfg = config.get("plc_addresses", {}).get("outputs", {})
                        hb_addr = (outputs_cfg.get("output8") or "").strip()
                        if hb_addr:
                            current = last_output_states.get("output8")
                            if current is None or bool(current) != bool(hb_in):
                                if _write_output(hb_addr, bool(hb_in)):
                                    last_output_states["output8"] = bool(hb_in)
                                    ui.set_plc_output_state("output8", bool(hb_in))
                except Exception:
                    pass
                # Update program image state and lamp (use background result)
                try:
                    if ready_now is not None:
                        instrument_ready = bool(ready_now)
                    if hasattr(ui, "set_instrument_running"):
                        ui.set_instrument_running(bool(instrument_ready))
                except Exception:
                    pass
                _update_outputs_for_processing(trigger_busy)
                plc_poll_inflight = False
                # schedule next poll
                nonlocal plc_job
                plc_job = root.after(500, _poll_plc_inputs)

            root.after(0, apply)

        import threading
        t = threading.Thread(target=worker, daemon=True)
        t.start()

    def _on_processing_done() -> None:
        nonlocal trigger_busy
        trigger_busy = False
        _update_outputs_for_processing(False)

    def _write_output(addr: str, state: bool) -> bool:
        if not addr:
            return False
        if not _ensure_plc_connected():
            return False
        try:
            # Write single bit as 1 or 0
            plc.batchwrite_bitunits(headdevice=addr, values=[1 if state else 0])
            return True
        except Exception:
            _disconnect_plc()
            return False

    def _pulse_output(key: str, duration_ms: int = 300) -> None:
        nonlocal measure_start_t
        outputs = config.get("plc_addresses", {}).get("outputs", {})
        addr = (outputs.get(key) or "").strip()
        if not addr:
            return
        if _write_output(addr, True):
            last_output_states[key] = True
            ui.set_plc_output_state(key, True)
            # If Output3 (send complete) turns ON, log measurement time since Input1
            if key == "output3":
                try:
                    import time as _t
                    if measure_start_t is not None:
                        elapsed_ms = int(round((_t.perf_counter() - measure_start_t) * 1000.0))
                        try:
                            print(f"[MEASURE] 測定時間: {elapsed_ms} ms (via output3)")
                        except Exception:
                            pass
                        measure_start_t = None
                    else:
                        try:
                            ui.append_log(line_to_log)
                        except Exception:
                            pass
                except Exception:
                    pass
            root.after(duration_ms, lambda: (_write_output(addr, False), ui.set_plc_output_state(key, False), last_output_states.__setitem__(key, False)))

    def _update_outputs_for_processing(processing: bool) -> None:
        outputs = config.get("plc_addresses", {}).get("outputs", {})
        plc_connected = plc is not None
        img_ready = bool(instrument_ready)
        out2_want = True if processing else False
        # PC OUT 1: ON only when PLC connected AND image detected AND not processing
        out1_want = bool(plc_connected and img_ready and not processing)
        desired = {
            "output2": out2_want,
            "output1": out1_want,
        }
        for key, want in desired.items():
            current = last_output_states.get(key)
            if current is want:
                continue
            addr = (outputs.get(key) or "").strip()
            if addr:
                ok = _write_output(addr, want)
                if ok:
                    last_output_states[key] = want
                    ui.set_plc_output_state(key, want)

    def _write_scaled_int_to_dword(addr: str, ivalue: int) -> None:
        if not addr:
            return
        if not _ensure_plc_connected():
            return

        # Validate device type: must be a word/register device (e.g., D, W, R, ZR, SD, SW)
        m = re.match(r"^([A-Za-z]+)(\d+)$", addr.strip())
        if not m:
            return
        dev, num_str = m.group(1).upper(), m.group(2)
        word_devices = {"D", "W", "R", "ZR", "SD", "SW", "ZRD", "ZRW"}
        if dev not in word_devices:
            return

        # Normalize ivalue to unsigned 32-bit and compute words (LE)
        if ivalue < 0:
            ivalue = (ivalue + (1 << 32)) & 0xFFFFFFFF
        else:
            ivalue = ivalue & 0xFFFFFFFF
        w0 = ivalue & 0xFFFF
        w1 = (ivalue >> 16) & 0xFFFF
        try:
            print(f"[PLC] Preparing DWORD write: addr={addr}, value={ivalue} (0x{ivalue:08X}), low=0x{w0:04X}, high=0x{w1:04X}")
        except Exception:
            pass

        # pymcprotocol encodes word values as signed shorts by default.
        # Convert 0..65535 values to their signed 16-bit equivalents so that
        # values > 32767 are transmitted correctly using two's complement.
        def _to_s16(v: int) -> int:
            v &= 0xFFFF
            return v if v <= 0x7FFF else v - 0x10000

        w0_s = _to_s16(w0)
        w1_s = _to_s16(w1)

        # Always write as two separate word writes: base (e.g., D10) then next (e.g., D11)
        try:
            m2 = re.match(r"^([A-Za-z]+)(\d+)$", addr.strip())
            if not m2:
                raise ValueError("Bad base address")
            dev2, base_str = m2.group(1).upper(), m2.group(2)
            base = int(base_str)
            addr0 = f"{dev2}{base}"
            addr1 = f"{dev2}{base+1}"
        except Exception:
            _disconnect_plc()
            return

        # Preferred: write both words in a single command
        try:
            plc.batchwrite_wordunits(headdevice=addr0, values=[w0_s, w1_s])
            return
        except Exception:
            # Fallback: try random dword write
            try:
                sval = ivalue if ivalue <= 0x7FFFFFFF else ivalue - 0x100000000
                plc.randomwrite(word_devices=[], word_values=[], dword_devices=[addr0], dword_values=[sval])
                return
            except Exception:
                _disconnect_plc()

    def _verify_dword(addr: str) -> None:
        # Read-back disabled per request
        return

    def _write_dword_random(addr: str, ivalue: int) -> None:
        if not addr:
            print("[PLC] No value_output address configured; skip write.")
            return
        if not _ensure_plc_connected():
            print("[PLC] Not connected; skip write.")
            return
        try:
            # Normalize device like D10 -> base word address
            m2 = re.match(r"^([A-Za-z]+)(\d+)$", addr.strip())
            if not m2:
                print(f"[PLC] Bad address '{addr}' for DWORD write.")
                return
            dev2, base_str = m2.group(1).upper(), m2.group(2)
            base = int(base_str)
            addr0 = f"{dev2}{base}"
            # Normalize to unsigned then convert to signed 32-bit for library
            if ivalue < 0:
                ivalue = (ivalue + (1 << 32)) & 0xFFFFFFFF
            else:
                ivalue = ivalue & 0xFFFFFFFF
            sval = ivalue if ivalue <= 0x7FFFFFFF else ivalue - 0x100000000
            print(f"[PLC] randomwrite dword -> device={addr0}, value={sval} (orig 0x{ivalue:08X})")
            plc.randomwrite(word_devices=[], word_values=[], dword_devices=[addr0], dword_values=[sval])
        except Exception as e:
            print(f"[PLC] randomwrite dword failed: {e}")
            try:
                _disconnect_plc()
            except Exception:
                pass
            return

    def _start_plc_monitoring() -> None:
        nonlocal plc_job
        if plc_job is None:
            plc_job = root.after(0, _poll_plc_inputs)

    def _stop_plc_monitoring() -> None:
        nonlocal plc_job
        if plc_job is not None:
            try:
                root.after_cancel(plc_job)
            except Exception:
                pass
            plc_job = None
        _disconnect_plc()
        for i in range(1, 9):
            ui.set_plc_input_state(f"input{i}", False)

    # Read whole file then clear
    def read_and_clear_file(path: str, on_done=None) -> None:
        nonlocal measure_start_t
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
        except Exception as e:
            messagebox.showerror("Read Failed", f"Could not read file:\n{e}")
            return

        # Logging removed per request

        # After logging, parse last value, scale by 10000, and send as DWORD; if empty, send 99999
        try:
            last_line = ""
            for ln in content.splitlines():
                if ln.strip():
                    last_line = ln.strip()
            # Prepare last value for logging; append later together with measurement time
            line_to_log = last_line if last_line else "NG"
            # Also write the same log line to daily CSV if a folder is configured
            folder = ui.get_csv_dir()
            if folder:
                try:
                    os.makedirs(folder, exist_ok=True)
                    filename = datetime.date.today().strftime("%Y-%m-%d") + ".csv"
                    fpath = os.path.join(folder, filename)
                    rows = []
                    if os.path.exists(fpath) and os.path.getsize(fpath) > 0:
                        with open(fpath, "r", encoding="utf-8", newline="") as rf:
                            rows = list(csv.reader(rf))
                    if not rows:
                        rows = [["timestamp", "value"]]
                    # Fill next empty value cell if present, else append
                    if len(rows) > 1 and (len(rows[-1]) < 2 or not str(rows[-1][1]).strip()):
                        # Keep existing timestamp or insert current time
                        if not rows[-1] or len(rows[-1]) == 0 or not str(rows[-1][0]).strip():
                            rows[-1] = [datetime.datetime.now().strftime("%H:%M:%S"), line_to_log]
                        else:
                            if len(rows[-1]) == 1:
                                rows[-1].append(line_to_log)
                            else:
                                rows[-1][1] = line_to_log
                    else:
                        ts = datetime.datetime.now().strftime("%H:%M:%S")
                        rows.append([ts, line_to_log])
                    with open(fpath, "w", encoding="utf-8", newline="") as wf:
                        w = csv.writer(wf)
                        w.writerows(rows)
                except Exception:
                    pass
            if _ensure_plc_connected():
                if not last_line:
                    value_to_send = 99999
                else:
                    m = re.search(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", last_line)
                    if m:
                        try:
                            f = float(m.group(0))
                            value_to_send = int(round(f * 10000.0))
                        except Exception:
                            value_to_send = 99999
                    else:
                        value_to_send = 99999
                try:
                    vo_addr = config["plc_addresses"].get("value_output", "")
                except Exception:
                    vo_addr = ""
                print(f"[PLC] Send value -> parsed={last_line!r}, scaled={value_to_send}, addr={vo_addr}")
                _write_dword_random(vo_addr, value_to_send)
                # Log measurement time even if output3 pulse fails
                try:
                    if measure_start_t is not None:
                        elapsed_ms = int(round((time.perf_counter() - measure_start_t) * 1000.0))
                        try:
                            print(f"[MEASURE] 測定時間: {elapsed_ms} ms (post-write)")
                        except Exception:
                            pass
                        ui.append_log(f"{line_to_log}  -  測定時間: {elapsed_ms} ms")
                        measure_start_t = None
                except Exception:
                    pass
                print("[PLC] PULSE output3 (send complete)")
                _pulse_output("output3")
        except Exception:
            pass

        def do_clear() -> None:
            try:
                with open(path, "w", encoding="utf-8") as f:
                    f.write("")
                    f.flush()
                    os.fsync(f.fileno())
            except Exception as e:
                messagebox.showerror("Clear Failed", f"Could not clear file:\n{e}")
                return
            # Persist settings including new log content
            save_settings()
            if on_done:
                try:
                    on_done()
                except Exception:
                    pass

        delay = int(config.get("clear_delay_ms", 0))
        if delay > 0:
            root.after(delay, do_clear)
        else:
            do_clear()

    def on_start() -> None:
        path = ui.get_selected_file()
        if not path or path.lower() == "no file selected":
            messagebox.showwarning("No File", "Please select a text file first.")
            return
        if not os.path.exists(path):
            messagebox.showerror("Missing File", f"File not found:\n{path}")
            return
        _start_plc_monitoring()
        press_f3_windows()
        # Give external writer a configurable delay, then read and clear
        root.after(int(config.get("f3_delay_ms", 300)), lambda p=path: read_and_clear_file(p))

    def on_stop() -> None:
        # No extra log text per request
        _stop_plc_monitoring()
        save_settings()

    # Bind logic to UI
    ui.set_on_select_file(on_select_file)
    ui.set_on_select_csv_dir(on_select_csv_dir)
    # Debug buttons removed; no binding
    # Buttons removed: do not bind test/start/stop
    ui.set_on_clear_log(lambda: (ui.clear_log(), save_settings()))

    # Settings dialog
    def open_settings_dialog() -> None:
        dlg = tk.Toplevel(root)
        dlg.title("Settings")
        dlg.resizable(False, False)
        dlg.transient(root)
        dlg.grab_set()

        frm = tk.Frame(dlg, padx=12, pady=12)
        frm.pack(fill="both", expand=True)

        # F3 delay
        tk.Label(frm, text="測定開始後ﾃﾞｰﾀ取得遅延 (ms):").grid(row=0, column=0, sticky="w")
        f3_var = tk.StringVar(value=str(int(config.get("f3_delay_ms", 300))))
        f3_entry = tk.Spinbox(frm, from_=0, to=60000, increment=50, textvariable=f3_var, width=8)
        f3_entry.grid(row=0, column=1, sticky="w", padx=(8, 0))

        # Clear delay
        tk.Label(frm, text="ﾃﾞｰﾀ転送後ﾃｷｽﾄ削除遅延(ms):").grid(row=1, column=0, sticky="w", pady=(8, 0))
        clr_var = tk.StringVar(value=str(int(config.get("clear_delay_ms", 0))))
        clr_entry = tk.Spinbox(frm, from_=0, to=60000, increment=50, textvariable=clr_var, width=8)
        clr_entry.grid(row=1, column=1, sticky="w", padx=(8, 0), pady=(8, 0))

        # Buttons
        btns = tk.Frame(frm)
        btns.grid(row=2, column=0, columnspan=2, sticky="e", pady=(12, 0))
        def on_save() -> None:
            try:
                config["f3_delay_ms"] = max(0, int(f3_var.get()))
            except Exception:
                config["f3_delay_ms"] = 300
            try:
                config["clear_delay_ms"] = max(0, int(clr_var.get()))
            except Exception:
                config["clear_delay_ms"] = 0
            # Reconnect PLC next cycle in case IP/port changed
            try:
                _disconnect_plc()
            except Exception:
                pass
            save_settings()
            dlg.destroy()
        tk.Button(btns, text="保存", width=10, command=on_save).pack(side="right", padx=(8,0))
        tk.Button(btns, text="ｷｬﾝｾﾙ", width=10, command=dlg.destroy).pack(side="right")

    ui.set_on_open_settings(open_settings_dialog)

    # PLC address settings dialog
    def open_plc_settings_dialog() -> None:
        dlg = tk.Toplevel(root)
        dlg.title("PLC ｱﾄﾞﾚｽ設定")
        dlg.resizable(False, True)
        dlg.transient(root)
        dlg.grab_set()

        frm = tk.Frame(dlg, padx=12, pady=12)
        frm.pack(fill="both", expand=True)

        # Descriptions
        input_desc = {
            "input1": "自動運転中",
            "input2": "測定開始",
            "input3": "予備",
            "input5": "予備",
            "input4": "予備",
            "input6": "予備",
            "input7": "予備",
        }
        output_desc = {
            "output1": "準備OK",
            "output2": "測定中",
            "output3": "ﾃﾞｰﾀ転送完了",
            "output4": "予備",
            "output5": "予備",
            "output6": "予備",
            "output7": "予備",
        }

        # Add new heartbeat entries to settings
        try:
            input_desc.setdefault("input8", "ﾊｰﾄﾋﾞｰﾄ")
            output_desc.setdefault("output8", "ﾊｰﾄﾋﾞｰﾄ")
        except Exception:
            pass
        # Inputs frame
        in_frame = tk.LabelFrame(frm, text="PLC 入力", padx=8, pady=8)
        in_frame.grid(row=0, column=0, sticky="nsew")
        # Outputs frame
        out_frame = tk.LabelFrame(frm, text="PC出力", padx=8, pady=8)
        out_frame.grid(row=0, column=1, sticky="nsew", padx=(12, 0))

        frm.grid_columnconfigure(0, weight=1)
        frm.grid_columnconfigure(1, weight=1)

        input_vars = {}
        output_vars = {}

        # Helper to create rows
        def make_rows(container, items, values, prefix):
            r = 0
            for key, desc in items.items():
                tk.Label(container, text=f"[{key.replace(prefix, '')}] {desc}").grid(row=r, column=0, sticky="w", padx=(0, 6), pady=2)
                var = tk.StringVar(value=values.get(key, ""))
                ent = tk.Entry(container, textvariable=var, width=14)
                ent.grid(row=r, column=1, sticky="w")
                if prefix == "input":
                    input_vars[key] = var
                else:
                    output_vars[key] = var
                r += 1

        make_rows(in_frame, input_desc, config["plc_addresses"]["inputs"], "input")
        make_rows(out_frame, output_desc, config["plc_addresses"]["outputs"], "output")

        # {Output}: 貂ｬ螳壼､ (word device for writing measured value)
        value_frame = tk.LabelFrame(frm, text="測定値", padx=8, pady=8)
        value_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(12, 0))
        tk.Label(value_frame, text="測定値書き込み先頭ｱﾄﾞﾚｽ､").grid(row=0, column=0, sticky="w")
        value_var = tk.StringVar(value=config["plc_addresses"].get("value_output", ""))
        tk.Entry(value_frame, textvariable=value_var, width=14).grid(row=0, column=1, sticky="w", padx=(8, 0))

        # Buttons
        btns = tk.Frame(frm)
        btns.grid(row=2, column=0, columnspan=2, sticky="e", pady=(12, 0))

        def on_save() -> None:
            for k, v in input_vars.items():
                config["plc_addresses"]["inputs"][k] = v.get().strip()
            for k, v in output_vars.items():
                config["plc_addresses"]["outputs"][k] = v.get().strip()
            config["plc_addresses"]["value_output"] = value_var.get().strip()
            save_settings()
            dlg.destroy()

        tk.Button(btns, text="保存", width=10, command=on_save).pack(side="right", padx=(8, 0))
        tk.Button(btns, text="ｷｬﾝｾﾙ", width=10, command=dlg.destroy).pack(side="right")

    ui.set_on_open_plc_settings(open_plc_settings_dialog)

    def on_close() -> None:
        _stop_plc_monitoring()
        save_settings()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)

    # Load settings at startup and begin PLC scanning automatically (non-blocking)
    load_settings()
    try:
        _start_plc_monitoring()
    except Exception:
        pass

    root.mainloop()


if __name__ == "__main__":
    main()

