"""
Evolution Panel
Control self-evolution cycles
"""

import tkinter as tk
import threading
import queue
from queue import Empty
from tkinter import ttk, scrolledtext
from typing import Optional, Callable

from ..theme import COLORS, FONTS, STYLES


class EvolutionPanel:
    """Panel for controlling self-evolution"""
    
    def __init__(self, parent, evolution_callback: Optional[Callable] = None):
        """
        Initialize evolution panel
        
        Args:
            parent: Parent widget
            evolution_callback: Callback to trigger evolution
        """
        self.parent = parent
        self.evolution_callback = evolution_callback
        
        self.frame = tk.Frame(parent, **STYLES['frame'])
        self.frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self._main_thread_id = threading.get_ident()
        self._ui_queue = queue.Queue()
        self._ui_queue_pump_active = False
        
        self._create_widgets()
        self._start_ui_queue_pump()

    def _start_ui_queue_pump(self):
        if self._ui_queue_pump_active:
            return

        self._ui_queue_pump_active = True
        try:
            self.frame.after(100, self._drain_ui_queue)
        except tk.TclError:
            self._ui_queue_pump_active = False

    def _drain_ui_queue(self):
        try:
            for _ in range(100):
                try:
                    callback = self._ui_queue.get_nowait()
                except Empty:
                    break
                callback()
        finally:
            if self._ui_queue_pump_active:
                try:
                    if self.frame.winfo_exists():
                        self.frame.after(100, self._drain_ui_queue)
                    else:
                        self._ui_queue_pump_active = False
                except tk.TclError:
                    self._ui_queue_pump_active = False

    def _run_on_ui_thread(self, callback):
        if threading.get_ident() == self._main_thread_id:
            callback()
        else:
            self._ui_queue.put(callback)
    
    def _create_widgets(self):
        """Create evolution control widgets"""
        # Title
        title = tk.Label(
            self.frame,
            text="🧬 SELF-EVOLUTION ENGINE 🧬",
            font=FONTS['heading'],
            fg=COLORS['accent_pink'],
            bg=COLORS['bg_panel']
        )
        title.pack(pady=10)
        
        # Objectives input
        obj_frame = tk.Frame(self.frame, bg=COLORS['bg_panel'])
        obj_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Label(
            obj_frame,
            text="Evolution Objectives:",
            font=FONTS['default'],
            fg=COLORS['text_primary'],
            bg=COLORS['bg_panel']
        ).pack(anchor=tk.W)
        
        self.objectives_text = scrolledtext.ScrolledText(
            obj_frame,
            height=4,
            bg=COLORS['bg_secondary'],
            fg=COLORS['text_primary'],
            insertbackground=COLORS['accent_cyan'],
            font=FONTS['mono'],
            relief=tk.FLAT,
            borderwidth=1
        )
        self.objectives_text.pack(fill=tk.X, pady=5)
        self.objectives_text.insert('1.0', "Optimize retry strategy\nImprove template generation\nEnhance evaluation metrics")
        
        # Parameters
        param_frame = tk.Frame(self.frame, bg=COLORS['bg_panel'])
        param_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Label(
            param_frame,
            text="Modules per Cycle:",
            font=FONTS['default'],
            fg=COLORS['text_primary'],
            bg=COLORS['bg_panel']
        ).pack(side=tk.LEFT, padx=5)
        
        self.num_modules = tk.StringVar(value="3")
        modules_entry = tk.Entry(
            param_frame,
            textvariable=self.num_modules,
            width=5,
            **STYLES['entry']
        )
        modules_entry.pack(side=tk.LEFT, padx=5)
        
        # Control buttons
        button_frame = tk.Frame(self.frame, bg=COLORS['bg_panel'])
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        
        start_button_style = STYLES['button'].copy()
        start_button_style['font'] = FONTS['heading']
        self.start_button = tk.Button(
            button_frame,
            text="▶ START EVOLUTION",
            command=self._start_evolution,
            **start_button_style
        )
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        stop_button_style = STYLES['button'].copy()
        stop_button_style['fg'] = COLORS['error']
        self.stop_button = tk.Button(
            button_frame,
            text="⏹ STOP",
            command=self._stop_evolution,
            **stop_button_style
        )
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        # Status
        self.status_label = tk.Label(
            self.frame,
            text="Status: Ready",
            font=FONTS['default'],
            fg=COLORS['accent_green'],
            bg=COLORS['bg_panel']
        )
        self.status_label.pack(pady=5)
        
        # Evolution log
        log_frame = tk.Frame(self.frame, bg=COLORS['bg_panel'])
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        tk.Label(
            log_frame,
            text="Evolution Log:",
            font=FONTS['default'],
            fg=COLORS['text_primary'],
            bg=COLORS['bg_panel']
        ).pack(anchor=tk.W)
        
        self.log_text = scrolledtext.ScrolledText(
            log_frame,
            height=10,
            bg=COLORS['bg_secondary'],
            fg=COLORS['accent_green'],
            insertbackground=COLORS['accent_cyan'],
            font=FONTS['mono'],
            relief=tk.FLAT,
            borderwidth=1
        )
        self.log_text.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.evolution_running = False

    def _start_evolution(self):
        """Start evolution cycle"""
        if self.evolution_running:
            self._append_log("[INFO] Evolution is already running\n")
            return

        if self.evolution_callback:
            objectives = self.objectives_text.get('1.0', tk.END).strip().split('\n')
            objectives = [obj for obj in objectives if obj.strip()]
            if not objectives:
                self._append_log("[ERROR] Please provide at least one objective\n")
                return

            try:
                num_modules = int(self.num_modules.get())
            except ValueError:
                self._append_log("[ERROR] Modules per Cycle must be an integer\n")
                return

            self.evolution_running = True
            self.start_button.config(state=tk.DISABLED)
            self.status_label.config(text="Status: Evolving...", fg=COLORS['accent_pink'])
            self._append_log(f"[EVOLUTION] Starting cycle with {len(objectives)} objectives\n")

            thread = threading.Thread(
                target=self._run_evolution_worker,
                args=(objectives, num_modules),
                daemon=True,
                name="EvolutionWorker"
            )
            thread.start()
        else:
            self._append_log("[ERROR] Evolution executor is not available\n")

    def _run_evolution_worker(self, objectives, num_modules):
        """Run evolution without blocking the Tk main loop."""
        result = None
        error = None

        try:
            result = self.evolution_callback(objectives, num_modules)
        except Exception as e:
            error = e

        self._run_on_ui_thread(lambda: self._finish_evolution(result, error))

    def _finish_evolution(self, result, error):
        """Update UI after an evolution worker completes."""
        if error:
            self._append_log(f"[ERROR] {str(error)}\n")
        elif result:
            self._append_log("[SUCCESS] Evolution cycle completed\n")
            self._append_log(f"  Best module: {result.best_module}\n")
            self._append_log(f"  Score: {result.improvement_score:.3f}\n")
        else:
            self._append_log("[INFO] Evolution completed without a result\n")

        self.evolution_running = False
        self.start_button.config(state=tk.NORMAL)
        self.status_label.config(text="Status: Complete", fg=COLORS['accent_green'])

    def _append_log(self, message: str):
        """Append to the evolution log from the Tk thread."""
        def update():
            self.log_text.insert(tk.END, message)
            self.log_text.see(tk.END)

        self._run_on_ui_thread(update)

    def _stop_evolution(self):
        """Stop evolution"""
        if self.evolution_running:
            self._append_log("[STOP] Stop requested; current LLM request will finish first\n")
        else:
            self._append_log("[STOP] Evolution stopped by user\n")

        self.evolution_running = False
        self.status_label.config(text="Status: Stopped", fg=COLORS['error'])
