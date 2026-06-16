"""
Step 5: Simulation & Testing
Handles simulation submission and monitoring with concurrent slot-based execution
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
from typing import List, Dict
import logging
import time
import threading

from ...theme import COLORS, FONTS, STYLES
from generation_two.core.simulator_tester import MAX_CONCURRENT_SIMULATIONS, SIMULATION_MAX_WAIT_TIME

logger = logging.getLogger(__name__)


class Step5Simulation:
    """Step 5: Simulation & Testing"""

    def __init__(self, parent_frame, workflow_panel):
        """
        Initialize Step 5

        Args:
            parent_frame: Parent frame to pack into
            workflow_panel: Reference to main WorkflowPanel for callbacks
        """
        self.parent_frame = parent_frame
        self.workflow = workflow_panel
        self.frame = tk.Frame(parent_frame, bg=COLORS['bg_panel'])

        # Simulation state
        self.simulation_running = False
        self.simulation_threads = []
        self.slot_manager = None
        self.stop_simulation_flag = False

        self._create_widgets()

    def _validate_before_submit(
        self,
        template: str,
        template_region: str,
        available_operators: List[Dict] = None,
        available_fields: List[Dict] = None,
    ):
        """Run local checks that can prevent avoidable WorldQuant FAIL responses."""
        if not available_operators and self.workflow.generator and self.workflow.generator.template_generator:
            if hasattr(self.workflow.generator.template_generator, 'operator_fetcher'):
                operator_fetcher = self.workflow.generator.template_generator.operator_fetcher
                available_operators = operator_fetcher.operators if operator_fetcher else None

        if not available_fields and self.workflow.generator and self.workflow.generator.template_generator:
            available_fields = self.workflow.generator.template_generator.get_data_fields_for_region(template_region)

        if not available_operators or not available_fields:
            logger.debug("Skipping local expression validation: missing operators or fields")
            return None

        try:
            from generation_two.core.local_expression_validator import validate_expression_locally

            return validate_expression_locally(template, available_operators, available_fields)
        except Exception as e:
            logger.debug(f"Local expression validation failed unexpectedly: {e}")
            return None

    def _normalize_before_submit(
        self,
        template: str,
        available_operators: List[Dict] = None,
    ) -> str:
        """Apply local operator-parameter normalization before validation/submission."""
        if not available_operators and self.workflow.generator and self.workflow.generator.template_generator:
            if hasattr(self.workflow.generator.template_generator, 'operator_fetcher'):
                operator_fetcher = self.workflow.generator.template_generator.operator_fetcher
                available_operators = operator_fetcher.operators if operator_fetcher else None

        if not available_operators:
            return template

        try:
            from generation_two.core.operator_parameter_normalizer import normalize_operator_parameters

            normalized_template, fixes = normalize_operator_parameters(template, available_operators)
            if fixes and normalized_template != template:
                logger.info(f"Applied pre-submit operator normalization: {fixes}")
                return normalized_template
        except Exception as e:
            logger.debug(f"Pre-submit operator normalization skipped: {e}")

        return template

    def _create_widgets(self):
        """Create Step 5 widgets"""
        # Title
        title_label = tk.Label(
            self.frame,
            text="Step 5: Simulation & Testing",
            font=FONTS['heading'],
            fg=COLORS['accent_cyan'],
            bg=COLORS['bg_panel']
        )
        title_label.pack(pady=10)

        tk.Label(
            self.frame,
            text="Simulate generated alphas and view results:",
            font=FONTS['default'],
            fg=COLORS['text_primary'],
            bg=COLORS['bg_panel']
        ).pack(pady=5)

        # Simulation settings
        settings_frame = tk.Frame(self.frame, bg=COLORS['bg_secondary'], relief=tk.RAISED, bd=2)
        settings_frame.pack(fill=tk.X, padx=10, pady=5)

        tk.Label(
            settings_frame,
            text="⚙️ Simulation Settings:",
            font=FONTS['heading'],
            fg=COLORS['accent_cyan'],
            bg=COLORS['bg_secondary']
        ).pack(anchor=tk.W, padx=5, pady=5)

        settings_inner = tk.Frame(settings_frame, bg=COLORS['bg_secondary'])
        settings_inner.pack(fill=tk.X, padx=5, pady=5)

        # Region selection
        tk.Label(settings_inner, text="Region:", bg=COLORS['bg_secondary'], fg=COLORS['text_primary']).grid(row=0, column=0, padx=5, pady=2, sticky=tk.W)
        self.sim_region_var = tk.StringVar(value="USA")
        sim_region_combo = ttk.Combobox(settings_inner, textvariable=self.sim_region_var, values=["USA", "EUR", "CHN", "ASI", "GLB", "IND"], width=10, state="readonly")
        sim_region_combo.grid(row=0, column=1, padx=5, pady=2)

        # Test period
        tk.Label(settings_inner, text="Test Period:", bg=COLORS['bg_secondary'], fg=COLORS['text_primary']).grid(row=0, column=2, padx=5, pady=2, sticky=tk.W)
        self.test_period_var = tk.StringVar(value="P5Y0M0D")
        test_period_combo = ttk.Combobox(settings_inner, textvariable=self.test_period_var,
                                         values=["P1Y0M0D", "P2Y0M0D", "P3Y0M0D", "P5Y0M0D", "P10Y0M0D"],
                                         width=12, state="readonly")
        test_period_combo.grid(row=0, column=3, padx=5, pady=2)

        # Neutralization
        tk.Label(settings_inner, text="Neutralization:", bg=COLORS['bg_secondary'], fg=COLORS['text_primary']).grid(row=1, column=0, padx=5, pady=2, sticky=tk.W)
        self.neutralization_var = tk.StringVar(value="INDUSTRY")
        neutralization_combo = ttk.Combobox(settings_inner, textvariable=self.neutralization_var,
                                            values=["INDUSTRY", "SUBINDUSTRY", "SECTOR", "COUNTRY", "MARKET"],
                                            width=12, state="readonly")
        neutralization_combo.grid(row=1, column=1, padx=5, pady=2)

        # Truncation
        tk.Label(settings_inner, text="Truncation:", bg=COLORS['bg_secondary'], fg=COLORS['text_primary']).grid(row=1, column=2, padx=5, pady=2, sticky=tk.W)
        self.truncation_var = tk.StringVar(value="0.05")
        truncation_entry = tk.Entry(settings_inner, textvariable=self.truncation_var, width=10)
        truncation_entry.grid(row=1, column=3, padx=5, pady=2)

        # Simulation controls
        sim_frame = tk.Frame(self.frame, bg=COLORS['bg_panel'])
        sim_frame.pack(fill=tk.X, padx=10, pady=5)

        self.simulate_button = tk.Button(
            sim_frame,
            text="🚀 SIMULATE SELECTED",
            command=self._simulate_selected,
            **STYLES['button']
        )
        self.simulate_button.pack(side=tk.LEFT, padx=5)

        self.simulate_all_button = tk.Button(
            sim_frame,
            text="🚀 SIMULATE ALL",
            command=self._simulate_all,
            **STYLES['button']
        )
        self.simulate_all_button.pack(side=tk.LEFT, padx=5)

        self.stop_simulation_button = tk.Button(
            sim_frame,
            text="⏹️ STOP",
            command=self._stop_simulation,
            **STYLES['button'],
            state=tk.DISABLED
        )
        self.stop_simulation_button.pack(side=tk.LEFT, padx=5)

        # Progress indicator
        self.sim_progress_label = tk.Label(
            sim_frame,
            text="",
            font=FONTS['default'],
            fg=COLORS['accent_yellow'],
            bg=COLORS['bg_panel']
        )
        self.sim_progress_label.pack(side=tk.LEFT, padx=10)

        # Slot visualization
        slots_frame = tk.Frame(self.frame, bg=COLORS['bg_secondary'], relief=tk.RAISED, bd=2)
        slots_frame.pack(fill=tk.X, padx=10, pady=5)

        tk.Label(
            slots_frame,
            text=f"📊 Simulation Slots ({MAX_CONCURRENT_SIMULATIONS} concurrent):",
            font=FONTS['heading'],
            fg=COLORS['accent_cyan'],
            bg=COLORS['bg_secondary']
        ).pack(anchor=tk.W, padx=5, pady=5)

        # Create slot display grid
        slots_grid = tk.Frame(slots_frame, bg=COLORS['bg_secondary'])
        slots_grid.pack(fill=tk.X, padx=5, pady=5)

        self.slot_widgets = {}  # {slot_id: {frame, status_label, template_label, log_text}}
        columns = min(MAX_CONCURRENT_SIMULATIONS, 3)
        for slot_id in range(MAX_CONCURRENT_SIMULATIONS):
            row = slot_id // columns
            col = slot_id % columns
            slot_frame = tk.Frame(slots_grid, bg=COLORS['bg_panel'], relief=tk.RAISED, bd=1)
            slot_frame.grid(row=row, column=col, padx=2, pady=2, sticky=tk.NSEW)
            slots_grid.columnconfigure(col, weight=1)
            slots_grid.rowconfigure(row, weight=1)

            # Slot header
            header = tk.Frame(slot_frame, bg=COLORS['bg_panel'])
            header.pack(fill=tk.X, padx=2, pady=2)

            slot_num_label = tk.Label(
                header,
                text=f"Slot {slot_id + 1}",
                font=FONTS['default'],
                fg=COLORS['accent_cyan'],
                bg=COLORS['bg_panel']
            )
            slot_num_label.pack(side=tk.LEFT)

            status_label = tk.Label(
                header,
                text="IDLE",
                font=FONTS['default'],
                fg=COLORS['accent_green'],
                bg=COLORS['bg_panel']
            )
            status_label.pack(side=tk.RIGHT)

            # Template preview
            template_label = tk.Label(
                slot_frame,
                text="",
                font=FONTS['mono'],
                fg=COLORS['text_secondary'],
                bg=COLORS['bg_panel'],
                wraplength=150,
                justify=tk.LEFT
            )
            template_label.pack(fill=tk.X, padx=2, pady=2)

            # Progress bar
            progress_frame = tk.Frame(slot_frame, bg=COLORS['bg_panel'])
            progress_frame.pack(fill=tk.X, padx=2, pady=2)

            progress_label = tk.Label(
                progress_frame,
                text="0%",
                font=('Courier', 7),
                fg=COLORS['accent_cyan'],
                bg=COLORS['bg_panel'],
                width=5
            )
            progress_label.pack(side=tk.LEFT, padx=2)

            # Use ttk.Progressbar instead of Canvas to reduce memory usage (avoids DIBSECTION bitmaps)
            progress_bar = ttk.Progressbar(
                progress_frame,
                mode='determinate',
                length=120,
                maximum=100
            )
            progress_bar.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)

            progress_text = tk.Label(
                progress_frame,
                text="",
                font=('Courier', 6),
                fg=COLORS['text_secondary'],
                bg=COLORS['bg_panel'],
                wraplength=120
            )
            progress_text.pack(fill=tk.X, padx=2)

            # Slot log (small terminal)
            log_text = scrolledtext.ScrolledText(
                slot_frame,
                height=6,
                width=25,
                bg=COLORS['bg_secondary'],
                fg=COLORS['accent_green'],
                font=('Courier', 8),
                wrap=tk.WORD,
                state=tk.DISABLED
            )
            log_text.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

            self.slot_widgets[slot_id] = {
                'frame': slot_frame,
                'status_label': status_label,
                'template_label': template_label,
                'progress_label': progress_label,
                'progress_bar': progress_bar,
                'progress_text': progress_text,
                'log_text': log_text
            }

        # Results display (summary)
        results_frame = tk.Frame(self.frame, bg=COLORS['bg_panel'])
        results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        tk.Label(
            results_frame,
            text="📋 Summary Log:",
            font=FONTS['heading'],
            fg=COLORS['accent_cyan'],
            bg=COLORS['bg_panel']
        ).pack(anchor=tk.W, pady=5)

        self.results_text = scrolledtext.ScrolledText(
            results_frame,
            height=10,
            bg=COLORS['bg_secondary'],
            fg=COLORS['accent_green'],
            font=FONTS['mono'],
            wrap=tk.WORD
        )
        self.results_text.pack(fill=tk.BOTH, expand=True)

        # Ensure frame is properly configured
        self.frame.pack_propagate(True)

    def _simulate_selected(self):
        """Simulate selected templates"""
        # Access templates_listbox from Step 4
        if not hasattr(self.workflow, 'step4') or not hasattr(self.workflow.step4, 'templates_listbox'):
            messagebox.showwarning("Warning", "No templates available. Please generate templates in Step 4 first.")
            return

        selection = self.workflow.step4.templates_listbox.curselection()
        if not selection:
            messagebox.showwarning("Warning", "Please select templates to simulate")
            return

        if not hasattr(self.workflow.step4, 'generated_templates'):
            messagebox.showwarning("Warning", "No templates generated yet")
            return

        if self.simulation_running:
            messagebox.showwarning("Warning", "Simulation already running. Please wait or stop current simulation.")
            return

        # Get selected templates
        selected_templates = [self.workflow.step4.generated_templates[i] for i in selection]
        self._run_simulations(selected_templates)

    def _simulate_all(self):
        """Simulate all templates"""
        if not hasattr(self.workflow, 'step4') or not hasattr(self.workflow.step4, 'generated_templates'):
            messagebox.showwarning("Warning", "No templates generated yet")
            return

        if self.simulation_running:
            messagebox.showwarning("Warning", "Simulation already running. Please wait or stop current simulation.")
            return

        # Simulate all templates
        self._run_simulations(self.workflow.step4.generated_templates)

    def _stop_simulation(self):
        """Stop running simulations"""
        self.simulation_running = False
        self.stop_simulation_flag = True
        self.sim_progress_label.config(text="Stopping simulations...")
        self.stop_simulation_button.config(state=tk.DISABLED)
        self._log_result("⚠️ Simulation stopped by user\n")

    def _run_simulations(self, templates: List[Dict]):
        """Run simulations for given templates using slot-based concurrent execution"""
        if not self.workflow.generator or not self.workflow.generator.template_generator:
            messagebox.showerror("Error", "Generator not initialized")
            return

        # Check authentication
        if not self.workflow.generator.template_generator.sess or not self.workflow.generator.template_generator.sess.cookies:
            messagebox.showerror("Error", "Not authenticated. Please complete Step 1 first.")
            return

        self.simulation_running = True
        self.stop_simulation_flag = False  # Reset stop flag
        self.simulate_button.config(state=tk.DISABLED)
        self.simulate_all_button.config(state=tk.DISABLED)
        self.stop_simulation_button.config(state=tk.NORMAL)

        # Clear results and slots
        self.results_text.delete(1.0, tk.END)
        self._log_result("=" * 80 + "\n")
        self._log_result(f"🚀 STARTING CONCURRENT SIMULATIONS ({MAX_CONCURRENT_SIMULATIONS} slots)\n")
        self._log_result("=" * 80 + "\n\n")

        # Clear all slot displays
        for slot_id in range(MAX_CONCURRENT_SIMULATIONS):
            self._update_slot_display(slot_id, "IDLE", "", "", [])

        # Get simulation settings
        region = self.sim_region_var.get()
        test_period = self.test_period_var.get()
        neutralization = self.neutralization_var.get()
        try:
            truncation = float(self.truncation_var.get())
        except ValueError:
            truncation = 0.05

        # Import simulation classes
        try:
            from generation_two.core.simulator_tester import SimulatorTester, SimulationSettings
            from generation_two.core.region_config import REGION_DEFAULT_UNIVERSE
            from generation_two.core.slot_manager import SlotManager, SlotStatus

            # Initialize slot manager
            self.slot_manager = SlotManager(max_slots=MAX_CONCURRENT_SIMULATIONS)

            # Setup simulator tester
            region_configs = {}
            for reg in ["USA", "EUR", "CHN", "ASI", "GLB", "IND"]:
                universe = REGION_DEFAULT_UNIVERSE.get(reg, "TOP3000")
                region_configs[reg] = type('RegionConfig', (), {
                    'region': reg,
                    'universe': universe,
                    'delay': 1
                })()

            simulator = SimulatorTester(
                session=self.workflow.generator.template_generator.sess,
                region_configs=region_configs,
                template_generator=self.workflow.generator.template_generator
            )

            # Queue for templates waiting for slots
            template_queue = list(enumerate(templates))
            completed_count = {'successful': 0, 'failed': 0, 'total': len(templates)}

            def run_simulation_in_slot(template_index: int, template_dict: Dict, slot_ids: List[int]):
                """Run a single simulation in assigned slot(s)"""
                try:
                    # Get primary slot ID first (needed for logging during placeholder replacement)
                    primary_slot_id = slot_ids[0]

                    template = template_dict.get('template', '')
                    template_region = template_dict.get('region', region)

                    # Clean template - remove backticks and fix common errors
                    template = template.replace('`', '').strip()
                    available_operators = None
                    available_fields = None

                    # V4 Approach: Replace placeholders using Ollama selection (if using algorithmic generation)
                    if self.workflow.generator and self.workflow.generator.template_generator:
                        if hasattr(self.workflow.generator.template_generator, 'operator_fetcher'):
                            available_operators = self.workflow.generator.template_generator.operator_fetcher.operators if self.workflow.generator.template_generator.operator_fetcher else None

                        available_fields = self.workflow.generator.template_generator.get_data_fields_for_region(template_region)

                        # Check if template has placeholders
                        has_operator_placeholders = template and ('OPERATOR' in template.upper() or 'operator' in template.lower())
                        has_field_placeholders = template and ('DATA_FIELD' in template.upper() or 'data_field' in template.lower())

                        if has_operator_placeholders or has_field_placeholders:
                            # Use Ollama to select indices for replacement
                            if available_operators and available_fields and hasattr(self.workflow.generator.template_generator, 'ollama_manager'):
                                self._log_to_slot(primary_slot_id, "🤖 Asking Ollama to select operators and fields...")

                                def progress_callback(msg):
                                    self._log_to_slot(primary_slot_id, f"🤖 {msg}")

                                # Get backtest_storage for field usage tracking
                                backtest_storage = None
                                if hasattr(self.workflow.generator, 'backtest_storage'):
                                    backtest_storage = self.workflow.generator.backtest_storage

                                replaced = self.workflow.generator.template_generator.ollama_manager.replace_placeholders_with_selection(
                                    template,
                                    available_operators,
                                    available_fields,
                                    progress_callback=progress_callback,
                                    region=template_region,
                                    backtest_storage=backtest_storage
                                )
                                if replaced:
                                    template = replaced
                                    # Verify all placeholders were actually replaced
                                    import re
                                    remaining_ops = re.findall(r'\b(OPERATOR\d+|operator\d+|Operator\d+)\b', template, re.IGNORECASE)
                                    remaining_fields = re.findall(r'\b(DATA_FIELD\d+|data_field\d+|Data_Field\d+)\b', template, re.IGNORECASE)

                                    if remaining_ops or remaining_fields:
                                        self._log_to_slot(primary_slot_id, f"⚠️ Some placeholders not replaced! Remaining: {remaining_ops + remaining_fields}")
                                        self._log_to_slot(primary_slot_id, "🔄 Retrying placeholder replacement...")
                                        # Retry once more
                                        # Get backtest_storage for field usage tracking
                                        backtest_storage = None
                                        if hasattr(self.workflow.generator, 'backtest_storage'):
                                            backtest_storage = self.workflow.generator.backtest_storage

                                        replaced_retry = self.workflow.generator.template_generator.ollama_manager.replace_placeholders_with_selection(
                                            template,
                                            available_operators,
                                            available_fields,
                                            progress_callback=progress_callback,
                                            region=template_region,
                                            backtest_storage=backtest_storage
                                        )
                                        if replaced_retry:
                                            template = replaced_retry
                                            # Check again
                                            remaining_ops = re.findall(r'\b(OPERATOR\d+|operator\d+|Operator\d+)\b', template, re.IGNORECASE)
                                            remaining_fields = re.findall(r'\b(DATA_FIELD\d+|data_field\d+|Data_Field\d+)\b', template, re.IGNORECASE)
                                            if remaining_ops or remaining_fields:
                                                self._log_to_slot(primary_slot_id, f"❌ FAILED: Still has placeholders after retry: {remaining_ops + remaining_fields}")
                                                self._log_to_slot(primary_slot_id, f"❌ Skipping submission - template: {template[:100]}...")
                                                # Mark as failed and skip submission
                                                self.slot_manager.update_slot_status(slot_ids, "FAILED", f"Placeholders not replaced: {remaining_ops + remaining_fields}")
                                                return
                                            else:
                                                self._log_to_slot(primary_slot_id, "✅ All placeholders replaced after retry")
                                        else:
                                            self._log_to_slot(primary_slot_id, "❌ Retry replacement failed, skipping submission")
                                            self.slot_manager.update_slot_status(slot_ids, "FAILED", "Placeholder replacement failed")
                                            return
                                    else:
                                        self._log_to_slot(primary_slot_id, "✅ Ollama selection completed, all placeholders replaced")
                                else:
                                    self._log_to_slot(primary_slot_id, "⚠️ Ollama selection failed, using fallback replacement")
                                    # Fallback to old method
                                    if has_operator_placeholders:
                                        template = self.workflow.generator.template_generator._replace_operator_placeholders(
                                            template, available_operators
                                        )
                                    if has_field_placeholders:
                                        template = self.workflow.generator.template_generator._replace_field_placeholders(
                                            template, available_fields, template_region
                                        )
                                    # Verify fallback worked
                                    import re
                                    remaining_ops = re.findall(r'\b(OPERATOR\d+|operator\d+|Operator\d+)\b', template, re.IGNORECASE)
                                    remaining_fields = re.findall(r'\b(DATA_FIELD\d+|data_field\d+|Data_Field\d+)\b', template, re.IGNORECASE)
                                    if remaining_ops or remaining_fields:
                                        self._log_to_slot(primary_slot_id, f"❌ FAILED: Fallback replacement incomplete. Remaining: {remaining_ops + remaining_fields}")
                                        self.slot_manager.update_slot_status(slot_ids, "FAILED", f"Placeholders not replaced: {remaining_ops + remaining_fields}")
                                        return
                            else:
                                # Fallback to old method
                                if has_operator_placeholders and available_operators:
                                    template = self.workflow.generator.template_generator._replace_operator_placeholders(
                                        template, available_operators
                                    )
                                if has_field_placeholders and available_fields:
                                    template = self.workflow.generator.template_generator._replace_field_placeholders(
                                        template, available_fields, template_region
                                    )

                    # Final check: Ensure NO placeholders remain before submission
                    import re
                    remaining_ops = re.findall(r'\b(OPERATOR\d+|operator\d+|Operator\d+)\b', template, re.IGNORECASE)
                    remaining_fields = re.findall(r'\b(DATA_FIELD\d+|data_field\d+|Data_Field\d+)\b', template, re.IGNORECASE)

                    if remaining_ops or remaining_fields:
                        self._log_to_slot(primary_slot_id, f"❌ CRITICAL: Template still has placeholders before submission!")
                        self._log_to_slot(primary_slot_id, f"   Remaining operators: {remaining_ops}")
                        self._log_to_slot(primary_slot_id, f"   Remaining fields: {remaining_fields}")
                        self._log_to_slot(primary_slot_id, f"   Template: {template[:100]}...")
                        self.slot_manager.update_slot_status(slot_ids, "FAILED", f"Cannot submit: placeholders remain ({remaining_ops + remaining_fields})")
                        self._log_to_slot(primary_slot_id, "❌ Skipping submission - template has unreplaced placeholders")
                        return

                    normalized_template = self._normalize_before_submit(template, available_operators)
                    if normalized_template != template:
                        self._log_to_slot(primary_slot_id, f"🔧 Normalized expression before submit: {normalized_template[:80]}...")
                        template = normalized_template

                    local_validation = self._validate_before_submit(
                        template,
                        template_region,
                        available_operators=available_operators,
                        available_fields=available_fields,
                    )
                    if local_validation and not local_validation.is_valid:
                        error_msg = local_validation.summary()
                        completed_count['failed'] += 1
                        self._log_to_slot(primary_slot_id, f"❌ Local validation failed: {error_msg}")
                        self._log_to_slot(primary_slot_id, "❌ Skipping API submission to avoid known invalid expression")
                        if self.workflow.generator.template_generator.template_validator:
                            self.workflow.generator.template_generator.template_validator.learn_from_simulation_error(
                                template, error_msg, None
                            )
                        self.slot_manager.release_slots(slot_ids, success=False, error=f"Local validation failed: {error_msg}")
                        self._update_slot_display(
                            primary_slot_id,
                            "FAILED",
                            template[:40] + "...",
                            f"❌ Local validation",
                            [f"❌ FAILED", error_msg[:80]]
                        )
                        self._log_result(f"❌ [{template_index+1}] LOCAL VALIDATION FAILED: {error_msg}\n")
                        return

                    # Update template dict with cleaned and replaced version
                    template_dict['template'] = template

                    # Update slot display
                    self._update_slot_display(primary_slot_id, "RUNNING", template[:40] + "...", f"Region: {template_region}", [f"Starting simulation..."])
                    self._log_to_slot(primary_slot_id, f"[{template_index+1}/{len(templates)}] Starting: {template[:50]}...")

                    # Create simulation settings
                    settings = SimulationSettings(
                        region=template_region,
                        universe=REGION_DEFAULT_UNIVERSE.get(template_region, "TOP3000"),
                        testPeriod=test_period,
                        neutralization=neutralization,
                        truncation=truncation
                    )

                    # Submit simulation
                    self._log_to_slot(primary_slot_id, f"Submitting to API...")
                    submission = simulator.submit_simulation(template, template_region, settings)

                    if submission:
                        progress_url = submission.progress_url
                        self.slot_manager.update_slot_progress(
                            primary_slot_id,
                            progress_url=progress_url,
                            percent=10,
                            message=f"✅ Submitted: {progress_url}",
                            api_status="SUBMITTED"
                        )
                        self._log_to_slot(primary_slot_id, f"✅ Submitted: {progress_url}")
                        self._log_result(f"[{template_index+1}/{len(templates)}] ✅ Submitted: {template[:50]}...\n")

                        # Monitor simulation (with timeout and progress callback)
                        self._log_to_slot(primary_slot_id, "Monitoring progress...")

                        # Create progress callback for this slot
                        def progress_callback(percent: float, message: str, api_status: str):
                            """Update slot progress"""
                            if self.slot_manager:
                                slot = self.slot_manager.get_slot_status(primary_slot_id)
                                slot.update_progress(percent, message, api_status)
                                self._update_slot_progress(primary_slot_id, percent, message, api_status)

                        try:
                            result = simulator.monitor_simulation(
                                progress_url,
                                template,
                                template_region,
                                settings,
                                max_wait_time=SIMULATION_MAX_WAIT_TIME,
                                progress_callback=progress_callback
                            )
                        finally:
                            simulator.release_simulation_slot(submission)

                        if result.success:
                            completed_count['successful'] += 1
                            # Ensure alpha_id is a string, not a tuple or list
                            alpha_id = result.alpha_id
                            if isinstance(alpha_id, (tuple, list)):
                                alpha_id = str(alpha_id[0]) if alpha_id and len(alpha_id) > 0 else ''
                            elif alpha_id is None:
                                alpha_id = ''
                            else:
                                alpha_id = str(alpha_id)
                            result_dict = {
                                'alpha_id': alpha_id,
                                'returns': result.returns,
                                'sharpe': result.sharpe,
                                'turnover': result.turnover,
                                'max_drawdown': result.max_drawdown
                            }
                            self.slot_manager.release_slots(slot_ids, success=True, result=result_dict)
                            self._update_slot_display(primary_slot_id, "COMPLETED", template[:40] + "...",
                                                     f"✅ Alpha: {alpha_id}",
                                                     [f"✅ SUCCESS", f"Returns: {result.returns:.4f}", f"Sharpe: {result.sharpe:.4f}"])
                            self._log_to_slot(primary_slot_id, f"✅ SUCCESS - Alpha ID: {alpha_id}")
                            self._log_to_slot(primary_slot_id, f"   Returns: {result.returns:.4f}, Sharpe: {result.sharpe:.4f}")
                            self._log_result(f"✅ [{template_index+1}] SUCCESS - Alpha: {alpha_id}, Returns: {result.returns:.4f}, Sharpe: {result.sharpe:.4f}\n")
                        else:
                            # V2-style refeed: Try to fix and retry
                            error_msg = result.error_message or "Unknown error"
                            self._log_to_slot(primary_slot_id, f"❌ FAILED: {error_msg}")

                            # Check if template still has placeholders - replace them first before refeed
                            has_operator_placeholders = template and ('OPERATOR' in template.upper() or 'operator' in template.lower())
                            has_field_placeholders = template and ('DATA_FIELD' in template.upper() or 'data_field' in template.lower())

                            if (has_operator_placeholders or has_field_placeholders) and self.workflow.generator and self.workflow.generator.template_generator:
                                self._log_to_slot(primary_slot_id, "⚠️ Template still has placeholders, replacing before refeed...")
                                available_operators = None
                                if hasattr(self.workflow.generator.template_generator, 'operator_fetcher'):
                                    available_operators = self.workflow.generator.template_generator.operator_fetcher.operators if self.workflow.generator.template_generator.operator_fetcher else None

                                available_fields = self.workflow.generator.template_generator.get_data_fields_for_region(template_region)

                                if available_operators and available_fields and hasattr(self.workflow.generator.template_generator, 'ollama_manager'):
                                    def progress_callback_refeed(msg):
                                        self._log_to_slot(primary_slot_id, f"🤖 {msg}")

                                    # Get backtest_storage for field usage tracking
                                    backtest_storage = None
                                    if hasattr(self.workflow.generator, 'backtest_storage'):
                                        backtest_storage = self.workflow.generator.backtest_storage

                                    replaced = self.workflow.generator.template_generator.ollama_manager.replace_placeholders_with_selection(
                                        template,
                                        available_operators,
                                        available_fields,
                                        progress_callback=progress_callback_refeed,
                                        region=template_region,
                                        backtest_storage=backtest_storage
                                    )
                                    if replaced:
                                        template = replaced
                                        self._log_to_slot(primary_slot_id, "✅ Placeholders replaced before refeed")
                                    else:
                                        self._log_to_slot(primary_slot_id, "⚠️ Placeholder replacement failed, using fallback")
                                        # Fallback to old method
                                        if has_operator_placeholders:
                                            template = self.workflow.generator.template_generator._replace_operator_placeholders(
                                                template, available_operators
                                            )
                                        if has_field_placeholders:
                                            template = self.workflow.generator.template_generator._replace_field_placeholders(
                                                template, available_fields, template_region
                                            )
                                else:
                                    # Fallback to old method
                                    if has_operator_placeholders and available_operators:
                                        template = self.workflow.generator.template_generator._replace_operator_placeholders(
                                            template, available_operators
                                        )
                                    if has_field_placeholders and available_fields:
                                        template = self.workflow.generator.template_generator._replace_field_placeholders(
                                            template, available_fields, template_region
                                        )

                            self._log_to_slot(primary_slot_id, "🔄 Attempting refeed correction...")
                            self._update_slot_progress(primary_slot_id, 50.0, "Fixing template...", "")

                            # Use refeed mechanism with both AST and prompt engineering
                            if self.workflow.generator.template_generator.template_validator:
                                # Check if this is an event input error - use unlimited retries
                                is_event_input_error = 'does not support event inputs' in error_msg.lower() or 'expects only event inputs' in error_msg.lower()
                                max_refeed_attempts = 999 if is_event_input_error else 3  # Unlimited for event inputs

                                fixed_template, fixes = self.workflow.generator.template_generator.template_validator.refeed_with_correction(
                                    template, error_msg, template_region, max_attempts=max_refeed_attempts
                                )

                                if fixed_template and fixed_template != template:
                                    # UNLIMITED RETRY LOOP for event input errors
                                    current_template_for_retry = fixed_template
                                    current_error_msg = error_msg
                                    refeed_attempt = 0
                                    max_refeed_retries = 999 if is_event_input_error else 1  # Unlimited for event inputs

                                    while refeed_attempt < max_refeed_retries:
                                        refeed_attempt += 1

                                        # Check if fixed template still has placeholders - replace them
                                        has_op_placeholders = current_template_for_retry and ('OPERATOR' in current_template_for_retry.upper() or 'operator' in current_template_for_retry.lower())
                                        has_field_placeholders = current_template_for_retry and ('DATA_FIELD' in current_template_for_retry.upper() or 'data_field' in current_template_for_retry.lower())

                                        if (has_op_placeholders or has_field_placeholders) and self.workflow.generator and self.workflow.generator.template_generator:
                                            self._log_to_slot(primary_slot_id, f"⚠️ Fixed template still has placeholders, replacing...")
                                            available_operators = None
                                            if hasattr(self.workflow.generator.template_generator, 'operator_fetcher'):
                                                available_operators = self.workflow.generator.template_generator.operator_fetcher.operators if self.workflow.generator.template_generator.operator_fetcher else None

                                            available_fields = self.workflow.generator.template_generator.get_data_fields_for_region(template_region)

                                            if available_operators and available_fields and hasattr(self.workflow.generator.template_generator, 'ollama_manager'):
                                                def progress_callback_refeed_retry(msg):
                                                    self._log_to_slot(primary_slot_id, f"🤖 {msg}")

                                                # Get backtest_storage for field usage tracking
                                                backtest_storage = None
                                                if hasattr(self.workflow.generator, 'backtest_storage'):
                                                    backtest_storage = self.workflow.generator.backtest_storage

                                                replaced_retry = self.workflow.generator.template_generator.ollama_manager.replace_placeholders_with_selection(
                                                    current_template_for_retry,
                                                    available_operators,
                                                    available_fields,
                                                    progress_callback=progress_callback_refeed_retry,
                                                    region=template_region,
                                                    backtest_storage=backtest_storage
                                                )
                                                if replaced_retry:
                                                    current_template_for_retry = replaced_retry
                                                    self._log_to_slot(primary_slot_id, "✅ Placeholders replaced in fixed template")
                                                else:
                                                    # Fallback
                                                    if has_op_placeholders:
                                                        current_template_for_retry = self.workflow.generator.template_generator._replace_operator_placeholders(
                                                            current_template_for_retry, available_operators
                                                        )
                                                    if has_field_placeholders:
                                                        current_template_for_retry = self.workflow.generator.template_generator._replace_field_placeholders(
                                                            current_template_for_retry, available_fields, template_region
                                                        )
                                            else:
                                                # Fallback
                                                if has_op_placeholders and available_operators:
                                                    current_template_for_retry = self.workflow.generator.template_generator._replace_operator_placeholders(
                                                        current_template_for_retry, available_operators
                                                    )
                                                if has_field_placeholders and available_fields:
                                                    current_template_for_retry = self.workflow.generator.template_generator._replace_field_placeholders(
                                                        current_template_for_retry, available_fields, template_region
                                                    )

                                        self._log_to_slot(primary_slot_id, f"✅ Fixed with {len(fixes)} corrections (refeed attempt {refeed_attempt})")
                                        self._log_to_slot(primary_slot_id, f"Retrying with fixed template...")
                                        self._update_slot_progress(primary_slot_id, 60.0, f"Retrying simulation (attempt {refeed_attempt})...", "")

                                        normalized_retry = self._normalize_before_submit(
                                            current_template_for_retry,
                                            available_operators,
                                        )
                                        if normalized_retry != current_template_for_retry:
                                            self._log_to_slot(primary_slot_id, f"🔧 Normalized retry expression: {normalized_retry[:80]}...")
                                            current_template_for_retry = normalized_retry

                                        local_validation_retry = self._validate_before_submit(
                                            current_template_for_retry,
                                            template_region,
                                            available_operators=available_operators,
                                            available_fields=available_fields,
                                        )
                                        if local_validation_retry and not local_validation_retry.is_valid:
                                            current_error_msg = local_validation_retry.summary()
                                            self._log_to_slot(primary_slot_id, f"❌ Local validation failed before retry: {current_error_msg}")
                                            if self.workflow.generator.template_generator.template_validator:
                                                self.workflow.generator.template_generator.template_validator.learn_from_simulation_error(
                                                    current_template_for_retry, current_error_msg, None
                                                )
                                            break

                                        # Retry with fixed template
                                        submission_retry = simulator.submit_simulation(current_template_for_retry, template_region, settings)

                                        if submission_retry:
                                            progress_url_retry = submission_retry.progress_url
                                            self._log_to_slot(primary_slot_id, f"✅ Resubmitted: {progress_url_retry}")
                                            try:
                                                result_retry = simulator.monitor_simulation(
                                                    progress_url_retry,
                                                    current_template_for_retry,
                                                    template_region,
                                                    settings,
                                                    max_wait_time=SIMULATION_MAX_WAIT_TIME,
                                                    progress_callback=progress_callback
                                                )
                                            finally:
                                                simulator.release_simulation_slot(submission_retry)

                                            if result_retry.success:
                                                completed_count['successful'] += 1
                                                # Ensure alpha_id is a string, not a tuple or list
                                                alpha_id_retry = result_retry.alpha_id
                                                if isinstance(alpha_id_retry, (tuple, list)):
                                                    alpha_id_retry = str(alpha_id_retry[0]) if alpha_id_retry and len(alpha_id_retry) > 0 else ''
                                                elif alpha_id_retry is None:
                                                    alpha_id_retry = ''
                                                else:
                                                    alpha_id_retry = str(alpha_id_retry)
                                                result_dict = {
                                                    'alpha_id': alpha_id_retry,
                                                    'returns': result_retry.returns,
                                                    'sharpe': result_retry.sharpe,
                                                    'turnover': result_retry.turnover,
                                                    'max_drawdown': result_retry.max_drawdown
                                                }
                                                self.slot_manager.release_slots(slot_ids, success=True, result=result_dict)
                                                self._update_slot_display(primary_slot_id, "COMPLETED", current_template_for_retry[:40] + "...",
                                                                         f"✅ Alpha: {alpha_id_retry} (REFED)",
                                                                         [f"✅ SUCCESS (REFED)", f"Returns: {result_retry.returns:.4f}", f"Sharpe: {result_retry.sharpe:.4f}"])
                                                self._log_to_slot(primary_slot_id, f"✅ REFEED SUCCESS - Alpha ID: {alpha_id_retry}")
                                                self._log_result(f"✅ [{template_index+1}] REFEED SUCCESS - Alpha: {alpha_id_retry}, Returns: {result_retry.returns:.4f}, Sharpe: {result_retry.sharpe:.4f}\n")
                                                return  # Success after refeed
                                            else:
                                                # Refeed retry failed - check if it's still an event input error
                                                retry_error_msg = result_retry.error_message or "Unknown error"
                                                is_still_event_input = 'does not support event inputs' in retry_error_msg.lower() or 'expects only event inputs' in retry_error_msg.lower()

                                                self._log_to_slot(primary_slot_id, f"❌ Refeed retry {refeed_attempt} failed: {retry_error_msg[:50]}")

                                                # Learn from error
                                                if self.workflow.generator.template_generator.template_validator:
                                                    self.workflow.generator.template_generator.template_validator.learn_from_simulation_error(
                                                        current_template_for_retry, retry_error_msg, None
                                                    )

                                                # If still event input error, retry refeed with new error
                                                if is_still_event_input and refeed_attempt < max_refeed_retries:
                                                    self._log_to_slot(primary_slot_id, f"🔄 Still event input error, fixing again...")
                                                    self._update_slot_progress(primary_slot_id, 50.0, "Fixing template again...", "")

                                                    # Fix again with new error message
                                                    fixed_template_again, fixes_again = self.workflow.generator.template_generator.template_validator.refeed_with_correction(
                                                        current_template_for_retry, retry_error_msg, template_region, max_attempts=max_refeed_attempts
                                                    )

                                                    if fixed_template_again and fixed_template_again != current_template_for_retry:
                                                        # Check if fixed template has placeholders and replace them
                                                        has_op_ph = fixed_template_again and ('OPERATOR' in fixed_template_again.upper() or 'operator' in fixed_template_again.lower())
                                                        has_field_ph = fixed_template_again and ('DATA_FIELD' in fixed_template_again.upper() or 'data_field' in fixed_template_again.lower())

                                                        if (has_op_ph or has_field_ph) and self.workflow.generator and self.workflow.generator.template_generator:
                                                            available_operators = None
                                                            if hasattr(self.workflow.generator.template_generator, 'operator_fetcher'):
                                                                available_operators = self.workflow.generator.template_generator.operator_fetcher.operators if self.workflow.generator.template_generator.operator_fetcher else None

                                                            available_fields = self.workflow.generator.template_generator.get_data_fields_for_region(template_region)

                                                            if available_operators and available_fields and hasattr(self.workflow.generator.template_generator, 'ollama_manager'):
                                                                # Get backtest_storage for field usage tracking
                                                                backtest_storage = None
                                                                if hasattr(self.workflow.generator, 'backtest_storage'):
                                                                    backtest_storage = self.workflow.generator.backtest_storage

                                                                replaced_again = self.workflow.generator.template_generator.ollama_manager.replace_placeholders_with_selection(
                                                                    fixed_template_again,
                                                                    available_operators,
                                                                    available_fields,
                                                                    region=template_region,
                                                                    backtest_storage=backtest_storage
                                                                )
                                                                if replaced_again:
                                                                    fixed_template_again = replaced_again
                                                            else:
                                                                # Fallback
                                                                if has_op_ph and available_operators:
                                                                    fixed_template_again = self.workflow.generator.template_generator._replace_operator_placeholders(
                                                                        fixed_template_again, available_operators
                                                                    )
                                                                if has_field_ph and available_fields:
                                                                    fixed_template_again = self.workflow.generator.template_generator._replace_field_placeholders(
                                                                        fixed_template_again, available_fields, template_region
                                                                    )

                                                        current_template_for_retry = fixed_template_again
                                                        current_error_msg = retry_error_msg
                                                        fixes.extend(fixes_again)
                                                        self._log_to_slot(primary_slot_id, f"✅ Fixed again with {len(fixes_again)} more corrections")
                                                        continue  # Retry with new fixed template
                                                    else:
                                                        # Could not fix, try one more time with aggressive fix
                                                        self._log_to_slot(primary_slot_id, f"⚠️ Could not fix, trying aggressive fix...")
                                                        # The refeed_with_correction should have done aggressive fix, but if it didn't work, we'll continue
                                                        break  # Exit loop and mark as failed
                                                else:
                                                    # Not an event input error anymore, or max retries reached
                                                    break  # Exit loop
                                        else:
                                            # Failed to resubmit
                                            retry_submit_error = submission_retry.error_message or "Failed to resubmit fixed template"
                                            self._log_to_slot(primary_slot_id, f"❌ Failed to resubmit fixed template: {retry_submit_error}")
                                            current_error_msg = retry_submit_error
                                            break  # Exit loop

                                    # If we get here, all refeed retries failed
                                    completed_count['failed'] += 1
                                    self.slot_manager.release_slots(slot_ids, success=False, error=current_error_msg)
                                    self._update_slot_display(primary_slot_id, "FAILED", current_template_for_retry[:40] + "...",
                                                             f"❌ Failed (refeed)",
                                                             [f"❌ FAILED", f"After {refeed_attempt} refeed attempts", f"Last error: {current_error_msg[:30]}"])
                                    self._log_result(f"❌ [{template_index+1}] FAILED (refeed after {refeed_attempt} attempts): {current_error_msg}\n")
                                    return
                                else:
                                    # Could not fix template - LEARN FROM ERROR
                                    self._log_to_slot(primary_slot_id, "❌ Could not fix template")
                                    if self.workflow.generator.template_generator.template_validator:
                                        self.workflow.generator.template_generator.template_validator.learn_from_simulation_error(
                                            template, error_msg, None
                                        )

                            # If refeed failed or not available, mark as failed - LEARN FROM ERROR
                            if self.workflow.generator.template_generator.template_validator:
                                self.workflow.generator.template_generator.template_validator.learn_from_simulation_error(
                                    template, error_msg, None
                                )
                            completed_count['failed'] += 1
                            self.slot_manager.release_slots(slot_ids, success=False, error=error_msg)
                            self._update_slot_display(primary_slot_id, "FAILED", template[:40] + "...",
                                                     f"❌ Failed",
                                                     [f"❌ FAILED", error_msg[:50]])
                            self._log_result(f"❌ [{template_index+1}] FAILED: {error_msg}\n")
                    else:
                        completed_count['failed'] += 1
                        error_msg = submission.error_message or "Failed to submit"
                        self.slot_manager.release_slots(slot_ids, success=False, error=error_msg)
                        self._update_slot_display(primary_slot_id, "FAILED", template[:40] + "...",
                                                 f"❌ Failed to submit",
                                                 [f"❌ Failed to submit", error_msg[:80]])
                        self._log_to_slot(primary_slot_id, f"❌ Failed to submit: {error_msg}")
                        self._log_result(f"❌ [{template_index+1}] Failed to submit: {error_msg}\n")

                except Exception as e:
                    logger.error(f"Simulation error in slot: {e}", exc_info=True)
                    completed_count['failed'] += 1
                    error_msg = str(e)
                    self.slot_manager.release_slots(slot_ids, success=False, error=error_msg)
                    self._update_slot_display(primary_slot_id, "FAILED", template[:40] + "...",
                                             f"❌ Error",
                                             [f"❌ ERROR: {error_msg[:50]}"])
                    self._log_to_slot(primary_slot_id, f"❌ ERROR: {error_msg}")
                    self._log_result(f"❌ [{template_index+1}] ERROR: {error_msg}\n")

            def simulation_coordinator():
                """Coordinate concurrent simulations using slots"""
                try:
                    while template_queue and self.simulation_running:
                        # Try to assign slots to queued templates
                        assigned = False
                        for template_index, template_dict in list(template_queue):
                            if not self.simulation_running:
                                break

                            template_region = template_dict.get('region', region)
                            num_slots = self.slot_manager.get_slots_required(template_region)

                            # Try to find available slots
                            slot_ids = self.slot_manager.find_available_slots(num_slots)

                            if slot_ids:
                                # Assign slots and start simulation
                                assigned_slots = self.slot_manager.assign_slot(
                                    template_dict.get('template', ''),
                                    template_region,
                                    template_index
                                )

                                if assigned_slots:
                                    # Remove from queue
                                    template_queue.remove((template_index, template_dict))

                                    # Start simulation in thread
                                    thread = threading.Thread(
                                        target=run_simulation_in_slot,
                                        args=(template_index, template_dict, assigned_slots),
                                        daemon=True
                                    )
                                    thread.start()
                                    self.simulation_threads.append(thread)
                                    assigned = True

                        if not assigned:
                            # No slots available, wait a bit
                            time.sleep(1)

                        # Update progress
                        remaining = len(template_queue)
                        completed = completed_count['successful'] + completed_count['failed']
                        self.workflow.run_on_ui_thread(lambda: self.sim_progress_label.config(
                            text=f"Queue: {remaining}, Completed: {completed}/{completed_count['total']}"
                        ))

                    # Wait for all simulations to complete (non-blocking check, avoid joining current thread)
                    import threading as threading_module
                    current_thread = threading_module.current_thread()
                    max_wait_time = 3600  # Max 1 hour total wait
                    wait_start = time.time()

                    while self.simulation_threads and (time.time() - wait_start) < max_wait_time:
                        if self.stop_simulation_flag:
                            break
                        # Remove completed threads (check without joining)
                        alive_threads = []
                        for t in self.simulation_threads:
                            if t is current_thread:
                                # Skip current thread (coordinator thread)
                                continue
                            if t.is_alive():
                                alive_threads.append(t)
                            else:
                                # Thread completed, remove it
                                pass
                        self.simulation_threads = alive_threads

                        if self.simulation_threads:
                            # Still have threads running, wait a bit
                            time.sleep(2)  # Check every 2 seconds
                        else:
                            # All threads completed
                            break

                    # Final summary
                    self.workflow.run_on_ui_thread(lambda: self.sim_progress_label.config(text=""))
                    self._log_result("\n" + "=" * 80 + "\n")
                    self._log_result(f"📊 SIMULATION SUMMARY\n")
                    self._log_result(f"   Total: {completed_count['total']}\n")
                    self._log_result(f"   ✅ Successful: {completed_count['successful']}\n")
                    self._log_result(f"   ❌ Failed: {completed_count['failed']}\n")
                    self._log_result("=" * 80 + "\n")

                except Exception as e:
                    logger.error(f"Simulation coordinator error: {e}", exc_info=True)
                    self._log_result(f"\n❌ COORDINATOR ERROR: {str(e)}\n")
                finally:
                    self.simulation_running = False
                    self.workflow.run_on_ui_thread(lambda: self.simulate_button.config(state=tk.NORMAL))
                    self.workflow.run_on_ui_thread(lambda: self.simulate_all_button.config(state=tk.NORMAL))
                    self.workflow.run_on_ui_thread(lambda: self.stop_simulation_button.config(state=tk.DISABLED))
                    self.workflow.run_on_ui_thread(lambda: self.sim_progress_label.config(text=""))

            # Start coordinator thread
            coordinator_thread = threading.Thread(target=simulation_coordinator, daemon=True)
            coordinator_thread.start()
            self.simulation_threads.append(coordinator_thread)

            # Start slot update thread (refresh slot displays) - queue GUI updates on the Tk thread
            def update_slots_display():
                """Periodically update slot displays"""
                while self.simulation_running:
                    try:
                        if self.slot_manager:
                            for slot_id in range(self.slot_manager.max_slots):
                                slot = self.slot_manager.get_slot_status(slot_id)
                                if slot.status != SlotStatus.IDLE:
                                    # Schedule GUI updates on main thread
                                    self.workflow.run_on_ui_thread(lambda sid=slot_id, s=slot: self._update_slot_display(
                                        sid,
                                        s.status.value.upper(),
                                        s.template[:40] + "..." if s.template else "",
                                        f"Region: {s.region}" if s.region else "",
                                        s.get_logs()[-5:]  # Last 5 log lines
                                    ))
                                    # Also update progress if available
                                    if slot.progress_percent > 0 or slot.status == SlotStatus.RUNNING:
                                        self.workflow.run_on_ui_thread(lambda sid=slot_id, s=slot: self._update_slot_progress(
                                            sid,
                                            s.progress_percent,
                                            s.progress_message,
                                            s.api_status
                                        ))
                        time.sleep(2.0)  # Update every 2 seconds (reduced frequency to reduce memory usage)
                    except Exception as e:
                        logger.debug(f"Slot update error: {e}")
                        time.sleep(1)

            update_thread = threading.Thread(target=update_slots_display, daemon=True)
            update_thread.start()

        except ImportError as e:
            messagebox.showerror("Error", f"Failed to import simulation modules: {e}")
            self.simulation_running = False
            self.simulate_button.config(state=tk.NORMAL)
            self.simulate_all_button.config(state=tk.NORMAL)
            self.stop_simulation_button.config(state=tk.DISABLED)

    def _log_result(self, message: str):
        """Log result to results text widget"""
        self.workflow.run_on_ui_thread(lambda: self.results_text.insert(tk.END, message))
        self.workflow.run_on_ui_thread(lambda: self.results_text.see(tk.END))

    def _log_to_slot(self, slot_id: int, message: str):
        """Log message to a specific slot"""
        if slot_id in self.slot_widgets and self.slot_manager:
            slot = self.slot_manager.get_slot_status(slot_id)
            # Ensure message is a string, not a tuple
            if isinstance(message, (tuple, list)):
                message = ' '.join(str(m) for m in message) if message else ''
            elif message is None:
                message = ''
            else:
                message = str(message)
            slot.add_log(message)
            # Update display
            self._update_slot_display(slot_id, slot.status.value.upper(),
                                     slot.template[:40] + "..." if slot.template else "",
                                     f"Region: {slot.region}" if slot.region else "",
                                     slot.get_logs()[-5:])

    def _update_slot_display(self, slot_id: int, status: str, template: str, info: str, logs: List[str]):
        """Update slot display widget"""
        if slot_id not in self.slot_widgets:
            return

        def update():
            widget = self.slot_widgets[slot_id]

            # Update status
            status_colors = {
                'IDLE': COLORS.get('text_secondary', '#666666'),
                'RUNNING': COLORS.get('accent_yellow', '#ffff00'),
                'COMPLETED': COLORS.get('accent_green', '#00ff41'),
                'FAILED': '#ff0000'  # Red for failures
            }
            widget['status_label'].config(
                text=status,
                fg=status_colors.get(status, COLORS['text_primary'])
            )

            # Update template
            widget['template_label'].config(text=template if template else "")

            # Update log
            widget['log_text'].config(state=tk.NORMAL)
            widget['log_text'].delete(1.0, tk.END)
            for log_line in logs:
                widget['log_text'].insert(tk.END, log_line + "\n")
            widget['log_text'].config(state=tk.DISABLED)
            widget['log_text'].see(tk.END)

        self.workflow.run_on_ui_thread(update)

    def _update_slot_progress(self, slot_id: int, percent: float, message: str, api_status: str):
        """Update slot progress bar and message - OPTIMIZED to reduce memory usage"""
        if slot_id not in self.slot_widgets:
            return

        # Ensure percent is a float (handle string inputs from API)
        try:
            percent = float(percent)
        except (ValueError, TypeError):
            logger.warning(f"Invalid percent value: {percent}, using 0.0")
            percent = 0.0

        # Clamp percent to valid range [0, 100]
        percent = max(0.0, min(100.0, percent))

        # Throttle updates - only update if progress changed significantly (5% threshold)
        if not hasattr(self, '_last_progress'):
            self._last_progress = {}
        last_progress = self._last_progress.get(slot_id, -1)
        if abs(percent - last_progress) < 5.0 and percent < 100:
            return  # Skip update if change is small
        self._last_progress[slot_id] = percent

        def update():
            try:
                widget = self.slot_widgets[slot_id]

                # Update progress label
                widget['progress_label'].config(text=f"{int(percent)}%")

                # Update progress bar - use ttk.Progressbar instead of canvas to reduce memory
                # Canvas operations create DIBSECTION bitmaps which consume memory
                progress_bar = widget.get('progress_bar')
                if progress_bar and isinstance(progress_bar, ttk.Progressbar):
                    # Use ttk.Progressbar which is more memory efficient
                    progress_bar['value'] = percent
                else:
                    # Fallback to canvas only if ttk.Progressbar not available
                    canvas = widget['progress_bar']
                    # Avoid update_idletasks - use cached dimensions or defaults
                    try:
                        width = canvas.winfo_width()
                        height = canvas.winfo_height()
                    except:
                        width = 100
                        height = 12

                    if width <= 1:
                        width = 100
                    if height <= 1:
                        height = 12

                    # Only redraw if dimensions are valid
                    if width > 1 and height > 1:
                        canvas.delete("all")
                        canvas.create_rectangle(0, 0, width, height, fill=COLORS['bg_secondary'], outline="")

                        progress_width = int((percent / 100.0) * width)
                        if progress_width > 0:
                            if percent >= 100:
                                color = COLORS.get('accent_green', '#00ff41')
                            elif percent >= 50:
                                color = COLORS.get('accent_yellow', '#ffff00')
                            else:
                                color = COLORS.get('accent_cyan', '#00ffff')
                            canvas.create_rectangle(0, 0, progress_width, height, fill=color, outline="")

                # Update progress message
                widget['progress_text'].config(text=message[:25] if message else "")
            except Exception as e:
                logger.debug(f"Progress update error: {e}")

        self.workflow.run_on_ui_thread(update)
