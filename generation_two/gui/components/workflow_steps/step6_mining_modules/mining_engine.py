"""
Mining Engine Module
Core mining logic with self-sustaining operation
"""

import logging
import json
import time
import threading
from typing import Optional, Callable, List
from generation_two.core.mining import MiningCoordinator, SearchStrategy
from generation_two.core.slot_manager import SlotManager, SlotStatus
from generation_two.core.simulator_tester import SimulatorTester, SimulationSettings
from generation_two.core.region_config import (
    REGION_DEFAULT_UNIVERSE,
    REGION_DEFAULT_NEUTRALIZATION,
    REGION_NEUTRALIZATIONS,
)

logger = logging.getLogger(__name__)


class MiningEngine:
    """
    Core mining engine with self-sustaining operation
    Handles generation, simulation, and error recovery
    """
    
    def __init__(
        self,
        generator,
        simulator_tester,
        backtest_storage,
        slot_manager: SlotManager,
        correlation_tracker,
        duplicate_detector,
        search_strategy_manager,
        sim_counter,
        log_callback: Optional[Callable[[str], None]] = None,
        update_slot_callback: Optional[Callable] = None
    ):
        """
        Initialize mining engine
        
        Args:
            generator: Template generator
            simulator_tester: Simulator tester
            backtest_storage: Backtest storage
            slot_manager: Slot manager
            correlation_tracker: Correlation tracker
            duplicate_detector: Duplicate detector
            search_strategy_manager: Search strategy manager
            sim_counter: Simulation counter
            log_callback: Log callback
            update_slot_callback: Slot update callback
        """
        self.generator = generator
        self.simulator_tester = simulator_tester
        self.backtest_storage = backtest_storage
        self.slot_manager = slot_manager
        self.correlation_tracker = correlation_tracker
        self.duplicate_detector = duplicate_detector
        self.search_strategy = search_strategy_manager
        self.sim_counter = sim_counter
        self.log_callback = log_callback
        self.update_slot_callback = update_slot_callback
        
        self.mining_active = False
        self.stop_flag = False
        self.generated_templates_queue = []
        self.error_recovery_count = 0
        self.max_error_recovery = 10  # Max consecutive errors before pause
        
        # Long-term operation: memory management
        self.max_queue_size = 100  # Max templates in queue
        self.cleanup_interval = 3600  # Cleanup every hour
        self.last_cleanup_time = time.time()
        self.operation_start_time = time.time()
        self.templates_generated_count = 0
        self.simulations_completed_count = 0
        self.robustness_max_tests_per_alpha = 6
        self.robustness_truncations = [0.08, 0.05, 0.03, 0.01]
        self.robustness_decays = [0, 3, 5]
        self.robustness_neutralization_preferences = [
            'INDUSTRY',
            'SUBINDUSTRY',
            'SECTOR',
            'STATISTICAL',
            'CROWDING',
        ]
        self.improvement_max_variants_per_alpha = 1
        self.improvement_strong_sharpe = 1.10
        self.improvement_strong_fitness = 0.70
        self.improvement_balanced_sharpe = 1.00
        self.improvement_balanced_fitness = 0.80
        self.improvement_repairable_sharpe = 0.95
        self.improvement_repairable_fitness = 0.75

    @staticmethod
    def _format_metric(value, precision: int = 2) -> str:
        """Format optional numeric metrics from the simulation API."""
        try:
            if value is None:
                return "N/A"
            return f"{float(value):.{precision}f}"
        except (TypeError, ValueError):
            return "N/A"

    def _validate_before_submit(self, template: str, region: str):
        """Run local expression checks before spending a remote simulation call."""
        if not self.generator or not self.generator.template_generator:
            return None

        available_operators = None
        if hasattr(self.generator.template_generator, 'operator_fetcher'):
            operator_fetcher = self.generator.template_generator.operator_fetcher
            available_operators = operator_fetcher.operators if operator_fetcher else None

        available_fields = self.generator.template_generator.get_data_fields_for_region(region)
        if not available_operators or not available_fields:
            logger.debug("Skipping mining local validation: missing operators or fields")
            return None

        try:
            from generation_two.core.local_expression_validator import validate_expression_locally

            return validate_expression_locally(template, available_operators, available_fields)
        except Exception as e:
            logger.debug(f"Mining local expression validation failed unexpectedly: {e}")
            return None

    def _normalize_before_submit(self, template: str) -> str:
        """Apply local operator-parameter normalization before validation/submission."""
        if not self.generator or not self.generator.template_generator:
            return template

        available_operators = None
        if hasattr(self.generator.template_generator, 'operator_fetcher'):
            operator_fetcher = self.generator.template_generator.operator_fetcher
            available_operators = operator_fetcher.operators if operator_fetcher else None

        if not available_operators:
            return template

        try:
            from generation_two.core.operator_parameter_normalizer import normalize_operator_parameters

            normalized_template, fixes = normalize_operator_parameters(template, available_operators)
            if fixes and normalized_template != template:
                logger.info(f"Applied mining pre-submit operator normalization: {fixes}")
                return normalized_template
        except Exception as e:
            logger.debug(f"Mining pre-submit operator normalization skipped: {e}")

        return template

    def _parse_result_tags(self, result) -> List[str]:
        """Parse JSON tags stored on a simulation result."""
        tags_raw = getattr(result, 'tags', '') or ''
        if not tags_raw:
            return []
        try:
            tags = json.loads(tags_raw)
            if isinstance(tags, list):
                return [str(tag) for tag in tags]
        except (TypeError, ValueError):
            logger.debug(f"Could not parse result tags: {tags_raw}")
        return []

    @staticmethod
    def _safe_float(value, default: float = 0.0) -> float:
        """Convert optional metric values to float."""
        try:
            if value is None:
                return default
            return float(value)
        except (TypeError, ValueError):
            return default

    def _parse_result_checks(self, result) -> List:
        """Parse JSON checks stored on a simulation result."""
        checks_raw = getattr(result, 'checks', '') or ''
        if not checks_raw:
            return []
        try:
            checks = json.loads(checks_raw)
            return checks if isinstance(checks, list) else []
        except (TypeError, ValueError):
            logger.debug(f"Could not parse result checks: {checks_raw[:120]}")
        return []

    def _should_run_robustness_sweep(self, result) -> bool:
        """Run robustness only for candidates and promising near-misses."""
        if not result:
            return False

        tags = set(self._parse_result_tags(result))
        if 'robustness' in tags:
            return False
        if result.success or 'candidate' in tags:
            return True
        return 'near-miss' in tags and 'promising' in tags

    def _settings_key(self, settings: SimulationSettings) -> tuple:
        """Deduplicate robustness settings."""
        return (
            settings.neutralization,
            settings.truncation,
            settings.decay,
            settings.delay,
            settings.testPeriod,
        )

    def _build_robustness_settings(
        self,
        region: str,
        base_settings: SimulationSettings
    ) -> List[SimulationSettings]:
        """Build a small, budgeted settings grid around a promising alpha."""
        supported_neutralizations = REGION_NEUTRALIZATIONS.get(region, ['INDUSTRY'])
        neutralizations = []
        for neutralization in [base_settings.neutralization] + self.robustness_neutralization_preferences:
            if neutralization in supported_neutralizations and neutralization not in neutralizations:
                neutralizations.append(neutralization)

        candidates = []
        seen = {self._settings_key(base_settings)}

        # First vary one dimension at a time around the initial successful setting.
        for truncation in self.robustness_truncations:
            settings = SimulationSettings(
                region=region,
                universe=base_settings.universe,
                neutralization=base_settings.neutralization,
                truncation=truncation,
                decay=base_settings.decay,
                delay=base_settings.delay,
                testPeriod=base_settings.testPeriod
            )
            key = self._settings_key(settings)
            if key not in seen:
                candidates.append(settings)
                seen.add(key)

        for decay in self.robustness_decays:
            settings = SimulationSettings(
                region=region,
                universe=base_settings.universe,
                neutralization=base_settings.neutralization,
                truncation=base_settings.truncation,
                decay=decay,
                delay=base_settings.delay,
                testPeriod=base_settings.testPeriod
            )
            key = self._settings_key(settings)
            if key not in seen:
                candidates.append(settings)
                seen.add(key)

        for neutralization in neutralizations:
            settings = SimulationSettings(
                region=region,
                universe=base_settings.universe,
                neutralization=neutralization,
                truncation=base_settings.truncation,
                decay=base_settings.decay,
                delay=base_settings.delay,
                testPeriod=base_settings.testPeriod
            )
            key = self._settings_key(settings)
            if key not in seen:
                candidates.append(settings)
                seen.add(key)

        # Then add a few conservative combinations if budget remains.
        for truncation in self.robustness_truncations[1:]:
            for decay in self.robustness_decays[1:]:
                settings = SimulationSettings(
                    region=region,
                    universe=base_settings.universe,
                    neutralization=base_settings.neutralization,
                    truncation=truncation,
                    decay=decay,
                    delay=base_settings.delay,
                    testPeriod=base_settings.testPeriod
                )
                key = self._settings_key(settings)
                if key not in seen:
                    candidates.append(settings)
                    seen.add(key)
                if len(candidates) >= self.robustness_max_tests_per_alpha:
                    return candidates[:self.robustness_max_tests_per_alpha]

        return candidates[:self.robustness_max_tests_per_alpha]

    def _mark_robustness_result(self, result):
        """Add robustness tags to a result before persistence."""
        tags = self._parse_result_tags(result)
        for tag in ['robustness', 'candidate-robust' if result.success else 'robustness-failed']:
            if tag not in tags:
                tags.append(tag)
        result.tags = json.dumps(tags)
        return result

    def _should_refeed_error(self, error_message: str) -> bool:
        """Only refeed errors that can plausibly be fixed by editing syntax."""
        message = (error_message or "").lower()
        if not message:
            return False

        structural_keywords = [
            'syntax',
            'parse',
            'parser',
            'compile',
            'compiler',
            'unbalanced',
            'parentheses',
            'unknown variable',
            'undefined variable',
            'invalid field',
            'field not found',
            'unknown field',
            'type error',
            'type mismatch',
            'incompatible',
            'operator',
            'operand',
            'placeholder',
            'data_field',
            'does not support event inputs',
            'local validation failed',
            'submit failed',
        ]
        if any(keyword in message for keyword in structural_keywords):
            return True

        # IS check failures such as LOW_SHARPE or CONCENTRATED_WEIGHT are
        # outcome problems, not expression-structure problems. Refeeding them
        # as syntax fixes burns LLM calls and simulation budget.
        if message.startswith('failed checks:') or message.startswith('warning checks:'):
            return False

        outcome_keywords = [
            'low_sharpe',
            'low_fitness',
            'low_turnover',
            'high_turnover',
            'concentrated_weight',
            'sub_universe',
            'self_correlation',
            'matches_competition',
            'ladder',
        ]
        if any(keyword in message for keyword in outcome_keywords):
            return False

        return False

    def _should_run_improvement_attempt(self, result) -> bool:
        """Use LLM improvement only for near-misses with enough signal to be worth one more test."""
        if not result or result.success:
            return False

        tags = set(self._parse_result_tags(result))
        if 'improved' in tags or 'robustness' in tags:
            return False
        if 'near-miss' not in tags:
            return False

        sharpe = self._safe_float(getattr(result, 'sharpe', 0.0))
        fitness = self._safe_float(getattr(result, 'fitness', 0.0))
        repairable_tags = {'concentration', 'subuniverse', 'turnover', 'high-turnover'}

        if sharpe >= self.improvement_strong_sharpe and fitness >= self.improvement_strong_fitness:
            return True
        if sharpe >= self.improvement_balanced_sharpe and fitness >= self.improvement_balanced_fitness:
            return True
        if (
            tags.intersection(repairable_tags)
            and sharpe >= self.improvement_repairable_sharpe
            and fitness >= self.improvement_repairable_fitness
        ):
            return True

        return False

    def _get_available_operator_names(self) -> List[str]:
        """Return operator names available to the template generator."""
        if not self.generator or not self.generator.template_generator:
            return []
        operator_fetcher = getattr(self.generator.template_generator, 'operator_fetcher', None)
        operators = getattr(operator_fetcher, 'operators', None) if operator_fetcher else None
        try:
            from generation_two.core.selection_safety import filter_generation_operators
            operators = filter_generation_operators(operators)
        except Exception as e:
            logger.debug(f"Operator safety filtering skipped for improve prompt: {e}")
        names = []
        for operator in operators or []:
            if isinstance(operator, dict) and operator.get('name'):
                names.append(str(operator['name']))
            elif isinstance(operator, str):
                names.append(operator)
        return names

    def _get_available_field_ids(self, region: str) -> List[str]:
        """Return field IDs available to the template generator for a region."""
        if not self.generator or not self.generator.template_generator:
            return []
        fields = self.generator.template_generator.get_data_fields_for_region(region)
        field_ids = []
        for field in fields or []:
            if isinstance(field, dict):
                field_id = field.get('id') or field.get('name')
                if field_id:
                    field_ids.append(str(field_id))
            elif isinstance(field, str):
                field_ids.append(field)
        return field_ids

    def _extract_llm_expression(self, response: str) -> str:
        """Extract a FASTEXPR expression from an LLM response."""
        validator = None
        if self.generator and self.generator.template_generator:
            validator = getattr(self.generator.template_generator, 'template_validator', None)
        if validator and hasattr(validator, '_extract_expression_from_response'):
            return validator._extract_expression_from_response(response)

        if not response:
            return ""
        response = response.strip().strip('`').strip('"\'')
        lines = [line.strip().strip('`').strip('"\'') for line in response.splitlines() if line.strip()]
        candidates = [line.rstrip('.,;') for line in lines if '(' in line and ')' in line]
        return max(candidates, key=len) if candidates else (lines[0] if lines else "")

    def _build_improvement_prompt(
        self,
        result,
        region: str,
        settings: SimulationSettings,
        available_operators: List[str],
        available_fields: List[str]
    ) -> str:
        """Build a performance-oriented alpha improvement prompt."""
        tags = self._parse_result_tags(result)
        checks = self._parse_result_checks(result)
        failed_checks = []
        pending_checks = []
        for check in checks:
            if not isinstance(check, dict):
                continue
            name = str(check.get('name') or check.get('title') or check.get('type') or 'check')
            state = str(check.get('result') or check.get('status') or check.get('severity') or '').upper()
            value = check.get('value')
            limit = check.get('limit')
            entry = f"{name}={state}"
            if value is not None:
                entry += f" value={value}"
            if limit is not None:
                entry += f" limit={limit}"
            if state in {'FAIL', 'FAILED', 'ERROR', 'WARNING', 'WARN'}:
                failed_checks.append(entry)
            elif state == 'PENDING':
                pending_checks.append(entry)

        guidance = []
        tags_set = set(tags)
        if 'concentration' in tags_set:
            guidance.append("Reduce concentrated weights by keeping the signal continuous: prefer rank/normalize/winsorize/zscore and avoid sign/sqrt-style discretization or compression.")
        if 'subuniverse' in tags_set:
            guidance.append("Improve sub-universe robustness by using broader, smoother signals and avoiding fragile single-field spikes.")
        if 'turnover' in tags_set or 'high-turnover' in tags_set:
            guidance.append("Reduce turnover by adding smoothing/time-series aggregation or less reactive transforms.")
        if 'sharpe' in tags_set or 'fitness' in tags_set:
            guidance.append("Improve risk-adjusted signal quality by denoising, changing direction when sensible, and combining one complementary transform.")
        if not guidance:
            guidance.append("Make one or two small economically plausible changes that improve robustness without inventing unavailable fields.")

        return f"""You are improving a WorldQuant Brain FASTEXPR alpha expression.

Goal: return ONE improved FASTEXPR expression that can be submitted directly.

Original expression:
{result.template}

Region: {region}
Settings: universe={settings.universe}, delay={settings.delay}, neutralization={settings.neutralization}, decay={settings.decay}, truncation={settings.truncation}

Current metrics:
Sharpe={self._format_metric(getattr(result, 'sharpe', None), 4)}
Fitness={self._format_metric(getattr(result, 'fitness', None), 4)}
Turnover={self._format_metric(getattr(result, 'turnover', None), 4)}
Returns={self._format_metric(getattr(result, 'returns', None), 4)}
Drawdown={self._format_metric(getattr(result, 'drawdown', None), 4)}
Margin={self._format_metric(getattr(result, 'margin', None), 6)}

Failed or warning checks:
{chr(10).join(failed_checks) if failed_checks else 'None'}

Pending checks:
{chr(10).join(pending_checks) if pending_checks else 'None'}

Tags:
{', '.join(tags) if tags else 'none'}

Improvement priorities:
{chr(10).join('- ' + item for item in guidance)}

Constraints:
- Return ONLY the improved FASTEXPR expression.
- Do not explain, do not return markdown, do not return JSON.
- Preserve the core economic idea, but change one or two expression-level details.
- Prefer simple robust transforms such as rank, zscore, winsorize, normalize, ts_mean, ts_rank, reverse, subtract, add, and divide when appropriate.
- Avoid adding sign or sqrt unless the original expression already uses it and removing it would clearly break the core idea.
- Use only operators from this available list when possible: {', '.join(available_operators[:80])}
- Use exact available field IDs; prefer reusing original fields unless replacing a weak field is necessary.
- Available fields sample: {', '.join(available_fields[:80])}
- Do not use placeholders like DATA_FIELD1 or OPERATOR1.
- Make the expression syntactically valid FASTEXPR with balanced parentheses.

Improved FASTEXPR expression:"""

    def _generate_improved_template(self, result, region: str, settings: SimulationSettings) -> str:
        """Ask the configured LLM for one performance-improved expression."""
        if not self.generator or not self.generator.template_generator:
            return ""

        ollama_manager = getattr(self.generator.template_generator, 'ollama_manager', None)
        if not ollama_manager:
            return ""

        available_operators = self._get_available_operator_names()
        available_fields = self._get_available_field_ids(region)
        prompt = self._build_improvement_prompt(
            result=result,
            region=region,
            settings=settings,
            available_operators=available_operators,
            available_fields=available_fields
        )

        response = ollama_manager.generate(
            prompt,
            temperature=0.35,
            max_tokens=700,
            progress_callback=None
        )
        improved_template = self._extract_llm_expression(response or "")
        if not improved_template:
            return ""

        improved_template = self._normalize_before_submit(improved_template)
        if improved_template.strip() == (result.template or "").strip():
            logger.info("LLM improvement returned the original expression; skipping")
            return ""

        return improved_template

    def _mark_improvement_result(self, result, source_result):
        """Add improvement tags to a simulation result."""
        tags = self._parse_result_tags(result)
        for tag in [
            'improved',
            'improved-from-near-miss' if not getattr(source_result, 'success', False) else 'improved-from-candidate',
            'improved-candidate' if result.success else 'improved-failed',
        ]:
            if tag not in tags:
                tags.append(tag)
        result.tags = json.dumps(tags)
        return result

    def _run_improvement_attempt(
        self,
        slot_id: int,
        source_result,
        region: str,
        settings: SimulationSettings
    ):
        """Run one LLM-driven expression improvement attempt for a promising near-miss."""
        if not self._should_run_improvement_attempt(source_result):
            return None
        if not self.sim_counter.can_simulate():
            self._log("⚠️ Daily simulation limit reached before improvement attempt")
            return None

        slot = self.slot_manager.get_slot_status(slot_id)
        if slot:
            slot.add_log("🧠 LLM improve: generating expression variant")
        self._log(f"🧠 {region}: LLM improve queued for near-miss alpha")

        improved_template = self._generate_improved_template(source_result, region, settings)
        if not improved_template:
            if slot:
                slot.add_log("🧠 LLM improve skipped: no usable expression returned")
            return None

        import re
        remaining_ops = re.findall(r'\b(OPERATOR\d+|operator\d+|Operator\d+)\b', improved_template, re.IGNORECASE)
        remaining_fields = re.findall(r'\b(DATA_FIELD\d+|data_field\d+|Data_Field\d+)\b', improved_template, re.IGNORECASE)
        if remaining_ops or remaining_fields:
            if slot:
                slot.add_log(f"🧠 LLM improve skipped: placeholders remain {remaining_ops + remaining_fields}")
            return None

        if self.backtest_storage and self.backtest_storage.has_been_simulated(improved_template, region):
            if slot:
                slot.add_log("🧠 LLM improve skipped: expression was already simulated")
            return None

        local_validation = self._validate_before_submit(improved_template, region)
        if local_validation and not local_validation.is_valid:
            error_msg = local_validation.summary()
            if slot:
                slot.add_log(f"🧠 LLM improve skipped: local validation failed: {error_msg[:80]}")
            logger.info("LLM improved expression failed local validation: %s", error_msg)
            return None

        status = self.sim_counter.increment_count()
        if not status['can_simulate']:
            self._log("⚠️ Daily simulation limit reached before improvement submit")
            return None

        if slot:
            slot.add_log(f"🧠 LLM improve submit: {improved_template[:100]}...")
        self._update_slot(slot_id, improved_template, region, 10, "Submitting LLM improvement...")

        submission = self.simulator_tester.submit_simulation(improved_template, region, settings)
        if not submission:
            submit_error = submission.error_message or "Failed to submit LLM improvement"
            if slot:
                slot.add_log(f"❌ LLM improve submit failed: {submit_error[:80]}")
            self._log(f"❌ {region}: LLM improve submit failed: {submit_error[:100]}")
            return None

        def progress_callback(percent, message, api_status):
            self.slot_manager.update_slot_progress(slot_id, percent=percent, message=message, api_status=api_status)
            active_slot = self.slot_manager.get_slot_status(slot_id)
            if active_slot:
                active_slot.add_log(f"[IMPROVE {api_status}] {message}")
            self._update_slot(slot_id, improved_template, region, percent, f"LLM improve: {message}")

        try:
            result = self.simulator_tester.monitor_simulation(
                submission.progress_url,
                improved_template,
                region,
                settings,
                progress_callback=progress_callback
            )
        finally:
            self.simulator_tester.release_simulation_slot(submission)

        if not result:
            return None

        result = self._mark_improvement_result(result, source_result)
        if self.backtest_storage:
            self.backtest_storage.store_result(result)

        if result.success and result.alpha_id:
            self.correlation_tracker.update_template_alpha_mapping(improved_template, str(result.alpha_id))
            self.search_strategy.add_successful_template(improved_template, region)

        outcome = "PASS" if result.success else "FAIL"
        self._log(
            f"🧠 {region}: LLM improve {outcome} "
            f"Sharpe={self._format_metric(result.sharpe)}, Fitness={self._format_metric(result.fitness)}"
        )
        return result

    def _run_robustness_sweep(
        self,
        slot_id: int,
        template: str,
        region: str,
        base_settings: SimulationSettings,
        base_result
    ) -> List:
        """Run a budgeted robustness sweep for a promising alpha."""
        if not self._should_run_robustness_sweep(base_result):
            return []

        settings_grid = self._build_robustness_settings(region, base_settings)
        if not settings_grid:
            return []

        slot = self.slot_manager.get_slot_status(slot_id)
        if slot:
            slot.add_log(f"🧪 Robustness sweep: {len(settings_grid)} settings")
        self._log(f"🧪 {region}: Robustness sweep queued ({len(settings_grid)} tests)")

        results = []
        for idx, settings in enumerate(settings_grid, start=1):
            if self.stop_flag:
                break
            if not self.sim_counter.can_simulate():
                self._log("⚠️ Daily simulation limit reached during robustness sweep")
                break

            status = self.sim_counter.increment_count()
            if not status['can_simulate']:
                self._log("⚠️ Daily simulation limit reached during robustness sweep")
                break

            slot = self.slot_manager.get_slot_status(slot_id)
            if slot:
                slot.add_log(
                    f"🧪 Robustness {idx}/{len(settings_grid)}: "
                    f"trunc={settings.truncation}, decay={settings.decay}, neut={settings.neutralization}"
                )

            submission = self.simulator_tester.submit_simulation(template, region, settings)
            if not submission:
                submit_error = submission.error_message or "Failed to submit robustness test"
                if slot:
                    slot.add_log(f"❌ Robustness submit failed: {submit_error[:80]}")
                continue

            def progress_callback(percent, message, api_status):
                self.slot_manager.update_slot_progress(slot_id, percent=percent, message=message, api_status=api_status)
                active_slot = self.slot_manager.get_slot_status(slot_id)
                if active_slot:
                    active_slot.add_log(f"[ROBUST {api_status}] {message}")
                self._update_slot(slot_id, template, region, percent, f"Robustness {idx}/{len(settings_grid)}")

            try:
                result = self.simulator_tester.monitor_simulation(
                    submission.progress_url,
                    template,
                    region,
                    settings,
                    progress_callback=progress_callback
                )
            finally:
                self.simulator_tester.release_simulation_slot(submission)

            if result:
                result = self._mark_robustness_result(result)
                results.append(result)
                if self.backtest_storage:
                    self.backtest_storage.store_result(result)

                if result.success and result.alpha_id:
                    self.correlation_tracker.update_template_alpha_mapping(template, str(result.alpha_id))
                    self.search_strategy.add_successful_template(template, region)

                sharpe = self._format_metric(result.sharpe)
                fitness = self._format_metric(result.fitness)
                outcome = "PASS" if result.success else "FAIL"
                self._log(
                    f"🧪 {region}: Robustness {outcome} "
                    f"trunc={settings.truncation}, decay={settings.decay}, "
                    f"neut={settings.neutralization}, Sharpe={sharpe}, Fitness={fitness}"
                )

        return results
    
    def start(self):
        """Start mining engine"""
        if self.mining_active:
            return
        
        self.mining_active = True
        self.stop_flag = False
        self.error_recovery_count = 0
        
        # Start main loop
        thread = threading.Thread(target=self._main_loop, daemon=True, name="MiningEngine")
        thread.start()
        
        self._log("✅ Mining engine started")
    
    def stop(self):
        """Stop mining engine"""
        self.mining_active = False
        self.stop_flag = True
        self._log("⏹ Mining engine stopping...")
    
    def _main_loop(self):
        """Main mining loop with error recovery and long-term operation support"""
        while self.mining_active and not self.stop_flag:
            try:
                # Periodic cleanup for long-term operation
                current_time = time.time()
                if current_time - self.last_cleanup_time > self.cleanup_interval:
                    self._periodic_cleanup()
                    self.last_cleanup_time = current_time
                
                # Check simulation limit
                if not self.sim_counter.can_simulate():
                    self._log("⚠️ Daily simulation limit reached. Waiting...")
                    time.sleep(3600)
                    continue
                
                # Process simulations (now handles generation internally with 20/80 logic)
                self._process_simulations()
                
                # Reset error recovery on success
                self.error_recovery_count = 0
                
                time.sleep(1)
                
            except Exception as e:
                self.error_recovery_count += 1
                logger.error(f"Mining engine error (recovery {self.error_recovery_count}): {e}", exc_info=True)
                self._log(f"❌ Error: {str(e)[:100]} (recovery {self.error_recovery_count}/{self.max_error_recovery})")
                
                # Exponential backoff for long-term stability
                backoff_time = min(5 * (2 ** min(self.error_recovery_count, 5)), 300)  # Max 5 minutes
                
                if self.error_recovery_count >= self.max_error_recovery:
                    self._log(f"⚠️ Too many errors, pausing for {backoff_time} seconds...")
                    time.sleep(backoff_time)
                    self.error_recovery_count = 0
                else:
                    time.sleep(backoff_time)
    
    def _generate_new_template_for_mining(self) -> tuple:
        """
        Generate a new placeholder template that's different from database
        
        Returns:
            (template, region) tuple or (None, None) if failed
        """
        try:
            region = self.search_strategy.get_next_region()
            if not region:
                return (None, None)
            
            # Use algorithmic generation (like Step 4) - generates placeholders
            from generation_two.core.algorithmic_template_generator import AlgorithmicTemplateGenerator
            from generation_two.core.selection_safety import filter_generation_fields, filter_generation_operators
            
            raw_operators = self.generator.template_generator.operator_fetcher.operators if self.generator.template_generator.operator_fetcher else None
            raw_fields = self.generator.template_generator.get_data_fields_for_region(region)
            available_operators = filter_generation_operators(raw_operators)
            available_fields = filter_generation_fields(raw_fields)
            
            if not available_operators or not available_fields:
                logger.warning(f"No operators or fields available for {region}")
                return (None, None)
            
            # Generate placeholder template algorithmically
            generator = AlgorithmicTemplateGenerator(available_operators, available_fields)
            
            # Try up to 10 times to generate a template different from database
            max_retries = 10
            for retry in range(max_retries):
                template_with_placeholders = generator.generate_placeholder_expression(
                    max_operators=5,
                    method='random'  # Can be 'random', 'brownian', 'tree', 'linear'
                )
                
                if not template_with_placeholders:
                    continue
                
                # Check if different from database
                if self.backtest_storage:
                    existing_templates = self.backtest_storage.get_all_templates(region=region, limit=1000)
                    if template_with_placeholders not in existing_templates:
                        # Check duplicates
                        is_dup, reason = self.duplicate_detector.is_duplicate(template_with_placeholders, region)
                        if not is_dup:
                            # Store in database
                            from generation_two.core import template_similarity
                            similarity_checker = template_similarity.TemplateSimilarityChecker()
                            operators_used = list(similarity_checker.extract_operators(template_with_placeholders))
                            fields_used = list(similarity_checker.extract_fields(template_with_placeholders))
                            
                            self.backtest_storage.store_template(
                                template=template_with_placeholders,
                                region=region,
                                operators_used=operators_used,
                                fields_used=fields_used
                            )
                            
                            self.templates_generated_count += 1
                            return (template_with_placeholders, region)
                    else:
                        logger.debug(f"Generated template already exists in database, retrying...")
                else:
                    # No database, just check duplicates
                    is_dup, reason = self.duplicate_detector.is_duplicate(template_with_placeholders, region)
                    if not is_dup:
                        self.templates_generated_count += 1
                        return (template_with_placeholders, region)
            
            logger.debug(f"Could not generate unique template after {max_retries} retries")
            return (None, None)
                    
        except Exception as e:
            logger.debug(f"Error generating template: {e}")
            return (None, None)
    
    def _pick_unsimulated_template_from_db(self) -> tuple:
        """
        Pick a random unsimulated template from database
        
        Returns:
            (template, region) tuple or (None, None) if none available
        """
        try:
            if not self.backtest_storage:
                return (None, None)
            
            # Get unsimulated templates from all regions or specific region
            region = self.search_strategy.get_next_region()
            unsimulated = self.backtest_storage.get_unsimulated_templates(region=region, limit=200)
            
            if not unsimulated:
                # Try all regions if specific region has none
                if region:
                    unsimulated = self.backtest_storage.get_unsimulated_templates(region=None, limit=200)
            
            if unsimulated:
                import random
                # Filter out templates currently being simulated (check slot manager)
                available_templates = []
                for template, template_region in unsimulated:
                    # Check if this template is currently in any slot
                    is_in_use = False
                    for slot_id in range(self.slot_manager.max_slots):
                        slot = self.slot_manager.get_slot_status(slot_id)
                        if slot and slot.template == template and slot.region == template_region:
                            is_in_use = True
                            break
                    
                    if not is_in_use:
                        # Double-check it hasn't been simulated (race condition protection)
                        if not self.backtest_storage.has_been_simulated(template, template_region):
                            available_templates.append((template, template_region))
                
                if available_templates:
                    template, template_region = random.choice(available_templates)
                    return (template, template_region)
            
            return (None, None)
            
        except Exception as e:
            logger.debug(f"Error picking unsimulated template: {e}")
            return (None, None)
    
    def _generate_templates_batch(self, batch_size: int = 5):
        """Generate a batch of templates using algorithmic generation with placeholders (legacy method, kept for compatibility)"""
        # This method is now mostly unused since we generate on-demand in _process_simulations
        # But keeping it for backward compatibility
        pass
    
    def _process_simulations(self):
        """Process pending simulations with 20/80 logic: 20% new generation, 80% reuse from database"""
        import random
        
        # Determine how many slots we need to fill.
        available_slots = self.slot_manager.max_slots - len([
            slot_id
            for slot_id in range(self.slot_manager.max_slots)
            if self.slot_manager.get_slot_status(slot_id).status != SlotStatus.IDLE
        ])
        if available_slots <= 0:
            return
        
        selected_templates = []
        
        # 20% chance to generate new, 80% chance to reuse from database
        for _ in range(available_slots):
            if self.stop_flag:
                break
            
            # 20% chance: Generate new placeholder template
            if random.random() < 0.2:
                template, region = self._generate_new_template_for_mining()
                if template and region:
                    selected_templates.append((template, region))
                    self._log(f"🆕 Generated new template for {region}")
            else:
                # 80% chance: Pick unsimulated template from database
                template, region = self._pick_unsimulated_template_from_db()
                if template and region:
                    selected_templates.append((template, region))
                    self._log(f"♻️ Reusing unsimulated template for {region}")
                else:
                    # Fallback: Generate new if no unsimulated templates available
                    template, region = self._generate_new_template_for_mining()
                    if template and region:
                        selected_templates.append((template, region))
                        self._log(f"🆕 Fallback: Generated new template for {region}")
        
        if not selected_templates:
            return
        
        # Select low-correlation templates from selected batch
        if len(selected_templates) > 1:
            low_corr_templates = self.correlation_tracker.get_low_correlation_templates(
                selected_templates,
                max_correlation=0.3,
                limit=len(selected_templates)
            )
            if low_corr_templates:
                selected_templates = [(t, r) for t, r, _ in low_corr_templates]
        
        # Submit simulations
        for template, region in selected_templates:
            if self.stop_flag:
                break
            
            # Check simulation limit
            status = self.sim_counter.increment_count()
            if not status['can_simulate']:
                self._log("⚠️ Daily simulation limit reached")
                break
            
            # Assign slot (GLB uses 2 slots)
            slot_count = 2 if region == 'GLB' else 1
            slot_ids = self.slot_manager.find_available_slots(slot_count)
            
            if not slot_ids:
                # Wait for slots
                wait_count = 0
                while wait_count < 10 and not self.stop_flag:
                    time.sleep(2)
                    wait_count += 1
                    slot_ids = self.slot_manager.find_available_slots(slot_count)
                    if slot_ids:
                        break
                
                if not slot_ids:
                    self._log(f"⚠️ No slots available for {region}")
                    continue
            
            # Assign slots
            assigned_slots = self.slot_manager.assign_slot(template, region, 0)
            if not assigned_slots:
                continue
            
            # Start simulation in thread
            for slot_id in assigned_slots:
                thread = threading.Thread(
                    target=self._run_simulation,
                    args=(slot_id, template, region),
                    daemon=True
                )
                thread.start()
            
            time.sleep(0.5)  # Brief pause between submissions
    
    def _run_simulation(self, slot_id: int, template: str, region: str):
        """Run a single simulation with placeholder replacement"""
        try:
            # Replace placeholders using Ollama selection (like Step 5)
            template_with_placeholders = template
            if self.generator and self.generator.template_generator:
                available_operators = None
                if hasattr(self.generator.template_generator, 'operator_fetcher'):
                    available_operators = self.generator.template_generator.operator_fetcher.operators if self.generator.template_generator.operator_fetcher else None
                
                available_fields = self.generator.template_generator.get_data_fields_for_region(region)
                
                # Check if template has placeholders
                has_operator_placeholders = template and ('OPERATOR' in template.upper() or 'operator' in template.lower())
                has_field_placeholders = template and ('DATA_FIELD' in template.upper() or 'data_field' in template.lower())
                
                if (has_operator_placeholders or has_field_placeholders) and available_operators and available_fields:
                    slot = self.slot_manager.get_slot_status(slot_id)
                    if slot:
                        slot.add_log("🤖 Asking Ollama to select operators and fields...")
                    
                    def progress_callback(msg):
                        slot = self.slot_manager.get_slot_status(slot_id)
                        if slot:
                            slot.add_log(f"🤖 {msg}")
                    
                    # Use Ollama to select and replace
                    if hasattr(self.generator.template_generator, 'ollama_manager'):
                        # Get backtest_storage for field usage tracking
                        backtest_storage = None
                        if hasattr(self.generator, 'backtest_storage'):
                            backtest_storage = self.generator.backtest_storage
                        
                        replaced = self.generator.template_generator.ollama_manager.replace_placeholders_with_selection(
                            template,
                            available_operators,
                            available_fields,
                            progress_callback=progress_callback,
                            region=region,
                            backtest_storage=backtest_storage
                        )
                        if replaced:
                            template = replaced
                            # Verify all placeholders were actually replaced
                            import re
                            remaining_ops = re.findall(r'\b(OPERATOR\d+|operator\d+|Operator\d+)\b', template, re.IGNORECASE)
                            remaining_fields = re.findall(r'\b(DATA_FIELD\d+|data_field\d+|Data_Field\d+)\b', template, re.IGNORECASE)
                            
                            if remaining_ops or remaining_fields:
                                slot = self.slot_manager.get_slot_status(slot_id)
                                if slot:
                                    slot.add_log(f"⚠️ Some placeholders not replaced! Remaining: {remaining_ops + remaining_fields}")
                                    slot.add_log("🔄 Retrying placeholder replacement...")
                                # Retry once more
                                # Get backtest_storage for field usage tracking
                                backtest_storage = None
                                if hasattr(self.generator, 'backtest_storage'):
                                    backtest_storage = self.generator.backtest_storage
                                
                                replaced_retry = self.generator.template_generator.ollama_manager.replace_placeholders_with_selection(
                                    template,
                                    available_operators,
                                    available_fields,
                                    progress_callback=progress_callback,
                                    region=region,
                                    backtest_storage=backtest_storage
                                )
                                if replaced_retry:
                                    template = replaced_retry
                                    # Check again
                                    remaining_ops = re.findall(r'\b(OPERATOR\d+|operator\d+|Operator\d+)\b', template, re.IGNORECASE)
                                    remaining_fields = re.findall(r'\b(DATA_FIELD\d+|data_field\d+|Data_Field\d+)\b', template, re.IGNORECASE)
                                    if remaining_ops or remaining_fields:
                                        slot = self.slot_manager.get_slot_status(slot_id)
                                        if slot:
                                            slot.add_log(f"❌ FAILED: Still has placeholders after retry: {remaining_ops + remaining_fields}")
                                            slot.add_log(f"❌ Skipping submission - template: {template[:100]}...")
                                        self.slot_manager.release_slot(slot_id, success=False, error=f"Placeholders not replaced: {remaining_ops + remaining_fields}")
                                        return
                                    else:
                                        slot = self.slot_manager.get_slot_status(slot_id)
                                        if slot:
                                            slot.add_log("✅ All placeholders replaced after retry")
                                else:
                                    slot = self.slot_manager.get_slot_status(slot_id)
                                    if slot:
                                        slot.add_log("❌ Retry replacement failed, skipping submission")
                                    self.slot_manager.release_slot(slot_id, success=False, error="Placeholder replacement failed")
                                    return
                            else:
                                slot = self.slot_manager.get_slot_status(slot_id)
                                if slot:
                                    slot.add_log("✅ Ollama selection completed, all placeholders replaced")
                        else:
                            slot = self.slot_manager.get_slot_status(slot_id)
                            if slot:
                                slot.add_log("⚠️ Ollama selection failed, using fallback replacement")
                            # Fallback to old method
                            if has_operator_placeholders:
                                template = self.generator.template_generator._replace_operator_placeholders(
                                    template, available_operators
                                )
                            if has_field_placeholders:
                                template = self.generator.template_generator._replace_field_placeholders(
                                    template, available_fields, region
                                )
                            # Verify fallback worked
                            import re
                            remaining_ops = re.findall(r'\b(OPERATOR\d+|operator\d+|Operator\d+)\b', template, re.IGNORECASE)
                            remaining_fields = re.findall(r'\b(DATA_FIELD\d+|data_field\d+|Data_Field\d+)\b', template, re.IGNORECASE)
                            if remaining_ops or remaining_fields:
                                slot = self.slot_manager.get_slot_status(slot_id)
                                if slot:
                                    slot.add_log(f"❌ FAILED: Fallback replacement incomplete. Remaining: {remaining_ops + remaining_fields}")
                                self.slot_manager.release_slot(slot_id, success=False, error=f"Placeholders not replaced: {remaining_ops + remaining_fields}")
                                return
                    else:
                        # Fallback to old method
                        if has_operator_placeholders:
                            template = self.generator.template_generator._replace_operator_placeholders(
                                template, available_operators
                            )
                        if has_field_placeholders:
                            template = self.generator.template_generator._replace_field_placeholders(
                                template, available_fields, region
                            )
                        # Verify fallback worked
                        import re
                        remaining_ops = re.findall(r'\b(OPERATOR\d+|operator\d+|Operator\d+)\b', template, re.IGNORECASE)
                        remaining_fields = re.findall(r'\b(DATA_FIELD\d+|data_field\d+|Data_Field\d+)\b', template, re.IGNORECASE)
                        if remaining_ops or remaining_fields:
                            slot = self.slot_manager.get_slot_status(slot_id)
                            if slot:
                                slot.add_log(f"❌ FAILED: Fallback replacement incomplete. Remaining: {remaining_ops + remaining_fields}")
                            self.slot_manager.release_slot(slot_id, success=False, error=f"Placeholders not replaced: {remaining_ops + remaining_fields}")
                            return
            
            settings = SimulationSettings(
                universe=REGION_DEFAULT_UNIVERSE.get(region, 'TOP3000'),
                neutralization=REGION_DEFAULT_NEUTRALIZATION.get(region, 'INDUSTRY'),
                delay=1,
                testPeriod="P5Y0M0D"
            )
            
            # Final check: Ensure NO placeholders remain before submission
            import re
            remaining_ops = re.findall(r'\b(OPERATOR\d+|operator\d+|Operator\d+)\b', template, re.IGNORECASE)
            remaining_fields = re.findall(r'\b(DATA_FIELD\d+|data_field\d+|Data_Field\d+)\b', template, re.IGNORECASE)
            
            if remaining_ops or remaining_fields:
                slot = self.slot_manager.get_slot_status(slot_id)
                if slot:
                    slot.add_log(f"❌ CRITICAL: Template still has placeholders before submission!")
                    slot.add_log(f"   Remaining operators: {remaining_ops}")
                    slot.add_log(f"   Remaining fields: {remaining_fields}")
                    slot.add_log(f"   Template: {template[:100]}...")
                self.slot_manager.release_slot(slot_id, success=False, error=f"Cannot submit: placeholders remain ({remaining_ops + remaining_fields})")
                slot = self.slot_manager.get_slot_status(slot_id)
                if slot:
                    slot.add_log("❌ Skipping submission - template has unreplaced placeholders")
                return

            normalized_template = self._normalize_before_submit(template)
            if normalized_template != template:
                slot = self.slot_manager.get_slot_status(slot_id)
                if slot:
                    slot.add_log(f"🔧 Normalized expression before submit: {normalized_template[:80]}...")
                template = normalized_template
            
            # Update slot
            slot = self.slot_manager.get_slot_status(slot_id)
            if slot:
                slot.add_log(f"[{region}] Submitting...")

            local_validation = self._validate_before_submit(template, region)
            if local_validation and not local_validation.is_valid:
                error_msg = local_validation.summary()
                slot = self.slot_manager.get_slot_status(slot_id)
                if slot:
                    slot.add_log(f"❌ Local validation failed: {error_msg}")
                    slot.add_log("❌ Skipping API submission to avoid known invalid expression")
                self.slot_manager.release_slot(slot_id, success=False, error=f"Local validation failed: {error_msg}")
                self._update_slot(slot_id, template, region, 0, "Local validation failed", "FAILED")
                self._log(f"❌ {region}: Local validation failed: {error_msg}")
                if self.generator.template_generator.template_validator:
                    self.generator.template_generator.template_validator.learn_from_simulation_error(
                        template, error_msg, None
                    )
                return

            self.slot_manager.update_slot_progress(slot_id, percent=10, message="Submitting...", api_status="PENDING")
            self._update_slot(slot_id, template, region, 10, "Submitting...")

            # Submit
            submission = self.simulator_tester.submit_simulation(template, region, settings)
            if not submission:
                submit_error = submission.error_message or "Failed to submit"
                self.slot_manager.release_slot(slot_id, success=False, error=submit_error)
                self._update_slot(slot_id, template, region, 0, submit_error, "FAILED")
                self._log(f"❌ {region}: {submit_error[:100]}")
                return
            progress_url = submission.progress_url
            
            # Monitor
            def progress_callback(percent, message, api_status):
                self.slot_manager.update_slot_progress(slot_id, percent=percent, message=message, api_status=api_status)
                slot = self.slot_manager.get_slot_status(slot_id)
                if slot:
                    slot.add_log(f"[{api_status}] {message}")
                self._update_slot(slot_id, template, region, percent, message)
            
            try:
                result = self.simulator_tester.monitor_simulation(
                    progress_url, template, region, settings,
                    progress_callback=progress_callback
                )
            finally:
                self.simulator_tester.release_simulation_slot(submission)
            
            # Handle refeed if failed
            if result and not result.success:
                original_result = result
                if self._should_refeed_error(result.error_message):
                    refed_result = self._handle_refeed(slot_id, template, region, result.error_message, settings)
                    if refed_result is not None:
                        result = refed_result
                    else:
                        result = original_result
                else:
                    slot = self.slot_manager.get_slot_status(slot_id)
                    if slot:
                        slot.add_log("⏭ Skipping refeed for IS performance/check failure")
                    logger.info(
                        "Skipping refeed for non-structural failure in %s: %s",
                        region,
                        (result.error_message or "Unknown error")[:160],
                    )
            
            # Check if result is None (simulation failed completely)
            if result is None:
                self.slot_manager.release_slot(slot_id, success=False, error="Simulation failed")
                self._update_slot(slot_id, template, region, 0, "Simulation failed", "FAILED")
                self._log(f"❌ {region}: Simulation failed (no result)")
                return
            
            # Save result
            if self.backtest_storage:
                self.backtest_storage.store_result(result)

            improved_result = self._run_improvement_attempt(
                slot_id=slot_id,
                source_result=result,
                region=region,
                settings=settings
            )
            robustness_base_result = improved_result if improved_result and (
                improved_result.success
                or self._safe_float(getattr(improved_result, 'sharpe', 0.0)) >= self._safe_float(getattr(result, 'sharpe', 0.0))
                or self._safe_float(getattr(improved_result, 'fitness', 0.0)) >= self._safe_float(getattr(result, 'fitness', 0.0))
            ) else result

            self._run_robustness_sweep(
                slot_id=slot_id,
                template=robustness_base_result.template if getattr(robustness_base_result, 'template', None) else template,
                region=region,
                base_settings=settings,
                base_result=robustness_base_result
            )
            
            # Update correlation tracker
            if result.success and result.alpha_id:
                self.correlation_tracker.update_template_alpha_mapping(template, str(result.alpha_id))
                self.search_strategy.add_successful_template(template, region)
            
            # Release slot
            self.slot_manager.release_slot(
                slot_id,
                success=result.success,
                result={
                    'sharpe': result.sharpe,
                    'fitness': result.fitness,
                    'alpha_id': str(result.alpha_id) if result.alpha_id else ""
                } if result.success else None,
                error=result.error_message if not result.success else None
            )
            
            # Update display
            status = "SUCCESS" if result.success else "FAILED"
            message = f"Sharpe: {self._format_metric(result.sharpe)}" if result.success else (result.error_message[:30] if result.error_message else "Failed")
            self._update_slot(slot_id, template, region, 100 if result.success else 0, message, status)
            
            # Log
            if result.success:
                self.simulations_completed_count += 1
                sharpe = self._format_metric(result.sharpe)
                fitness = self._format_metric(result.fitness)
                self._log(f"✅ {region}: Sharpe={sharpe}, Fitness={fitness} (Total: {self.simulations_completed_count})")
            else:
                self._log(f"❌ {region}: {result.error_message[:50] if result.error_message else 'Unknown error'}")
                
        except Exception as e:
            logger.error(f"Simulation error: {e}", exc_info=True)
            self.slot_manager.release_slot(slot_id, success=False, error=str(e))
            self._update_slot(slot_id, template, region, 0, str(e)[:30], "FAILED")
    
    def _handle_refeed(self, slot_id: int, template: str, region: str, error_message: str, settings: SimulationSettings):
        """Handle refeed correction"""
        if not self.generator.template_generator.template_validator:
            return None

        slot = self.slot_manager.get_slot_status(slot_id)
        if slot:
            slot.add_log("🔄 Attempting refeed correction...")
        self._update_slot(slot_id, template, region, 50, "Fixing template...")

        # Check if template still has placeholders - replace them first before refeed
        import re
        has_operator_placeholders = template and ('OPERATOR' in template.upper() or 'operator' in template.lower())
        has_field_placeholders = template and ('DATA_FIELD' in template.upper() or 'data_field' in template.lower())

        if (has_operator_placeholders or has_field_placeholders) and self.generator and self.generator.template_generator:
            slot = self.slot_manager.get_slot_status(slot_id)
            if slot:
                slot.add_log("⚠️ Template still has placeholders, replacing before refeed...")
            available_operators = None
            if hasattr(self.generator.template_generator, 'operator_fetcher'):
                available_operators = self.generator.template_generator.operator_fetcher.operators if self.generator.template_generator.operator_fetcher else None

            available_fields = self.generator.template_generator.get_data_fields_for_region(region)

            if available_operators and available_fields and hasattr(self.generator.template_generator, 'ollama_manager'):
                def progress_callback_refeed(msg):
                    slot = self.slot_manager.get_slot_status(slot_id)
                    if slot:
                        slot.add_log(f"🤖 {msg}")

                # Get backtest_storage for field usage tracking
                backtest_storage = None
                if hasattr(self.generator, 'backtest_storage'):
                    backtest_storage = self.generator.backtest_storage
                
                replaced = self.generator.template_generator.ollama_manager.replace_placeholders_with_selection(
                    template,
                    available_operators,
                    available_fields,
                    progress_callback=progress_callback_refeed,
                    region=region,
                    backtest_storage=backtest_storage
                )
                if replaced:
                    template = replaced
                    slot = self.slot_manager.get_slot_status(slot_id)
                    if slot:
                        slot.add_log("✅ Placeholders replaced before refeed")
                else:
                    slot = self.slot_manager.get_slot_status(slot_id)
                    if slot:
                        slot.add_log("⚠️ Placeholder replacement failed, using fallback")
                    # Fallback to old method
                    if has_operator_placeholders:
                        template = self.generator.template_generator._replace_operator_placeholders(
                            template, available_operators
                        )
                    if has_field_placeholders:
                        template = self.generator.template_generator._replace_field_placeholders(
                            template, available_fields, region
                        )

        # Check if event input error (unlimited retries)
        is_event_input_error = 'does not support event inputs' in error_message.lower()
        max_attempts = 999 if is_event_input_error else 3

        fixed_template, fixes = self.generator.template_generator.template_validator.refeed_with_correction(
            template, error_message, region, max_attempts=max_attempts
        )

        if fixed_template and fixed_template != template:
            # Check if fixed template has placeholders - replace them
            has_op_ph = fixed_template and ('OPERATOR' in fixed_template.upper() or 'operator' in fixed_template.lower())
            has_field_ph = fixed_template and ('DATA_FIELD' in fixed_template.upper() or 'data_field' in fixed_template.lower())

            if (has_op_ph or has_field_ph) and self.generator and self.generator.template_generator:
                available_operators = None
                if hasattr(self.generator.template_generator, 'operator_fetcher'):
                    available_operators = self.generator.template_generator.operator_fetcher.operators if self.generator.template_generator.operator_fetcher else None

                available_fields = self.generator.template_generator.get_data_fields_for_region(region)

                if available_operators and available_fields and hasattr(self.generator.template_generator, 'ollama_manager'):
                    # Get backtest_storage for field usage tracking
                    backtest_storage = None
                    if hasattr(self.generator, 'backtest_storage'):
                        backtest_storage = self.generator.backtest_storage
                    
                    replaced_again = self.generator.template_generator.ollama_manager.replace_placeholders_with_selection(
                        fixed_template,
                        available_operators,
                        available_fields,
                        region=region,
                        backtest_storage=backtest_storage
                    )
                    if replaced_again:
                        fixed_template = replaced_again
                    else:
                        # Fallback
                        if has_op_ph and available_operators:
                            fixed_template = self.generator.template_generator._replace_operator_placeholders(
                                fixed_template, available_operators
                            )
                        if has_field_ph and available_fields:
                            fixed_template = self.generator.template_generator._replace_field_placeholders(
                                fixed_template, available_fields, region
                            )
            
            # Final check: Ensure NO placeholders remain before resubmission
            remaining_ops = re.findall(r'\b(OPERATOR\d+|operator\d+|Operator\d+)\b', fixed_template, re.IGNORECASE)
            remaining_fields = re.findall(r'\b(DATA_FIELD\d+|data_field\d+|Data_Field\d+)\b', fixed_template, re.IGNORECASE)
            
            if remaining_ops or remaining_fields:
                slot = self.slot_manager.get_slot_status(slot_id)
                if slot:
                    slot.add_log(f"❌ CRITICAL: Fixed template still has placeholders! Remaining: {remaining_ops + remaining_fields}")
                return None  # Cannot proceed with placeholders
            
            # Retry with fixed template
            slot = self.slot_manager.get_slot_status(slot_id)
            if slot:
                slot.add_log(f"✅ Fixed with {len(fixes)} corrections, retrying...")

            normalized_template = self._normalize_before_submit(fixed_template)
            if normalized_template != fixed_template:
                slot = self.slot_manager.get_slot_status(slot_id)
                if slot:
                    slot.add_log(f"🔧 Normalized retry expression: {normalized_template[:80]}...")
                fixed_template = normalized_template

            local_validation = self._validate_before_submit(fixed_template, region)
            if local_validation and not local_validation.is_valid:
                error_msg = local_validation.summary()
                slot = self.slot_manager.get_slot_status(slot_id)
                if slot:
                    slot.add_log(f"❌ Local validation failed before retry: {error_msg}")
                    slot.add_log("❌ Skipping API resubmission")
                if self.generator.template_generator.template_validator:
                    self.generator.template_generator.template_validator.learn_from_simulation_error(
                        fixed_template, error_msg, None
                    )
                return None

            submission = self.simulator_tester.submit_simulation(fixed_template, region, settings)
            
            if submission:
                progress_url = submission.progress_url
                def progress_callback(percent, message, api_status):
                    self.slot_manager.update_slot_progress(slot_id, percent=percent, message=message, api_status=api_status)
                    slot = self.slot_manager.get_slot_status(slot_id)
                    if slot:
                        slot.add_log(f"[{api_status}] {message}")
                    self._update_slot(slot_id, fixed_template, region, percent, message)
                
                try:
                    result = self.simulator_tester.monitor_simulation(
                        progress_url, fixed_template, region, settings,
                        progress_callback=progress_callback
                    )
                finally:
                    self.simulator_tester.release_simulation_slot(submission)
                return result
            else:
                submit_error = submission.error_message or "Failed to resubmit fixed template"
                slot = self.slot_manager.get_slot_status(slot_id)
                if slot:
                    slot.add_log(f"❌ Failed to resubmit fixed template: {submit_error}")
        
        return None
    
    def _update_slot(self, slot_id: int, template: str, region: str, progress: float, message: str, status: str = "RUNNING"):
        """Update slot display"""
        if self.update_slot_callback:
            slot = self.slot_manager.get_slot_status(slot_id)
            logs = slot.get_logs()[-5:] if slot else []
            self.update_slot_callback(
                slot_id, status, template[:40] + "..." if len(template) > 40 else template,
                f"Region: {region}", progress, message, logs
            )
    
    def _periodic_cleanup(self):
        """Periodic cleanup for long-term operation"""
        try:
            # Log operation statistics
            uptime_hours = (time.time() - self.operation_start_time) / 3600
            self._log(f"🧹 Cleanup: Uptime={uptime_hours:.1f}h, Generated={self.templates_generated_count}, Completed={self.simulations_completed_count}")
            
            # Trim queue if too large
            if len(self.generated_templates_queue) > self.max_queue_size:
                old_size = len(self.generated_templates_queue)
                self.generated_templates_queue = self.generated_templates_queue[-self.max_queue_size//2:]
                self._log(f"🧹 Trimmed queue from {old_size} to {len(self.generated_templates_queue)} templates")
            
            # Force garbage collection periodically
            import gc
            gc.collect()
            
        except Exception as e:
            logger.debug(f"Cleanup error: {e}")
    
    def _log(self, message: str):
        """Log message"""
        logger.info(message)
        if self.log_callback:
            self.log_callback(message)
