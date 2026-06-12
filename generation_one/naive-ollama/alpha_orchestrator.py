import argparse
import requests
import json
import os
import time
import logging
import schedule
from datetime import datetime, timedelta
from typing import List, Dict
from requests.auth import HTTPBasicAuth
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue
from dataclasses import dataclass
from alpha_generator_ollama import _parse_credentials_file

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('alpha_orchestrator.log')
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ModelInfo:
    """Information about a model in the fleet."""
    name: str
    size_mb: int
    priority: int  # Lower number = higher priority (used first)
    description: str

class ModelFleetManager:
    """Manages a fleet of models with automatic downgrading on VRAM issues."""
    
    def __init__(self, ollama_url: str = "http://localhost:11434"):
        self.ollama_url = ollama_url
        self.current_model_index = 0
        self.vram_error_count = 0
        self.max_vram_errors = 3  # Number of VRAM errors before downgrading
        
        # Model fleet ordered by priority (largest to smallest)
        # Optimized for RTX A4000 (16GB VRAM) with DeepSeek-R1 reasoning models
        self.model_fleet = [
            ModelInfo("deepseek-r1:8b", 5200, 1, "DeepSeek-R1 8B - Reasoning model (RTX A4000 optimized)"),
            ModelInfo("deepseek-r1:7b", 4700, 2, "DeepSeek-R1 7B - Reasoning model"),
            ModelInfo("deepseek-r1:1.5b", 1100, 3, "DeepSeek-R1 1.5B - Reasoning model"),
            ModelInfo("llama3:3b", 2048, 4, "Llama 3.2 3B - Fallback model"),
            ModelInfo("phi3:mini", 2200, 5, "Phi3 mini - Emergency fallback"),
        ]
        
        # State file to persist current model selection
        self.state_file = "model_fleet_state.json"
        self.load_state()
        
    def load_state(self):
        """Load the current model state from file."""
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                    self.current_model_index = state.get('current_model_index', 0)
                    self.vram_error_count = state.get('vram_error_count', 0)
                    logger.info(f"Loaded state: model_index={self.current_model_index}, vram_errors={self.vram_error_count}")
        except Exception as e:
            logger.warning(f"Could not load state: {e}")
            self.current_model_index = 0
            self.vram_error_count = 0
    
    def save_state(self):
        """Save the current model state to file."""
        try:
            state = {
                'current_model_index': self.current_model_index,
                'vram_error_count': self.vram_error_count,
                'current_model': self.get_current_model().name,
                'timestamp': time.time()
            }
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2)
            logger.info(f"Saved state: {state}")
        except Exception as e:
            logger.error(f"Could not save state: {e}")
    
    def get_current_model(self) -> ModelInfo:
        """Get the current model in use."""
        if self.current_model_index >= len(self.model_fleet):
            self.current_model_index = len(self.model_fleet) - 1
        return self.model_fleet[self.current_model_index]
    
    def get_available_models(self) -> List[str]:
        """Get list of available models via Ollama API."""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags")
            if response.status_code == 200:
                models_data = response.json()
                return [model['name'] for model in models_data.get('models', [])]
            else:
                logger.error(f"Failed to get available models: {response.status_code}")
                return []
        except Exception as e:
            logger.error(f"Error getting available models: {e}")
            return []
    
    def ensure_model_available(self, model_name: str) -> bool:
        """Ensure a specific model is available, download if needed."""
        available_models = self.get_available_models()
        
        if model_name in available_models:
            logger.info(f"Model {model_name} is already available")
            return True
        
        logger.info(f"Model {model_name} not found, downloading...")
        try:
            response = requests.post(f"{self.ollama_url}/api/pull", json={'name': model_name})
            if response.status_code == 200:
                logger.info(f"Successfully downloaded model {model_name}")
                return True
            else:
                logger.error(f"Failed to download model {model_name}: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"Error downloading model {model_name}: {e}")
            return False
    
    def detect_vram_error(self, log_line: str) -> bool:
        """Detect VRAM recovery timeout errors in log lines."""
        vram_error_indicators = [
            "gpu VRAM usage didn't recover within timeout",
            "VRAM usage didn't recover",
            "gpu memory exhausted",
            "CUDA out of memory",
            "GPU memory allocation failed",
            "msg=\"gpu VRAM usage didn't recover within timeout\"",
            "level=WARN source=sched.go"
        ]
        
        return any(indicator.lower() in log_line.lower() for indicator in vram_error_indicators)
    
    def handle_vram_error(self) -> bool:
        """Handle VRAM error by downgrading to a smaller model."""
        self.vram_error_count += 1
        logger.warning(f"VRAM error detected! Count: {self.vram_error_count}/{self.max_vram_errors}")
        
        if self.vram_error_count >= self.max_vram_errors:
            return self.downgrade_model()
        
        self.save_state()
        return False
    
    def downgrade_model(self) -> bool:
        """Downgrade to the next smaller model in the fleet."""
        if self.current_model_index >= len(self.model_fleet) - 1:
            logger.error("Already using the smallest model in the fleet!")
            logger.warning("VRAM error persists with smallest model - triggering application reset")
            return self.trigger_application_reset()
        
        old_model = self.get_current_model()
        self.current_model_index += 1
        new_model = self.get_current_model()
        
        logger.warning(f"Downgrading model: {old_model.name} -> {new_model.name}")
        
        # Ensure the new model is available
        if not self.ensure_model_available(new_model.name):
            logger.error(f"Failed to ensure model {new_model.name} is available")
            self.current_model_index -= 1  # Revert
            return False
        
        # Reset VRAM error count
        self.vram_error_count = 0
        
        # Save state
        self.save_state()
        
        # Update the alpha generator configuration
        self.update_alpha_generator_config(new_model.name)
        
        logger.info(f"Successfully downgraded to {new_model.name}")
        return True
    
    def trigger_application_reset(self) -> bool:
        """Trigger a complete application reset when VRAM issues persist with smallest model."""
        try:
            logger.warning("Triggering application reset due to persistent VRAM issues")
            
            # Reset to the largest model
            self.current_model_index = 0
            self.vram_error_count = 0
            self.save_state()
            
            # Update configuration to use largest model
            self.update_alpha_generator_config(self.get_current_model().name)
            
            logger.info("Application reset completed - returning to largest model")
            return True
            
        except Exception as e:
            logger.error(f"Error during application reset: {e}")
            return False
    
    def update_alpha_generator_config(self, model_name: str):
        """Update the alpha generator configuration to use the new model."""
        try:
            # Update the default model in alpha_generator_ollama.py
            with open('alpha_generator_ollama.py', 'r') as f:
                content = f.read()
            
            # Replace the default model
            content = content.replace(
                    "default='llama3.2:8b'",
                f"default='{model_name}'"
            )
            content = content.replace(
                "getattr(self, 'model_name', 'llama3.2:8b')",
                f"getattr(self, 'model_name', '{model_name}')"
            )
            
            with open('alpha_generator_ollama.py', 'w') as f:
                f.write(content)
            
            logger.info(f"Updated alpha generator config to use {model_name}")
        except Exception as e:
            logger.error(f"Error updating alpha generator config: {e}")
    
    def get_fleet_status(self) -> Dict:
        """Get the current status of the model fleet."""
        current_model = self.get_current_model()
        available_models = self.get_available_models()
        
        return {
            'current_model': {
                'name': current_model.name,
                'size_mb': current_model.size_mb,
                'description': current_model.description,
                'index': self.current_model_index
            },
            'vram_error_count': self.vram_error_count,
            'max_vram_errors': self.max_vram_errors,
            'available_models': available_models,
            'fleet_size': len(self.model_fleet),
            'can_downgrade': self.current_model_index < len(self.model_fleet) - 1
        }
    
    def reset_to_largest_model(self):
        """Reset to the largest model in the fleet."""
        self.current_model_index = 0
        self.vram_error_count = 0
        self.save_state()
        logger.info("Reset to largest model in fleet")
        return self.update_alpha_generator_config(self.get_current_model().name)

class AlphaOrchestrator:
    def __init__(self, credentials_path: str, ollama_url: str = "http://localhost:11434"):
        self.sess = requests.Session()
        self.credentials_path = credentials_path
        self.ollama_url = ollama_url
        self.setup_auth(credentials_path)
        self.last_submission_date = None
        self.submission_log_file = "submission_log.json"
        self.load_submission_history()
        
        # Concurrency control
        self.max_concurrent_simulations = 3
        self.simulation_semaphore = threading.Semaphore(self.max_concurrent_simulations)
        self.running = True
        self.generator_process = None
        self.miner_process = None
        
        # Model fleet management
        self.model_fleet_manager = ModelFleetManager(ollama_url)
        self.vram_monitoring_active = False
        self.vram_monitor_thread = None
        
        # Restart mechanism
        self.restart_interval = 1800  # 30 minutes in seconds
        self.last_restart_time = time.time()
        self.restart_thread = None
        
    def setup_auth(self, credentials_path: str) -> None:
        """Set up authentication with WorldQuant Brain."""
        logger.info(f"Loading credentials from {credentials_path}")
        credentials, _, resolved_path = _parse_credentials_file(credentials_path)
        
        username, password = credentials
        self.sess.auth = HTTPBasicAuth(username, password)
        self.credentials_path = resolved_path
        
        logger.info("Authenticating with WorldQuant Brain...")
        response = self.sess.post('https://api.worldquantbrain.com/authentication')
        logger.info(f"Authentication response status: {response.status_code}")
        
        if response.status_code != 201:
            raise Exception(f"Authentication failed: {response.text}")
        logger.info("Authentication successful")

    def load_submission_history(self):
        """Load submission history to track daily submissions."""
        if os.path.exists(self.submission_log_file):
            try:
                with open(self.submission_log_file, 'r') as f:
                    data = json.load(f)
                    self.last_submission_date = data.get('last_submission_date')
                    logger.info(f"Loaded submission history. Last submission: {self.last_submission_date}")
            except Exception as e:
                logger.warning(f"Could not load submission history: {e}")
                self.last_submission_date = None
        else:
            self.last_submission_date = None

    def save_submission_history(self):
        """Save submission history."""
        data = {
            'last_submission_date': self.last_submission_date,
            'updated_at': datetime.now().isoformat()
        }
        with open(self.submission_log_file, 'w') as f:
            json.dump(data, f, indent=2)

    def start_vram_monitoring(self):
        """Start VRAM monitoring in a separate thread."""
        if self.vram_monitoring_active:
            logger.info("VRAM monitoring already active")
            return
        
        self.vram_monitoring_active = True
        self.vram_monitor_thread = threading.Thread(target=self._vram_monitor_loop, daemon=True)
        self.vram_monitor_thread.start()
        logger.info("Started VRAM monitoring thread")

    def stop_vram_monitoring(self):
        """Stop VRAM monitoring."""
        self.vram_monitoring_active = False
        if self.vram_monitor_thread:
            self.vram_monitor_thread.join(timeout=5)
        logger.info("Stopped VRAM monitoring")

    def _vram_monitor_loop(self):
        """VRAM monitoring loop that checks for errors and handles model downgrading."""
        logger.info("VRAM monitoring loop started")
        
        while self.vram_monitoring_active and self.running:
            try:
                # Check for VRAM errors in recent logs
                if self._check_for_vram_errors():
                    logger.warning("VRAM error detected in monitoring loop")
                    if self.model_fleet_manager.handle_vram_error():
                        logger.info("Model fleet action taken due to VRAM issues")
                        # Restart the alpha generator with new model configuration
                        self._restart_alpha_generator()
                
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in VRAM monitoring loop: {e}")
                time.sleep(60)  # Wait longer on error
        
        logger.info("VRAM monitoring loop stopped")

    def _check_for_vram_errors(self) -> bool:
        """Check recent logs for VRAM errors."""
        try:
            # Check Ollama logs and application logs for VRAM errors
            log_files_to_check = [
                '/app/logs/ollama.log',  # Ollama logs redirected to file
                'alpha_orchestrator.log',
                'alpha_generator_ollama.log'
            ]
            
            for log_file in log_files_to_check:
                try:
                    if os.path.exists(log_file):
                        # Read last 50 lines of log file
                        with open(log_file, 'r') as f:
                            lines = f.readlines()
                            recent_lines = lines[-50:] if len(lines) > 50 else lines
                            
                            for line in recent_lines:
                                if self.model_fleet_manager.detect_vram_error(line):
                                    logger.warning(f"VRAM error found in {log_file}: {line.strip()}")
                                    return True
                except Exception as e:
                    # Skip files that can't be read
                    continue
            
            return False
        except Exception as e:
            logger.error(f"Error checking for VRAM errors: {e}")
            return False

    def _restart_alpha_generator(self):
        """Restart the alpha generator with the new model."""
        try:
            logger.info("Restarting alpha generator with new model")
            
            # Stop current generator process if running
            if self.generator_process and self.generator_process.poll() is None:
                self.generator_process.terminate()
                self.generator_process.wait(timeout=30)
            
            # Start new generator process in continuous mode
            self.start_alpha_generator_continuous(batch_size=3, sleep_time=30)
            
        except Exception as e:
            logger.error(f"Error restarting alpha generator: {e}")
    
    def start_restart_monitoring(self):
        """Start restart monitoring in a separate thread."""
        if not self.restart_thread or not self.restart_thread.is_alive():
            self.restart_thread = threading.Thread(target=self._restart_monitor_loop, daemon=True)
            self.restart_thread.start()
            logger.info("🔄 Restart monitoring started (30-minute intervals)")
    
    def _restart_monitor_loop(self):
        """Monitor and restart processes every 30 minutes."""
        while self.running:
            try:
                current_time = time.time()
                time_since_last_restart = current_time - self.last_restart_time
                
                if time_since_last_restart >= self.restart_interval:
                    logger.info(f"⏰ 30 minutes elapsed since last restart, initiating restart...")
                    self.restart_all_processes()
                else:
                    remaining_time = self.restart_interval - time_since_last_restart
                    logger.debug(f"⏰ Next restart in {remaining_time/60:.1f} minutes")
                
                # Check every minute
                time.sleep(60)
                
            except Exception as e:
                logger.error(f"Error in restart monitoring: {e}")
                time.sleep(60)

    def get_model_fleet_status(self) -> Dict:
        """Get the current status of the model fleet."""
        return self.model_fleet_manager.get_fleet_status()

    def reset_model_fleet(self):
        """Reset the model fleet to the largest model."""
        return self.model_fleet_manager.reset_to_largest_model()

    def force_model_downgrade(self):
        """Force downgrade to the next smaller model."""
        return self.model_fleet_manager.downgrade_model()
    
    def force_application_reset(self):
        """Force a complete application reset."""
        logger.warning("Forcing application reset")
        return self.model_fleet_manager.trigger_application_reset()

    def can_submit_today(self) -> bool:
        """Check if we can submit alphas today (only once per day)."""
        today = datetime.now().date().isoformat()
        
        if self.last_submission_date == today:
            logger.info(f"Already submitted today ({today}). Skipping submission.")
            return False
        
        logger.info(f"Can submit today. Last submission was: {self.last_submission_date}")
        return True

    def run_alpha_expression_miner(self, promising_alpha_file: str = "hopeful_alphas.json"):
        """Run alpha expression miner on promising alphas."""
        logger.info("Starting alpha expression miner on promising alphas...")
        
        if not os.path.exists(promising_alpha_file):
            logger.warning(f"Promising alphas file {promising_alpha_file} not found. Skipping mining.")
            return
        
        try:
            with open(promising_alpha_file, 'r') as f:
                promising_alphas = json.load(f)
            
            if not promising_alphas:
                logger.info("No promising alphas found. Skipping mining.")
                return
            
            logger.info(f"Found {len(promising_alphas)} promising alphas to mine")
            
            # Run alpha expression miner for each promising alpha
            # Note: The miner will automatically remove successfully mined alphas from hopeful_alphas.json
            for i, alpha_data in enumerate(promising_alphas, 1):
                expression = alpha_data.get('expression', '')
                if not expression:
                    continue
                
                logger.info(f"Mining alpha {i}/{len(promising_alphas)}: {expression[:100]}...")
                
                # Run the alpha expression miner as a subprocess
                try:
                    result = subprocess.run([
                        sys.executable, 'alpha_expression_miner.py',
                        '--expression', expression,
                        '--auto-mode',  # Run in automated mode
                        '--output-file', f'mining_results_{i}.json'
                    ], capture_output=True, text=True, timeout=300)
                    
                    if result.returncode == 0:
                        logger.info(f"Successfully mined alpha {i}")
                        # The alpha will be automatically removed from hopeful_alphas.json by the miner
                    else:
                        logger.error(f"Failed to mine alpha {i}: {result.stderr}")
                        # Failed alphas remain in hopeful_alphas.json for retry
                        
                except subprocess.TimeoutExpired:
                    logger.error(f"Mining alpha {i} timed out")
                except Exception as e:
                    logger.error(f"Error mining alpha {i}: {e}")
                
                # Small delay between mining operations
                time.sleep(5)
                
        except Exception as e:
            logger.error(f"Error running alpha expression miner: {e}")

    def run_alpha_submitter(self, batch_size: int = 5):
        """Run alpha submitter with daily rate limiting."""
        logger.info("Starting alpha submitter...")
        
        if not self.can_submit_today():
            return
        
        try:
            # Run the alpha submitter as a subprocess
            result = subprocess.run([
                sys.executable, 'successful_alpha_submitter.py',
                '--batch-size', str(batch_size),
                '--auto-mode'  # Run in automated mode
            ], capture_output=True, text=True, timeout=600)
            
            if result.returncode == 0:
                logger.info("Successfully completed alpha submission")
                # Update submission date
                self.last_submission_date = datetime.now().date().isoformat()
                self.save_submission_history()
            else:
                logger.error(f"Alpha submission failed: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            logger.error("Alpha submission timed out")
        except Exception as e:
            logger.error(f"Error running alpha submitter: {e}")

    def run_alpha_generator(self, batch_size: int = 5, sleep_time: int = 30):
        """Run the main alpha generator with Ollama."""
        logger.info("Starting alpha generator with Ollama...")
        
        # Get current model from fleet manager
        current_model = self.model_fleet_manager.get_current_model().name
        logger.info(f"Using model: {current_model}")
        
        try:
            # Run the alpha generator as a subprocess
            result = subprocess.run([
                sys.executable, 'alpha_generator_ollama.py',
                '--credentials', self.credentials_path,
                '--batch-size', str(batch_size),
                '--sleep-time', str(sleep_time),
                '--ollama-url', self.ollama_url,
                '--ollama-model', current_model,
                '--max-concurrent', str(self.max_concurrent_simulations)
            ], capture_output=True, text=True, timeout=3600)  # 1 hour timeout
            
            if result.returncode == 0:
                logger.info("Alpha generator completed successfully")
            else:
                logger.error(f"Alpha generator failed: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            logger.error("Alpha generator timed out")
        except Exception as e:
            logger.error(f"Error running alpha generator: {e}")

    def start_alpha_generator_continuous(self, batch_size: int = 3, sleep_time: int = 30):
        """Start alpha generator in continuous mode as a background process."""
        logger.info("Starting alpha generator in continuous mode...")
        
        # Get current model from fleet manager
        current_model = self.model_fleet_manager.get_current_model().name
        logger.info(f"Using model: {current_model}")
        
        try:
            self.generator_process = subprocess.Popen([
                sys.executable, 'alpha_generator_ollama.py',
                '--credentials', self.credentials_path,
                '--batch-size', str(batch_size),
                '--sleep-time', str(sleep_time),
                '--ollama-url', self.ollama_url,
                '--ollama-model', current_model,
                '--max-concurrent', str(self.max_concurrent_simulations)
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            
            logger.info(f"Alpha generator started with PID: {self.generator_process.pid}")
            
        except Exception as e:
            logger.error(f"Error starting alpha generator: {e}")

    def start_alpha_expression_miner_continuous(self, check_interval: int = 300):
        """Start alpha expression miner in continuous mode."""
        logger.info("Starting alpha expression miner in continuous mode...")
        
        while self.running:
            try:
                # Check if hopeful_alphas.json exists and has content
                if os.path.exists("hopeful_alphas.json"):
                    try:
                        with open("hopeful_alphas.json", 'r') as f:
                            alphas = json.load(f)
                            if alphas and len(alphas) > 0:
                                logger.info(f"Found {len(alphas)} alphas to mine")
                                self.run_alpha_expression_miner()
                            else:
                                logger.info("No alphas found in hopeful_alphas.json")
                    except json.JSONDecodeError:
                        logger.warning("hopeful_alphas.json is not valid JSON, waiting for valid data...")
                    except Exception as e:
                        logger.error(f"Error reading hopeful_alphas.json: {e}")
                else:
                    logger.info("hopeful_alphas.json not found yet, waiting for alpha generator to create promising alphas...")
                
                # Wait before next check
                time.sleep(check_interval)
                
            except Exception as e:
                logger.error(f"Error in continuous miner: {e}")
                time.sleep(check_interval)

    def restart_all_processes(self):
        """Restart all running processes to prevent stuck jobs."""
        logger.info("🔄 Restarting all processes to prevent stuck jobs...")
        
        # Stop current processes
        self.stop_processes()
        
        # Wait a moment for processes to terminate
        time.sleep(5)
        
        # Restart processes
        try:
            # Restart alpha generator
            logger.info("🔄 Restarting alpha generator...")
            self.start_alpha_generator_continuous(batch_size=3, sleep_time=30)
            
            # Restart VRAM monitoring
            logger.info("🔄 Restarting VRAM monitoring...")
            self.start_vram_monitoring()
            
            logger.info("✅ All processes restarted successfully")
            self.last_restart_time = time.time()
            
        except Exception as e:
            logger.error(f"❌ Error during restart: {e}")
    
    def stop_processes(self):
        """Stop all running processes."""
        logger.info("Stopping all processes...")
        self.running = False
        
        # Stop restart thread
        if self.restart_thread and self.restart_thread.is_alive():
            logger.info("Stopping restart monitoring thread...")
        
        # Stop VRAM monitoring
        self.stop_vram_monitoring()
        
        if self.generator_process:
            logger.info("Terminating alpha generator process...")
            self.generator_process.terminate()
            try:
                self.generator_process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                logger.warning("Force killing alpha generator process...")
                self.generator_process.kill()
        
        if self.miner_process:
            logger.info("Terminating alpha miner process...")
            self.miner_process.terminate()
            try:
                self.miner_process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                logger.warning("Force killing alpha miner process...")
                self.miner_process.kill()

    def daily_workflow(self):
        """Run the complete daily workflow."""
        logger.info("Starting daily alpha workflow...")
        
        # 1. Run alpha generator for a few hours
        logger.info("Phase 1: Running alpha generator...")
        self.run_alpha_generator(batch_size=3, sleep_time=60)
        
        # 2. Run alpha expression miner on promising alphas
        logger.info("Phase 2: Running alpha expression miner...")
        self.run_alpha_expression_miner()
        
        # 3. Run alpha submitter (once per day)
        logger.info("Phase 3: Running alpha submitter...")
        self.run_alpha_submitter(batch_size=3)
        
        logger.info("Daily workflow completed")

    def continuous_mining(self, mining_interval_hours: int = 6):
        """Run continuous mining with concurrent alpha generation and expression mining."""
        logger.info(f"Starting continuous mining with {mining_interval_hours}h intervals...")
        
        try:
            # Start VRAM monitoring
            logger.info("Starting VRAM monitoring...")
            self.start_vram_monitoring()
            
            # Start restart monitoring
            logger.info("Starting restart monitoring...")
            self.start_restart_monitoring()
            
            # Start alpha generator in continuous mode
            self.start_alpha_generator_continuous(batch_size=3, sleep_time=30)
            
            # Start alpha expression miner in a separate thread
            miner_thread = threading.Thread(
                target=self.start_alpha_expression_miner_continuous,
                args=(mining_interval_hours * 3600,),  # Convert hours to seconds
                daemon=True
            )
            miner_thread.start()
            
            # Schedule daily submission at 2 PM
            schedule.every().day.at("14:00").do(self.run_alpha_submitter)
            
            logger.info("Both alpha generator and expression miner are running concurrently")
            logger.info(f"Max concurrent simulations: {self.max_concurrent_simulations}")
            
            while self.running:
                try:
                    # Run pending scheduled tasks
                    schedule.run_pending()
                    
                    # Check if generator process is still running
                    if self.generator_process and self.generator_process.poll() is not None:
                        logger.warning("Alpha generator process stopped, restarting...")
                        self.start_alpha_generator_continuous(batch_size=3, sleep_time=30)
                    
                    # Small delay before next cycle
                    time.sleep(60)
                    
                except KeyboardInterrupt:
                    logger.info("Received interrupt signal, stopping...")
                    break
                except Exception as e:
                    logger.error(f"Error in continuous mining: {e}")
                    time.sleep(300)  # Wait 5 minutes before retrying
                    
        finally:
            self.stop_processes()

def main():
    parser = argparse.ArgumentParser(description='Alpha Orchestrator - Manage alpha generation, mining, and submission')
    parser.add_argument('--credentials', type=str, default='./credentials.txt',
                      help='Path to credentials file (default: ./credentials.txt, falls back to credential.txt if present)')
    parser.add_argument('--ollama-url', type=str, default='http://localhost:11434',
                      help='Ollama API URL (default: http://localhost:11434)')
    parser.add_argument('--mode', type=str, choices=['daily', 'continuous', 'miner', 'submitter', 'generator', 'fleet-status', 'fleet-reset', 'fleet-downgrade', 'fleet-reset-app', 'restart'],
                      default='continuous', help='Operation mode (default: continuous)')
    parser.add_argument('--mining-interval', type=int, default=6,
                      help='Mining interval in hours for continuous mode (default: 6)')
    parser.add_argument('--batch-size', type=int, default=3,
                      help='Batch size for operations (default: 3)')
    parser.add_argument('--max-concurrent', type=int, default=3,
                      help='Maximum concurrent simulations (default: 3)')
    parser.add_argument('--restart-interval', type=int, default=30,
                      help='Restart interval in minutes (default: 30)')
    parser.add_argument('--ollama-model', type=str, default='deepseek-r1:8b',
                      help='Ollama model to use (default: deepseek-r1:8b)')
    
    args = parser.parse_args()
    
    try:
        orchestrator = AlphaOrchestrator(args.credentials, args.ollama_url)
        orchestrator.max_concurrent_simulations = args.max_concurrent
        orchestrator.restart_interval = args.restart_interval * 60  # Convert minutes to seconds
        
        # Update the model fleet to use the specified model
        if args.ollama_model:
            # Find the model in the fleet and set it as current
            for i, model_info in enumerate(orchestrator.model_fleet_manager.model_fleet):
                if model_info.name == args.ollama_model:
                    orchestrator.model_fleet_manager.current_model_index = i
                    orchestrator.model_fleet_manager.save_state()
                    logger.info(f"Set model fleet to use: {args.ollama_model}")
                    break
        
        if args.mode == 'daily':
            orchestrator.daily_workflow()
        elif args.mode == 'continuous':
            orchestrator.continuous_mining(args.mining_interval)
        elif args.mode == 'miner':
            orchestrator.run_alpha_expression_miner()
        elif args.mode == 'submitter':
            orchestrator.run_alpha_submitter(args.batch_size)
        elif args.mode == 'generator':
            orchestrator.run_alpha_generator(args.batch_size)
        elif args.mode == 'fleet-status':
            status = orchestrator.get_model_fleet_status()
            print(json.dumps(status, indent=2))
        elif args.mode == 'fleet-reset':
            orchestrator.reset_model_fleet()
            print("Model fleet reset to largest model")
        elif args.mode == 'fleet-downgrade':
            orchestrator.force_model_downgrade()
            print("Model fleet downgraded to next smaller model")
        elif args.mode == 'fleet-reset-app':
            orchestrator.force_application_reset()
            print("Application reset completed - returned to largest model")
        elif args.mode == 'restart':
            orchestrator.restart_all_processes()
            print("Manual restart completed")
            
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
