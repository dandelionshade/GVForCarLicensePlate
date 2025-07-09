# -*- coding: utf-8 -*-
"""
System monitoring utilities
"""

import time
import threading
from typing import Dict, Any, Optional, Callable
from datetime import datetime
import logging


# Fallback implementation if psutil is not available
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logging.warning("psutil not available - system monitoring will be limited")


class SystemMonitor:
    """Monitor system resources and health"""
    
    def __init__(self, check_interval: float = 5.0):
        self.check_interval = check_interval
        self.is_monitoring = False
        self.monitor_thread = None
        self.callbacks = []
        self.logger = logging.getLogger(__name__)
        
        # System metrics
        self.current_metrics = {}
        self.alert_thresholds = {
            'cpu_percent': 80.0,
            'memory_percent': 85.0,
            'disk_percent': 90.0
        }
        
        # Alert tracking
        self.active_alerts = set()
        self.alert_history = []
    
    def add_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Add a callback function to be called with metrics"""
        self.callbacks.append(callback)
    
    def remove_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Remove a callback function"""
        if callback in self.callbacks:
            self.callbacks.remove(callback)
    
    def set_alert_threshold(self, metric: str, threshold: float) -> None:
        """Set alert threshold for a metric"""
        self.alert_thresholds[metric] = threshold
    
    def start_monitoring(self) -> None:
        """Start system monitoring"""
        if self.is_monitoring:
            self.logger.warning("System monitoring is already running")
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        self.logger.info(f"System monitoring started (interval: {self.check_interval}s)")
    
    def stop_monitoring(self) -> None:
        """Stop system monitoring"""
        if not self.is_monitoring:
            return
        
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=10.0)
        self.logger.info("System monitoring stopped")
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current system metrics"""
        if PSUTIL_AVAILABLE:
            return self._collect_system_metrics()
        else:
            return self._collect_basic_metrics()
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status"""
        metrics = self.get_current_metrics()
        
        # Determine overall health
        health_status = "healthy"
        warnings = []
        
        for metric, threshold in self.alert_thresholds.items():
            if metric in metrics and metrics[metric] > threshold:
                health_status = "warning" if health_status == "healthy" else "critical"
                warnings.append(f"{metric}: {metrics[metric]:.1f}% (threshold: {threshold}%)")
        
        return {
            'timestamp': datetime.now().isoformat(),
            'health_status': health_status,
            'metrics': metrics,
            'warnings': warnings,
            'active_alerts': list(self.active_alerts),
            'psutil_available': PSUTIL_AVAILABLE
        }
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                # Collect metrics
                metrics = self.get_current_metrics()
                self.current_metrics = metrics
                
                # Check for alerts
                self._check_alerts(metrics)
                
                # Call callbacks
                for callback in self.callbacks:
                    try:
                        callback(metrics)
                    except Exception as e:
                        self.logger.error(f"Error in metrics callback: {e}")
                
                # Sleep until next check
                time.sleep(self.check_interval)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.check_interval)
    
    def _collect_system_metrics(self) -> Dict[str, Any]:
        """Collect system metrics using psutil"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            
            # Disk usage
            disk = psutil.disk_usage('/')
            
            # Process info
            process = psutil.Process()
            process_memory = process.memory_info()
            
            return {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_total_mb': memory.total / 1024 / 1024,
                'memory_used_mb': memory.used / 1024 / 1024,
                'memory_available_mb': memory.available / 1024 / 1024,
                'disk_percent': disk.percent,
                'disk_total_gb': disk.total / 1024 / 1024 / 1024,
                'disk_used_gb': disk.used / 1024 / 1024 / 1024,
                'disk_free_gb': disk.free / 1024 / 1024 / 1024,
                'process_memory_mb': process_memory.rss / 1024 / 1024,
                'process_memory_percent': process.memory_percent(),
                'timestamp': time.time()
            }
            
        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {e}")
            return self._collect_basic_metrics()
    
    def _collect_basic_metrics(self) -> Dict[str, Any]:
        """Collect basic metrics without psutil"""
        import os
        
        metrics = {
            'timestamp': time.time(),
            'psutil_available': False
        }
        
        # Try to get basic system info
        try:
            # Get load average on Unix systems
            if hasattr(os, 'getloadavg'):
                load1, load5, load15 = os.getloadavg()
                metrics.update({
                    'load_avg_1min': load1,
                    'load_avg_5min': load5,
                    'load_avg_15min': load15
                })
        except:
            pass
        
        return metrics
    
    def _check_alerts(self, metrics: Dict[str, Any]) -> None:
        """Check for alert conditions"""
        current_alerts = set()
        
        for metric, threshold in self.alert_thresholds.items():
            if metric in metrics and metrics[metric] > threshold:
                alert_key = f"{metric}_high"
                current_alerts.add(alert_key)
                
                # New alert
                if alert_key not in self.active_alerts:
                    alert_data = {
                        'timestamp': datetime.now().isoformat(),
                        'type': 'threshold_exceeded',
                        'metric': metric,
                        'value': metrics[metric],
                        'threshold': threshold,
                        'severity': 'warning' if metrics[metric] < threshold * 1.2 else 'critical'
                    }
                    
                    self.alert_history.append(alert_data)
                    self.logger.warning(f"Alert: {metric} = {metrics[metric]:.1f}% (threshold: {threshold}%)")
        
        # Clear resolved alerts
        resolved_alerts = self.active_alerts - current_alerts
        for alert in resolved_alerts:
            self.logger.info(f"Alert resolved: {alert}")
        
        self.active_alerts = current_alerts
    
    def get_alert_history(self, limit: int = 50) -> list:
        """Get recent alert history"""
        return self.alert_history[-limit:]
    
    def clear_alert_history(self) -> None:
        """Clear alert history"""
        self.alert_history.clear()
        self.active_alerts.clear()


class HealthChecker:
    """Health check utilities for system components"""
    
    def __init__(self):
        self.checks = {}
        self.logger = logging.getLogger(__name__)
    
    def register_check(self, name: str, check_func: Callable[[], Dict[str, Any]]) -> None:
        """Register a health check function"""
        self.checks[name] = check_func
    
    def unregister_check(self, name: str) -> None:
        """Unregister a health check"""
        if name in self.checks:
            del self.checks[name]
    
    def run_check(self, name: str) -> Dict[str, Any]:
        """Run a specific health check"""
        if name not in self.checks:
            return {
                'name': name,
                'status': 'error',
                'message': 'Check not found',
                'timestamp': datetime.now().isoformat()
            }
        
        try:
            result = self.checks[name]()
            result.update({
                'name': name,
                'timestamp': datetime.now().isoformat()
            })
            return result
            
        except Exception as e:
            self.logger.error(f"Health check '{name}' failed: {e}")
            return {
                'name': name,
                'status': 'error',
                'message': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def run_all_checks(self) -> Dict[str, Any]:
        """Run all registered health checks"""
        results = {}
        overall_status = 'healthy'
        
        for name in self.checks:
            result = self.run_check(name)
            results[name] = result
            
            # Update overall status
            if result.get('status') == 'error':
                overall_status = 'unhealthy'
            elif result.get('status') == 'warning' and overall_status == 'healthy':
                overall_status = 'degraded'
        
        return {
            'overall_status': overall_status,
            'timestamp': datetime.now().isoformat(),
            'checks': results,
            'summary': {
                'total': len(results),
                'healthy': sum(1 for r in results.values() if r.get('status') == 'healthy'),
                'warning': sum(1 for r in results.values() if r.get('status') == 'warning'),
                'error': sum(1 for r in results.values() if r.get('status') == 'error')
            }
        }
