# -*- coding: utf-8 -*-
"""
Performance monitoring utilities
"""

import time
import psutil
import threading
from typing import Dict, Any, List, Optional
from collections import deque, defaultdict
from datetime import datetime
import logging


class PerformanceMonitor:
    """Performance monitoring for license plate recognition system"""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.metrics = defaultdict(lambda: deque(maxlen=max_history))
        self.active_requests = {}
        self.lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
        
        # System monitoring
        self.system_metrics = defaultdict(lambda: deque(maxlen=max_history))
        self._monitoring = False
        self._monitor_thread = None
    
    def start_request(self, request_id: str, metadata: Dict[str, Any] = None) -> None:
        """Start monitoring a request"""
        with self.lock:
            self.active_requests[request_id] = {
                'start_time': time.time(),
                'metadata': metadata or {},
                'checkpoints': []
            }
    
    def add_checkpoint(self, request_id: str, checkpoint_name: str) -> None:
        """Add a checkpoint to track progress"""
        with self.lock:
            if request_id in self.active_requests:
                current_time = time.time()
                start_time = self.active_requests[request_id]['start_time']
                self.active_requests[request_id]['checkpoints'].append({
                    'name': checkpoint_name,
                    'timestamp': current_time,
                    'elapsed': current_time - start_time
                })
    
    def end_request(self, request_id: str, success: bool = True, error: str = None) -> Dict[str, Any]:
        """End monitoring a request and record metrics"""
        with self.lock:
            if request_id not in self.active_requests:
                return {}
            
            request_data = self.active_requests.pop(request_id)
            end_time = time.time()
            total_duration = end_time - request_data['start_time']
            
            # Record metrics
            metrics = {
                'request_id': request_id,
                'duration': total_duration,
                'success': success,
                'error': error,
                'timestamp': datetime.now().isoformat(),
                'checkpoints': request_data['checkpoints'],
                'metadata': request_data['metadata']
            }
            
            # Store in appropriate metric category
            category = request_data['metadata'].get('category', 'general')
            self.metrics[f'{category}_requests'].append(metrics)
            self.metrics[f'{category}_durations'].append(total_duration)
            
            if success:
                self.metrics[f'{category}_success'].append(1)
            else:
                self.metrics[f'{category}_errors'].append(error or 'Unknown error')
            
            return metrics
    
    def get_metrics_summary(self, category: str = 'general') -> Dict[str, Any]:
        """Get summary of metrics for a category"""
        with self.lock:
            durations = list(self.metrics[f'{category}_durations'])
            requests = list(self.metrics[f'{category}_requests'])
            success_count = len(self.metrics[f'{category}_success'])
            error_count = len(self.metrics[f'{category}_errors'])
            
            if not durations:
                return {
                    'category': category,
                    'total_requests': 0,
                    'success_rate': 0.0,
                    'avg_duration': 0.0,
                    'min_duration': 0.0,
                    'max_duration': 0.0
                }
            
            total_requests = len(requests)
            
            return {
                'category': category,
                'total_requests': total_requests,
                'success_count': success_count,
                'error_count': error_count,
                'success_rate': success_count / total_requests if total_requests > 0 else 0.0,
                'avg_duration': sum(durations) / len(durations),
                'min_duration': min(durations),
                'max_duration': max(durations),
                'recent_errors': list(self.metrics[f'{category}_errors'])[-5:]  # Last 5 errors
            }
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        categories = set()
        for key in self.metrics.keys():
            if '_' in key:
                category = key.split('_')[0]
                categories.add(category)
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'categories': {},
            'system_metrics': self.get_system_metrics(),
            'active_requests': len(self.active_requests)
        }
        
        for category in categories:
            report['categories'][category] = self.get_metrics_summary(category)
        
        return report
    
    def start_system_monitoring(self, interval: float = 1.0) -> None:
        """Start system resource monitoring"""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(
            target=self._system_monitor_loop,
            args=(interval,),
            daemon=True
        )
        self._monitor_thread.start()
        self.logger.info("System monitoring started")
    
    def stop_system_monitoring(self) -> None:
        """Stop system resource monitoring"""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
        self.logger.info("System monitoring stopped")
    
    def _system_monitor_loop(self, interval: float) -> None:
        """System monitoring loop"""
        while self._monitoring:
            try:
                # Collect system metrics
                cpu_percent = psutil.cpu_percent()
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                
                timestamp = time.time()
                
                with self.lock:
                    self.system_metrics['cpu_percent'].append((timestamp, cpu_percent))
                    self.system_metrics['memory_percent'].append((timestamp, memory.percent))
                    self.system_metrics['memory_used_mb'].append((timestamp, memory.used / 1024 / 1024))
                    self.system_metrics['disk_percent'].append((timestamp, disk.percent))
                
                time.sleep(interval)
                
            except Exception as e:
                self.logger.error(f"Error in system monitoring: {e}")
                time.sleep(interval)
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get current system metrics"""
        with self.lock:
            if not self.system_metrics:
                return {}
            
            # Get recent metrics (last 60 measurements)
            recent_limit = 60
            
            metrics = {}
            for metric_name, values in self.system_metrics.items():
                recent_values = list(values)[-recent_limit:]
                if recent_values:
                    timestamps, measurements = zip(*recent_values)
                    metrics[metric_name] = {
                        'current': measurements[-1],
                        'avg': sum(measurements) / len(measurements),
                        'min': min(measurements),
                        'max': max(measurements),
                        'count': len(measurements)
                    }
            
            return metrics
    
    def clear_metrics(self, category: Optional[str] = None) -> None:
        """Clear metrics for a category or all categories"""
        with self.lock:
            if category:
                keys_to_clear = [key for key in self.metrics.keys() if key.startswith(f'{category}_')]
                for key in keys_to_clear:
                    self.metrics[key].clear()
            else:
                self.metrics.clear()
                self.system_metrics.clear()
    
    def export_metrics(self, filepath: str) -> None:
        """Export metrics to file"""
        import json
        
        with self.lock:
            data = {
                'timestamp': datetime.now().isoformat(),
                'metrics': {key: list(values) for key, values in self.metrics.items()},
                'system_metrics': {key: list(values) for key, values in self.system_metrics.items()},
                'active_requests': len(self.active_requests)
            }
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
            self.logger.info(f"Metrics exported to {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to export metrics: {e}")


# Context manager for easy request monitoring
class RequestMonitor:
    """Context manager for monitoring individual requests"""
    
    def __init__(self, monitor: PerformanceMonitor, request_id: str, category: str = 'general', metadata: Dict[str, Any] = None):
        self.monitor = monitor
        self.request_id = request_id
        self.metadata = metadata or {}
        self.metadata['category'] = category
        self.success = True
        self.error = None
    
    def __enter__(self):
        self.monitor.start_request(self.request_id, self.metadata)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.success = False
            self.error = str(exc_val)
        
        self.monitor.end_request(self.request_id, self.success, self.error)
        return False  # Don't suppress exceptions
    
    def checkpoint(self, name: str):
        """Add a checkpoint"""
        self.monitor.add_checkpoint(self.request_id, name)
