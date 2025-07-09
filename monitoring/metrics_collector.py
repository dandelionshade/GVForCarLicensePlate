# -*- coding: utf-8 -*-
"""
Metrics collection utilities
"""

import time
import json
import threading
from typing import Dict, Any, List, Optional, Union
from collections import defaultdict, deque
from datetime import datetime
import logging


class MetricsCollector:
    """Collect and aggregate metrics for the system"""
    
    def __init__(self, max_history: int = 10000):
        self.max_history = max_history
        self.metrics = defaultdict(lambda: deque(maxlen=max_history))
        self.counters = defaultdict(int)
        self.gauges = defaultdict(float)
        self.lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
    
    def record_metric(self, name: str, value: Union[int, float], tags: Optional[Dict[str, str]] = None) -> None:
        """Record a metric value"""
        timestamp = time.time()
        metric_data = {
            'timestamp': timestamp,
            'value': value,
            'tags': tags or {}
        }
        
        with self.lock:
            self.metrics[name].append(metric_data)
    
    def increment_counter(self, name: str, value: int = 1, tags: Optional[Dict[str, str]] = None) -> None:
        """Increment a counter metric"""
        with self.lock:
            self.counters[name] += value
        self.record_metric(name, self.counters[name], tags)
    
    def set_gauge(self, name: str, value: Union[int, float], tags: Optional[Dict[str, str]] = None) -> None:
        """Set a gauge metric"""
        with self.lock:
            self.gauges[name] = value
        self.record_metric(name, value, tags)
    
    def record_timing(self, name: str, duration: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Record a timing metric"""
        timing_tags = (tags or {}).copy()
        timing_tags['type'] = 'timing'
        self.record_metric(name, duration, timing_tags)
    
    def get_metric_summary(self, name: str, time_window: Optional[float] = None) -> Dict[str, Any]:
        """Get summary statistics for a metric"""
        with self.lock:
            metric_data = list(self.metrics[name])
        
        if not metric_data:
            return {
                'name': name,
                'count': 0,
                'min': 0,
                'max': 0,
                'avg': 0,
                'sum': 0
            }
        
        # Filter by time window if specified
        if time_window:
            cutoff_time = time.time() - time_window
            metric_data = [m for m in metric_data if m['timestamp'] >= cutoff_time]
        
        if not metric_data:
            return {
                'name': name,
                'count': 0,
                'min': 0,
                'max': 0,
                'avg': 0,
                'sum': 0
            }
        
        values = [m['value'] for m in metric_data]
        
        return {
            'name': name,
            'count': len(values),
            'min': min(values),
            'max': max(values),
            'avg': sum(values) / len(values),
            'sum': sum(values),
            'latest': values[-1] if values else 0
        }
    
    def get_all_metrics_summary(self, time_window: Optional[float] = None) -> Dict[str, Dict[str, Any]]:
        """Get summary for all metrics"""
        with self.lock:
            metric_names = list(self.metrics.keys())
        
        return {name: self.get_metric_summary(name, time_window) for name in metric_names}
    
    def get_counter_values(self) -> Dict[str, int]:
        """Get current counter values"""
        with self.lock:
            return dict(self.counters)
    
    def get_gauge_values(self) -> Dict[str, float]:
        """Get current gauge values"""
        with self.lock:
            return dict(self.gauges)
    
    def reset_counters(self) -> None:
        """Reset all counters to zero"""
        with self.lock:
            self.counters.clear()
    
    def clear_metrics(self, name: Optional[str] = None) -> None:
        """Clear metrics data"""
        with self.lock:
            if name:
                if name in self.metrics:
                    self.metrics[name].clear()
                if name in self.counters:
                    del self.counters[name]
                if name in self.gauges:
                    del self.gauges[name]
            else:
                self.metrics.clear()
                self.counters.clear()
                self.gauges.clear()
    
    def export_metrics(self, filepath: str, time_window: Optional[float] = None) -> None:
        """Export metrics to JSON file"""
        try:
            data = {
                'timestamp': datetime.now().isoformat(),
                'time_window': time_window,
                'metrics_summary': self.get_all_metrics_summary(time_window),
                'counters': self.get_counter_values(),
                'gauges': self.get_gauge_values()
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
            
            self.logger.info(f"Metrics exported to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to export metrics: {e}")


class TimingContext:
    """Context manager for recording timing metrics"""
    
    def __init__(self, collector: MetricsCollector, metric_name: str, tags: Optional[Dict[str, str]] = None):
        self.collector = collector
        self.metric_name = metric_name
        self.tags = tags
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            duration = time.time() - self.start_time
            self.collector.record_timing(self.metric_name, duration, self.tags)
        return False  # Don't suppress exceptions


class MetricsAggregator:
    """Aggregate metrics from multiple sources"""
    
    def __init__(self):
        self.collectors = {}
        self.lock = threading.Lock()
    
    def add_collector(self, name: str, collector: MetricsCollector) -> None:
        """Add a metrics collector"""
        with self.lock:
            self.collectors[name] = collector
    
    def remove_collector(self, name: str) -> None:
        """Remove a metrics collector"""
        with self.lock:
            if name in self.collectors:
                del self.collectors[name]
    
    def get_aggregated_summary(self, time_window: Optional[float] = None) -> Dict[str, Any]:
        """Get aggregated summary from all collectors"""
        aggregated = {
            'timestamp': datetime.now().isoformat(),
            'collectors': {},
            'totals': defaultdict(lambda: {'count': 0, 'sum': 0, 'min': float('inf'), 'max': float('-inf')})
        }
        
        with self.lock:
            for collector_name, collector in self.collectors.items():
                collector_summary = collector.get_all_metrics_summary(time_window)
                aggregated['collectors'][collector_name] = collector_summary
                
                # Aggregate totals
                for metric_name, metric_data in collector_summary.items():
                    totals = aggregated['totals'][metric_name]
                    totals['count'] += metric_data['count']
                    totals['sum'] += metric_data['sum']
                    if metric_data['count'] > 0:
                        totals['min'] = min(totals['min'], metric_data['min'])
                        totals['max'] = max(totals['max'], metric_data['max'])
        
        # Calculate averages and clean up infinities
        for metric_name, totals in aggregated['totals'].items():
            if totals['count'] > 0:
                totals['avg'] = totals['sum'] / totals['count']
                if totals['min'] == float('inf'):
                    totals['min'] = 0
                if totals['max'] == float('-inf'):
                    totals['max'] = 0
            else:
                totals['min'] = 0
                totals['max'] = 0
                totals['avg'] = 0
        
        return aggregated
