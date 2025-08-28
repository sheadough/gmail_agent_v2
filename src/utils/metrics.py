# src/utils/metrics.py
from typing import Dict, Any, List
import time
import psutil
import threading
from datetime import datetime, timedelta
from collections import defaultdict, deque
import json
import logging
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

@dataclass
class RequestMetrics:
    """Individual request metrics"""
    endpoint: str
    response_time: float
    timestamp: datetime
    success: bool
    status_code: int = 200

class MetricsCollector:
    """Comprehensive metrics collection system"""
    
    def __init__(self, max_history: int = 10000):
        self.max_history = max_history
        
        # Request metrics
        self.request_history = deque(maxlen=max_history)
        self.endpoint_stats = defaultdict(lambda: {
            'count': 0,
            'total_time': 0.0,
            'success_count': 0,
            'error_count': 0,
            'avg_response_time': 0.0
        })
        
        # System metrics
        self.system_metrics_history = deque(maxlen=1000)
        
        # Agent-specific metrics
        self.agent_metrics = {
            'total_interactions': 0,
            'successful_interactions': 0,
            'tool_usage_count': defaultdict(int),
            'error_types': defaultdict(int)
        }
        
        # Start background monitoring
        self._start_system_monitoring()
        
        # Performance alerts
        self.alerts = []
        self.alert_thresholds = {
            'response_time': 5.0,  # seconds
            'error_rate': 0.1,     # 10%
            'cpu_usage': 80,       # percent
            'memory_usage': 85     # percent
        }
    
    def record_request(self, endpoint: str, response_time: float, success: bool, status_code: int = 200):
        """Record a request metric"""
        try:
            # Create request record
            request_metric = RequestMetrics(
                endpoint=endpoint,
                response_time=response_time,
                timestamp=datetime.now(),
                success=success,
                status_code=status_code
            )
            
            self.request_history.append(request_metric)
            
            # Update endpoint statistics
            stats = self.endpoint_stats[endpoint]
            stats['count'] += 1
            stats['total_time'] += response_time
            
            if success:
                stats['success_count'] += 1
            else:
                stats['error_count'] += 1
            
            # Update average response time
            stats['avg_response_time'] = stats['total_time'] / stats['count']
            
            # Check for alerts
            self._check_performance_alerts(endpoint, response_time, success)
            
        except Exception as e:
            logger.error(f"Failed to record request metric: {e}")
    
    def record_agent_interaction(self, tools_used: List[str], success: bool, error_type: str = None):
        """Record agent-specific metrics"""
        try:
            self.agent_metrics['total_interactions'] += 1
            
            if success:
                self.agent_metrics['successful_interactions'] += 1
            
            # Record tool usage
            for tool in tools_used:
                self.agent_metrics['tool_usage_count'][tool] += 1
            
            # Record error type if provided
            if error_type:
                self.agent_metrics['error_types'][error_type] += 1
                
        except Exception as e:
            logger.error(f"Failed to record agent interaction: {e}")
    
    def _start_system_monitoring(self):
        """Start background system monitoring"""
        def collect_system_metrics():
            while True:
                try:
                    metrics = {
                        'timestamp': datetime.now(),
                        'cpu_percent': psutil.cpu_percent(interval=1),
                        'memory_percent': psutil.virtual_memory().percent,
                        'memory_used_gb': psutil.virtual_memory().used / (1024**3),
                        'memory_available_gb': psutil.virtual_memory().available / (1024**3),
                        'disk_usage_percent': psutil.disk_usage('/').percent,
                        'disk_free_gb': psutil.disk_usage('/').free / (1024**3),
                        'network_sent_mb': psutil.net_io_counters().bytes_sent / (1024**2),
                        'network_recv_mb': psutil.net_io_counters().bytes_recv / (1024**2),
                        'process_count': len(psutil.pids())
                    }
                    
                    self.system_metrics_history.append(metrics)
                    
                    # Check system alerts
                    self._check_system_alerts(metrics)
                    
                except Exception as e:
                    logger.error(f"Error collecting system metrics: {e}")
                
                time.sleep(30)  # Collect every 30 seconds
        
        thread = threading.Thread(target=collect_system_metrics, daemon=True)
        thread.start()
    
    def _check_performance_alerts(self, endpoint: str, response_time: float, success: bool):
        """Check for performance-related alerts"""
        try:
            # Response time alert
            if response_time > self.alert_thresholds['response_time']:
                self._add_alert(
                    'high_response_time',
                    f"High response time on {endpoint}: {response_time:.2f}s",
                    'warning'
                )
            
            # Error rate alert (check last 100 requests for this endpoint)
            recent_requests = [r for r in self.request_history 
                             if r.endpoint == endpoint and 
                             r.timestamp > datetime.now() - timedelta(minutes=10)]
            
            if len(recent_requests) >= 10:
                error_rate = sum(1 for r in recent_requests if not r.success) / len(recent_requests)
                if error_rate > self.alert_thresholds['error_rate']:
                    self._add_alert(
                        'high_error_rate',
                        f"High error rate on {endpoint}: {error_rate:.1%}",
                        'critical'
                    )
                    
        except Exception as e:
            logger.error(f"Error checking performance alerts: {e}")
    
    def _check_system_alerts(self, metrics: Dict[str, Any]):
        """Check for system-related alerts"""
        try:
            # CPU usage alert
            if metrics['cpu_percent'] > self.alert_thresholds['cpu_usage']:
                self._add_alert(
                    'high_cpu_usage',
                    f"High CPU usage: {metrics['cpu_percent']:.1f}%",
                    'warning'
                )
            
            # Memory usage alert
            if metrics['memory_percent'] > self.alert_thresholds['memory_usage']:
                self._add_alert(
                    'high_memory_usage',
                    f"High memory usage: {metrics['memory_percent']:.1f}%",
                    'critical' if metrics['memory_percent'] > 95 else 'warning'
                )
            
            # Disk space alert
            if metrics['disk_usage_percent'] > 90:
                self._add_alert(
                    'low_disk_space',
                    f"Low disk space: {metrics['disk_usage_percent']:.1f}% used",
                    'critical'
                )
                
        except Exception as e:
            logger.error(f"Error checking system alerts: {e}")
    
    def _add_alert(self, alert_type: str, message: str, severity: str):
        """Add an alert"""
        try:
            alert = {
                'type': alert_type,
                'message': message,
                'severity': severity,
                'timestamp': datetime.now(),
                'resolved': False
            }
            
            # Avoid duplicate recent alerts
            recent_alerts = [a for a in self.alerts 
                           if a['type'] == alert_type and 
                           a['timestamp'] > datetime.now() - timedelta(minutes=5)]
            
            if not recent_alerts:
                self.alerts.append(alert)
                logger.warning(f"ALERT [{severity.upper()}]: {message}")
                
                # Keep only last 100 alerts
                if len(self.alerts) > 100:
                    self.alerts = self.alerts[-100:]
                    
        except Exception as e:
            logger.error(f"Error adding alert: {e}")
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary"""
        try:
            now = datetime.now()
            
            # Request metrics
            total_requests = len(self.request_history)
            if total_requests > 0:
                successful_requests = sum(1 for r in self.request_history if r.success)
                total_response_time = sum(r.response_time for r in self.request_history)
                avg_response_time = total_response_time / total_requests
                
                # Recent performance (last hour)
                recent_requests = [r for r in self.request_history 
                                 if r.timestamp > now - timedelta(hours=1)]
                recent_avg_time = (sum(r.response_time for r in recent_requests) / 
                                 len(recent_requests)) if recent_requests else 0
            else:
                successful_requests = 0
                avg_response_time = 0
                recent_avg_time = 0
            
            # System metrics (latest)
            latest_system = self.system_metrics_history[-1] if self.system_metrics_history else {}
            
            # Endpoint performance
            endpoint_performance = {}
            for endpoint, stats in self.endpoint_stats.items():
                endpoint_performance[endpoint] = {
                    'total_requests': stats['count'],
                    'success_rate': (stats['success_count'] / stats['count']) * 100 if stats['count'] > 0 else 0,
                    'avg_response_time': stats['avg_response_time'],
                    'error_count': stats['error_count']
                }
            
            # Active alerts
            active_alerts = [a for a in self.alerts if not a.get('resolved', False)]
            
            return {
                'timestamp': now.isoformat(),
                'request_metrics': {
                    'total_requests': total_requests,
                    'successful_requests': successful_requests,
                    'error_rate_percent': ((total_requests - successful_requests) / total_requests * 100) if total_requests > 0 else 0,
                    'avg_response_time': avg_response_time,
                    'recent_avg_response_time': recent_avg_time
                },
                'system_metrics': {
                    'cpu_percent': latest_system.get('cpu_percent', 0),
                    'memory_percent': latest_system.get('memory_percent', 0),
                    'disk_usage_percent': latest_system.get('disk_usage_percent', 0),
                    'memory_used_gb': latest_system.get('memory_used_gb', 0),
                    'disk_free_gb': latest_system.get('disk_free_gb', 0)
                },
                'agent_metrics': {
                    **self.agent_metrics,
                    'tool_usage_count': dict(self.agent_metrics['tool_usage_count']),
                    'error_types': dict(self.agent_metrics['error_types']),
                    'success_rate': (self.agent_metrics['successful_interactions'] / 
                                   max(1, self.agent_metrics['total_interactions'])) * 100
                },
                'endpoint_performance': endpoint_performance,
                'alerts': {
                    'active_count': len(active_alerts),
                    'recent_alerts': active_alerts[-10:],  # Last 10 alerts
                    'alert_summary': self._get_alert_summary()
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to generate metrics summary: {e}")
            return {'error': str(e), 'timestamp': datetime.now().isoformat()}
    
    def _get_alert_summary(self) -> Dict[str, int]:
        """Get summary of alerts by type and severity"""
        try:
            summary = defaultdict(int)
            
            recent_alerts = [a for a in self.alerts 
                           if a['timestamp'] > datetime.now() - timedelta(hours=24)]
            
            for alert in recent_alerts:
                summary[f"{alert['severity']}_alerts"] += 1
                summary[f"{alert['type']}_count"] += 1
            
            return dict(summary)
            
        except Exception as e:
            logger.error(f"Error generating alert summary: {e}")
            return {}
    
    def export_metrics(self, filepath: str = None) -> str:
        """Export metrics to JSON file"""
        try:
            metrics_data = self.get_metrics_summary()
            
            if not filepath:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filepath = f"metrics_export_{timestamp}.json"
            
            with open(filepath, 'w') as f:
                json.dump(metrics_data, f, indent=2, default=str)
            
            logger.info(f"Metrics exported to {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Failed to export metrics: {e}")
            raise

# Global metrics collector instance
metrics_collector = MetricsCollector()

# Metrics dashboard endpoint for development
def create_metrics_dashboard() -> str:
    """Create a simple HTML dashboard for metrics"""
    return """
<!DOCTYPE html>
<html>
<head>
    <title>AI Agent Metrics Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .metric-card { border: 1px solid #ddd; padding: 15px; margin: 10px; border-radius: 5px; }
        .alert { padding: 10px; margin: 5px; border-radius: 3px; }
        .alert-warning { background-color: #fff3cd; border-color: #ffeaa7; }
        .alert-critical { background-color: #f8d7da; border-color: #f1c0c7; }
        .chart { height: 400px; margin: 20px 0; }
    </style>
</head>
<body>
    <h1>AI Agent Metrics Dashboard</h1>
    
    <div class="metric-card">
        <h3>System Status</h3>
        <div id="system-metrics"></div>
    </div>
    
    <div class="metric-card">
        <h3>Request Performance</h3>
        <div id="performance-chart" class="chart"></div>
    </div>
    
    <div class="metric-card">
        <h3>Recent Alerts</h3>
        <div id="alerts"></div>
    </div>
    
    <script>
        // Fetch and display metrics
        async function loadMetrics() {
            try {
                const response = await fetch('/api/v1/metrics');
                const data = await response.json();
                
                // Display system metrics
                document.getElementById('system-metrics').innerHTML = `
                    <p>CPU: ${data.system_metrics.cpu_percent}%</p>
                    <p>Memory: ${data.system_metrics.memory_percent}%</p>
                    <p>Disk: ${data.system_metrics.disk_usage_percent}%</p>
                `;
                
                // Display alerts
                const alertsDiv = document.getElementById('alerts');
                if (data.alerts.active_count > 0) {
                    alertsDiv.innerHTML = data.alerts.recent_alerts.map(alert => 
                        `<div class="alert alert-${alert.severity}">${alert.message}</div>`
                    ).join('');
                } else {
                    alertsDiv.innerHTML = '<p>No active alerts</p>';
                }
                
            } catch (error) {
                console.error('Failed to load metrics:', error);
            }
        }
        
        // Load metrics on page load
        loadMetrics();
               // Refresh metrics every 30 seconds
       setInterval(loadMetrics, 30000);
   </script>
</body>
</html>
"""