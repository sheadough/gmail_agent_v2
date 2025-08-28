# tests/test_production_api.py
import pytest
import asyncio
import httpx
from typing import Dict, Any
import time
from datetime import datetime
import json

# Test configuration
API_BASE_URL = "http://localhost:8000"
TEST_USER_ID = "test_user_001"

class TestProductionAPI:
    """Comprehensive API testing suite"""
    
    @pytest.fixture(scope="class")
    async def client(self):
        """Create HTTP client for testing"""
        async with httpx.AsyncClient(base_url=API_BASE_URL, timeout=30.0) as client:
            yield client
    
    @pytest.mark.asyncio
    async def test_health_check(self, client):
        """Test health check endpoint"""
        response = await client.get("/api/v1/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "status" in data
        assert "timestamp" in data
        assert "components" in data
        assert data["status"] in ["healthy", "degraded", "unhealthy"]
    
    @pytest.mark.asyncio
    async def test_root_endpoint(self, client):
        """Test root endpoint"""
        response = await client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["name"] == "AI Agent API"
        assert "version" in data
        assert data["status"] == "active"
    
    @pytest.mark.asyncio
    async def test_chat_endpoint_basic(self, client):
        """Test basic chat functionality"""
        payload = {
            "message": "Hello, can you introduce yourself?",
            "user_id": TEST_USER_ID
        }
        
        response = await client.post("/api/v1/chat", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        
        # Check response structure
        required_fields = ["agent_id", "response", "analysis", "tools_used", 
                          "processing_time", "success", "timestamp", "user_id"]
        for field in required_fields:
            assert field in data
        
        assert data["success"] is True
        assert data["user_id"] == TEST_USER_ID
        assert len(data["response"]) > 0
    
    @pytest.mark.asyncio
    async def test_chat_with_calculator(self, client):
        """Test chat with calculator tool usage"""
        payload = {
            "message": "What is the square root of 144?",
            "user_id": TEST_USER_ID
        }
        
        response = await client.post("/api/v1/chat", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert "calculator" in data["tools_used"]
        assert "12" in data["response"]  # sqrt(144) = 12
    
    @pytest.mark.asyncio
    async def test_chat_with_web_search(self, client):
        """Test chat with web search tool"""
        payload = {
            "message": "Search for information about artificial intelligence",
            "user_id": TEST_USER_ID
        }
        
        response = await client.post("/api/v1/chat", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert "web_search" in data["tools_used"]
    
    @pytest.mark.asyncio
    async def test_invalid_chat_request(self, client):
        """Test invalid chat request"""
        payload = {
            "message": "",  # Empty message
            "user_id": TEST_USER_ID
        }
        
        response = await client.post("/api/v1/chat", json=payload)
        
        assert response.status_code == 422  # Validation error
    
    @pytest.mark.asyncio
    async def test_capabilities_endpoint(self, client):
        """Test agent capabilities endpoint"""
        response = await client.get("/api/v1/agent/capabilities")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "capabilities" in data
        assert "available_tools" in data
        assert "tool_descriptions" in data
        assert len(data["available_tools"]) > 0
    
    @pytest.mark.asyncio
    async def test_metrics_endpoint(self, client):
        """Test metrics endpoint"""
        response = await client.get("/api/v1/metrics")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "system_metrics" in data
        assert "agent_metrics" in data
        assert "api_metrics" in data
    
    @pytest.mark.asyncio
    async def test_memory_search(self, client):
        """Test memory search endpoint"""
        # First, have a conversation to create memories
        chat_payload = {
            "message": "I love pizza and my favorite color is blue",
            "user_id": TEST_USER_ID
        }
        await client.post("/api/v1/chat", json=chat_payload)
        
        # Wait a bit for memory to be stored
        await asyncio.sleep(1)
        
        # Search memory
        search_payload = {
            "query": "pizza",
            "max_results": 5
        }
        
        response = await client.post("/api/v1/memory/search", json=search_payload)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "query" in data
        assert "results" in data
        assert data["query"] == "pizza"
    
    @pytest.mark.asyncio
    async def test_memory_clear(self, client):
        """Test memory clear endpoint"""
        response = await client.delete("/api/v1/memory/clear")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "message" in data
        assert "timestamp" in data
    
    @pytest.mark.asyncio
    async def test_agent_stats(self, client):
        """Test agent statistics endpoint"""
        response = await client.get("/api/v1/agent/stats")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "agent_id" in data
        assert "integration_stats" in data
        assert "langchain_performance" in data

class TestPerformance:
    """Performance testing suite"""
    
    @pytest.fixture(scope="class")
    async def client(self):
        """Create HTTP client for testing"""
        async with httpx.AsyncClient(base_url=API_BASE_URL, timeout=60.0) as client:
            yield client
    
    @pytest.mark.asyncio
    async def test_response_time_basic(self, client):
        """Test basic response time requirements"""
        payload = {
            "message": "Hello",
            "user_id": "perf_test_user"
        }
        
        start_time = time.time()
        response = await client.post("/api/v1/chat", json=payload)
        end_time = time.time()
        
        assert response.status_code == 200
        response_time = end_time - start_time
        
        # Should respond within 10 seconds for basic queries
        assert response_time < 10.0, f"Response time too slow: {response_time:.2f}s"
    
    @pytest.mark.asyncio
    async def test_concurrent_requests(self, client):
        """Test handling concurrent requests"""
        
        async def make_request(user_id: str):
            payload = {
                "message": f"Hello from user {user_id}",
                "user_id": f"concurrent_user_{user_id}"
            }
            
            start_time = time.time()
            response = await client.post("/api/v1/chat", json=payload)
            end_time = time.time()
            
            return {
                "status_code": response.status_code,
                "response_time": end_time - start_time,
                "success": response.status_code == 200
            }
        
        # Make 5 concurrent requests
        tasks = [make_request(str(i)) for i in range(5)]
        results = await asyncio.gather(*tasks)
        
        # All requests should succeed
        assert all(result["success"] for result in results)
        
        # Average response time should be reasonable
        avg_response_time = sum(result["response_time"] for result in results) / len(results)
        assert avg_response_time < 15.0, f"Average response time too slow: {avg_response_time:.2f}s"
    
    @pytest.mark.asyncio
    async def test_memory_usage(self, client):
        """Test memory doesn't grow excessively with multiple requests"""
        import psutil
        import os
        
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Make multiple requests
        for i in range(20):
            payload = {
                "message": f"Test message number {i}",
                "user_id": f"memory_test_user_{i}"
            }
            
            response = await client.post("/api/v1/chat", json=payload)
            assert response.status_code == 200
            
            await asyncio.sleep(0.1)  # Brief pause
        
        # Check final memory usage
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 100MB for 20 requests)
        assert memory_increase < 100, f"Memory usage increased too much: {memory_increase:.2f}MB"

class TestErrorHandling:
    """Error handling and edge cases testing"""
    
    @pytest.fixture(scope="class")
    async def client(self):
        async with httpx.AsyncClient(base_url=API_BASE_URL, timeout=30.0) as client:
            yield client
    
    @pytest.mark.asyncio
    async def test_malformed_json(self, client):
        """Test handling of malformed JSON"""
        response = await client.post(
            "/api/v1/chat",
            content='{"message": "test", "user_id":}',  # Invalid JSON
            headers={"content-type": "application/json"}
        )
        
        assert response.status_code == 422
    
    @pytest.mark.asyncio
    async def test_very_long_message(self, client):
        """Test handling of extremely long messages"""
        long_message = "x" * 20000  # 20k characters
        
        payload = {
            "message": long_message,
            "user_id": TEST_USER_ID
        }
        
        response = await client.post("/api/v1/chat", json=payload)
        
        # Should either process successfully or return validation error
        assert response.status_code in [200, 422]
    
    @pytest.mark.asyncio
    async def test_special_characters(self, client):
        """Test handling of special characters and emojis"""
        payload = {
            "message": "Hello! 🚀 Can you handle émojis and spëcial charâcters? 中文 العربية 🎉",
            "user_id": TEST_USER_ID
        }
        
        response = await client.post("/api/v1/chat", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
    
    @pytest.mark.asyncio
    async def test_nonexistent_endpoint(self, client):
        """Test accessing non-existent endpoints"""
        response = await client.get("/api/v1/nonexistent")
        
        assert response.status_code == 404

# Load testing utilities
class LoadTester:
    """Load testing utilities for production validation"""
    
    def __init__(self, base_url: str = API_BASE_URL):
        self.base_url = base_url
        self.results = []
    
    async def run_load_test(self, num_requests: int = 50, concurrent_users: int = 5):
        """Run load test with specified parameters"""
        print(f"🔥 Starting load test: {num_requests} requests, {concurrent_users} concurrent users")
        
        async def user_session(user_id: int, requests_per_user: int):
            """Simulate a user session"""
            user_results = []
            
            async with httpx.AsyncClient(base_url=self.base_url, timeout=60.0) as client:
                for i in range(requests_per_user):
                    start_time = time.time()
                    
                    try:
                        payload = {
                            "message": f"Load test message {i} from user {user_id}",
                            "user_id": f"load_test_user_{user_id}"
                        }
                        
                        response = await client.post("/api/v1/chat", json=payload)
                        
                        result = {
                            "user_id": user_id,
                            "request_id": i,
                            "response_time": time.time() - start_time,
                            "status_code": response.status_code,
                            "success": response.status_code == 200,
                            "timestamp": datetime.now()
                        }
                        
                        user_results.append(result)
                        
                    except Exception as e:
                        result = {
                            "user_id": user_id,
                            "request_id": i,
                            "response_time": time.time() - start_time,
                            "status_code": 0,
                            "success": False,
                            "error": str(e),
                            "timestamp": datetime.now()
                        }
                        
                        user_results.append(result)
                    
                    # Brief pause between requests
                    await asyncio.sleep(0.1)
            
            return user_results
        
        # Calculate requests per user
        requests_per_user = num_requests // concurrent_users
        
        # Run concurrent user sessions
        tasks = [
            user_session(user_id, requests_per_user) 
            for user_id in range(concurrent_users)
        ]
        
        all_results = await asyncio.gather(*tasks)
        
        # Flatten results
        self.results = [result for user_results in all_results for result in user_results]
        
        return self.analyze_results()
    
    def analyze_results(self) -> Dict[str, Any]:
        """Analyze load test results"""
        if not self.results:
            return {"error": "No results to analyze"}
        
        successful_requests = [r for r in self.results if r["success"]]
        failed_requests = [r for r in self.results if not r["success"]]
        
        response_times = [r["response_time"] for r in successful_requests]
        
        analysis = {
            "total_requests": len(self.results),
            "successful_requests": len(successful_requests),
            "failed_requests": len(failed_requests),
            "success_rate": len(successful_requests) / len(self.results) * 100,
            "response_times": {
                "min": min(response_times) if response_times else 0,
                "max": max(response_times) if response_times else 0,
                "mean": sum(response_times) / len(response_times) if response_times else 0,
                "p50": self._percentile(response_times, 50) if response_times else 0,
                "p95": self._percentile(response_times, 95) if response_times else 0,
                "p99": self._percentile(response_times, 99) if response_times else 0,
            },
            "throughput": len(successful_requests) / max(1, max(r["response_time"] for r in self.results)),
            "errors": {}
        }
        
        # Analyze error types
        for result in failed_requests:
            error_type = result.get("error", f"HTTP_{result['status_code']}")
            analysis["errors"][error_type] = analysis["errors"].get(error_type, 0) + 1
        
        return analysis
    
    def _percentile(self, values: list, percentile: float) -> float:
        """Calculate percentile of values"""
        if not values:
            return 0
        
        sorted_values = sorted(values)
        index = int(len(sorted_values) * percentile / 100)
        return sorted_values[min(index, len(sorted_values) - 1)]

# Run tests
if __name__ == "__main__":
    # Run pytest with coverage
    import subprocess
    
    result = subprocess.run([
        "pytest", 
        __file__, 
        "-v",
        "--tb=short",
        "--cov=src",
        "--cov-report=html",
        "--cov-report=term-missing"
    ])
    
    exit(result.returncode)

    if response.status_code != 200:
                           failures += 1
                           
                   except Exception:
                       failures += 1
                       response_times.append(60.0)  # Timeout
               
               if not response_times:
                   return False, "No successful responses"
               
               avg_response_time = sum(response_times) / len(response_times)
               max_response_time = max(response_times)
               
               issues = []
               
               # Performance thresholds
               if avg_response_time > 10.0:
                   issues.append(f"Average response time too high: {avg_response_time:.2f}s")
               
               if max_response_time > 30.0:
                   issues.append(f"Maximum response time too high: {max_response_time:.2f}s")
               
               if failures > 0:
                   issues.append(f"{failures} requests failed")
               
               if issues:
                   return False, f"Performance issues: {', '.join(issues)}"
               
               return True, f"Performance baseline met - Avg: {avg_response_time:.2f}s, Max: {max_response_time:.2f}s"
               
       except Exception as e:
           return False, f"Performance validation error: {e}"
   
   async def validate_security(self) -> Tuple[bool, str]:
       """Validate security configuration"""
       try:
           security_checks = []
           
           async with httpx.AsyncClient(timeout=10.0) as client:
               # Test security headers
               response = await client.get(f"{self.base_url}/api/v1/health")
               headers = response.headers
               
               security_headers = {
                   'x-content-type-options': 'nosniff',
                   'x-frame-options': 'DENY',
                   'x-xss-protection': '1; mode=block'
               }
               
               missing_headers = []
               for header, expected_value in security_headers.items():
                   if header not in headers:
                       missing_headers.append(header)
                   elif expected_value and headers[header] != expected_value:
                       missing_headers.append(f"{header} (incorrect value)")
               
               if missing_headers:
                   security_checks.append(f"Missing security headers: {', '.join(missing_headers)}")
               
               # Test for common vulnerabilities
               try:
                   # Test SQL injection attempt
                   malicious_payload = {
                       "message": "'; DROP TABLE users; --",
                       "user_id": "test"
                   }
                   response = await client.post(f"{self.base_url}/api/v1/chat", json=malicious_payload)
                   # Should handle gracefully, not crash
                   
                   # Test XSS attempt
                   xss_payload = {
                       "message": "<script>alert('xss')</script>",
                       "user_id": "test"
                   }
                   response = await client.post(f"{self.base_url}/api/v1/chat", json=xss_payload)
                   # Should sanitize or handle safely
                   
               except Exception:
                   security_checks.append("Vulnerability tests caused errors")
               
               # Check environment variables
               env_file = Path(".env")
               if env_file.exists():
                   security_checks.append("WARNING: .env file present in production")
               
               if security_checks:
                   return False, f"Security issues: {'; '.join(security_checks)}"
               
               return True, "Security configuration validated"
               
       except Exception as e:
           return False, f"Security validation error: {e}"
   
   async def validate_monitoring(self) -> Tuple[bool, str]:
       """Validate monitoring and logging setup"""
       try:
           monitoring_issues = []
           
           # Check log files
           log_dir = Path("logs")
           if not log_dir.exists():
               monitoring_issues.append("Log directory not found")
           else:
               log_files = list(log_dir.glob("*.log"))
               if not log_files:
                   monitoring_issues.append("No log files found")
           
           # Test metrics endpoint
           try:
               async with httpx.AsyncClient(timeout=10.0) as client:
                   response = await client.get(f"{self.base_url}/api/v1/metrics")
                   if response.status_code != 200:
                       monitoring_issues.append("Metrics endpoint not accessible")
                   else:
                       metrics_data = response.json()
                       required_metrics = ["system_metrics", "agent_metrics", "api_metrics"]
                       missing_metrics = [m for m in required_metrics if m not in metrics_data]
                       if missing_metrics:
                           monitoring_issues.append(f"Missing metrics: {', '.join(missing_metrics)}")
           except Exception:
               monitoring_issues.append("Metrics endpoint error")
           
           # Check if Prometheus/Grafana are accessible (if configured)
           monitoring_services = []
           try:
               async with httpx.AsyncClient(timeout=5.0) as client:
                   # Test Prometheus
                   try:
                       response = await client.get("http://localhost:9090/api/v1/status/config")
                       if response.status_code == 200:
                           monitoring_services.append("Prometheus")
                   except:
                       pass
                   
                   # Test Grafana
                   try:
                       response = await client.get("http://localhost:3000/api/health")
                       if response.status_code == 200:
                           monitoring_services.append("Grafana")
                   except:
                       pass
           except:
               pass
           
           if monitoring_issues:
               return False, f"Monitoring issues: {', '.join(monitoring_issues)}"
           
           services_msg = f" ({', '.join(monitoring_services)} available)" if monitoring_services else ""
           return True, f"Monitoring setup validated{services_msg}"
           
       except Exception as e:
           return False, f"Monitoring validation error: {e}"
   
   async def validate_database_connectivity(self) -> Tuple[bool, str]:
       """Validate database connections"""
       try:
           # Test vector database
           async with httpx.AsyncClient(timeout=10.0) as client:
               # Test memory search (which uses vector DB)
               response = await client.post(
                   f"{self.base_url}/api/v1/memory/search",
                   json={"query": "test", "max_results": 1}
               )
               
               if response.status_code not in [200, 404]:  # 404 is OK for empty memory
                   return False, f"Vector database connectivity issue: {response.status_code}"
           
           # Test traditional database (if configured)
           # This would depend on your specific database setup
           
           return True, "Database connectivity verified"
           
       except Exception as e:
           return False, f"Database validation error: {e}"
   
   async def validate_error_handling(self) -> Tuple[bool, str]:
       """Validate error handling and recovery"""
       try:
           error_scenarios = [
               # Invalid JSON
               ("invalid_json", '{"message": "test", invalid}'),
               # Missing required fields
               ("missing_fields", '{"user_id": "test"}'),
               # Extremely long input
               ("long_input", json.dumps({"message": "x" * 50000, "user_id": "test"})),
           ]
           
           async with httpx.AsyncClient(timeout=30.0) as client:
               for scenario_name, payload in error_scenarios:
                   try:
                       response = await client.post(
                           f"{self.base_url}/api/v1/chat",
                           content=payload,
                           headers={"content-type": "application/json"}
                       )
                       
                       # Should return proper error codes, not crash
                       if response.status_code == 200:
                           return False, f"Error scenario '{scenario_name}' was not handled properly"
                       
                       # Should return valid JSON even for errors
                       try:
                           response.json()
                       except:
                           return False, f"Error response for '{scenario_name}' is not valid JSON"
                           
                   except Exception as e:
                       return False, f"Error scenario '{scenario_name}' caused exception: {e}"
           
           return True, "Error handling validated for all scenarios"
           
       except Exception as e:
           return False, f"Error handling validation failed: {e}"
   
   async def validate_load_capacity(self) -> Tuple[bool, str]:
       """Validate system can handle expected load"""
       try:
           print("   Running load test (this may take a moment)...")
           
           # Moderate load test - 20 concurrent requests
           async def make_concurrent_request(request_id: int):
               async with httpx.AsyncClient(timeout=60.0) as client:
                   start_time = time.time()
                   try:
                       response = await client.post(
                           f"{self.base_url}/api/v1/chat",
                           json={
                               "message": f"Load test request {request_id}",
                               "user_id": f"load_test_user_{request_id}"
                           }
                       )
                       return {
                           "success": response.status_code == 200,
                           "response_time": time.time() - start_time,
                           "status_code": response.status_code
                       }
                   except Exception as e:
                       return {
                           "success": False,
                           "response_time": time.time() - start_time,
                           "error": str(e)
                       }
           
           # Run 20 concurrent requests
           tasks = [make_concurrent_request(i) for i in range(20)]
           results = await asyncio.gather(*tasks)
           
           successful_requests = sum(1 for r in results if r["success"])
           success_rate = successful_requests / len(results) * 100
           
           response_times = [r["response_time"] for r in results if r["success"]]
           avg_response_time = sum(response_times) / len(response_times) if response_times else float('inf')
           
           # Acceptance criteria
           if success_rate < 95:
               return False, f"Load test failed: {success_rate:.1f}% success rate (minimum: 95%)"
           
           if avg_response_time > 20:
               return False, f"Load test failed: {avg_response_time:.2f}s average response time (maximum: 20s)"
           
           return True, f"Load capacity validated - {success_rate:.1f}% success rate, {avg_response_time:.2f}s avg response time"
           
       except Exception as e:
           return False, f"Load capacity validation error: {e}"
   
   async def validate_backup_systems(self) -> Tuple[bool, str]:
       """Validate backup and recovery systems"""
       try:
           backup_checks = []
           
           # Check if data directory exists and has content
           data_dir = Path("data")
           if not data_dir.exists():
               backup_checks.append("Data directory not found")
           else:
               # Check for vector database files
               vector_db_files = list(data_dir.rglob("*.sqlite*")) + list(data_dir.rglob("chroma*"))
               if not vector_db_files:
                   backup_checks.append("No vector database files found for backup")
           
           # Check if backup scripts exist
           backup_script = Path("scripts/backup.sh")
           if not backup_script.exists():
               backup_checks.append("Backup script not found")
           
           # Test backup functionality (simplified)
           try:
               # Create a test backup
               import shutil
               import tempfile
               
               with tempfile.TemporaryDirectory() as temp_dir:
                   if data_dir.exists():
                       test_backup_path = Path(temp_dir) / "test_backup"
                       shutil.copytree(data_dir, test_backup_path)
                       
                       # Verify backup was created
                       if not test_backup_path.exists():
                           backup_checks.append("Backup creation test failed")
           except Exception:
               backup_checks.append("Backup functionality test failed")
           
           if backup_checks:
               return False, f"Backup issues: {', '.join(backup_checks)}"
           
           return True, "Backup systems validated"
           
       except Exception as e:
           return False, f"Backup validation error: {e}"
   
   def generate_final_report(self) -> bool:
       """Generate final validation report"""
       print("\n" + "=" * 60)
       print("📋 PRODUCTION VALIDATION REPORT")
       print("=" * 60)
       
       total_checks = len(self.results)
       passed_checks = sum(1 for _, success, _ in self.results if success)
       failed_checks = total_checks - passed_checks
       
       print(f"Total Validation Checks: {total_checks}")
       print(f"✅ Passed: {passed_checks}")
       print(f"❌ Failed: {failed_checks}")
       
       if failed_checks > 0:
           print(f"\n🚨 Failed Validations:")
           for name, success, details in self.results:
               if not success:
                   print(f"   • {name}: {details}")
       
       if self.critical_failures:
           print(f"\n⚠️  Critical Failures ({len(self.critical_failures)}):")
           for failure in self.critical_failures:
               print(f"   • {failure}")
       
       success_rate = (passed_checks / total_checks) * 100
       print(f"\n📈 Overall Success Rate: {success_rate:.1f}%")
       
       # Determine deployment readiness
       if success_rate == 100:
           print("\n🎉 PRODUCTION READY!")
           print("All validation checks passed. System is ready for production deployment.")
           deployment_ready = True
       elif success_rate >= 90 and not self.critical_failures:
           print("\n✅ PRODUCTION READY WITH WARNINGS")
           print("Minor issues detected but system is suitable for production.")
           deployment_ready = True
       else:
           print("\n❌ NOT PRODUCTION READY")
           print("Critical issues must be resolved before production deployment.")
           deployment_ready = False
       
       # Generate recommendations
       print(f"\n📝 Recommendations:")
       if failed_checks == 0:
           print("   • System is fully validated and production ready")
           print("   • Continue with deployment")
       else:
           if self.critical_failures:
               print("   • Resolve critical failures before proceeding")
           print("   • Address failed validation checks")
           print("   • Re-run validation after fixes")
           print("   • Consider gradual rollout with monitoring")
       
       print(f"\n🕒 Validation completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
       
       return deployment_ready

# Deployment helper functions
async def deploy_with_validation(environment: str = "production"):
   """Deploy system with comprehensive validation"""
   print(f"🚀 Deploying AI Agent to {environment} environment")
   print("=" * 50)
   
   # Step 1: Pre-deployment validation
   print("Phase 1: Pre-deployment validation")
   validator = ProductionValidator()
   
   # Run basic checks first
   basic_checks = [
       ("System Resources", validator.validate_system_resources),
       ("Docker Environment", validator.validate_docker_environment),
   ]
   
   for name, check_func in basic_checks:
       print(f"Checking {name}...")
       success, details = await check_func()
       if not success:
           print(f"❌ Pre-deployment check failed: {details}")
           return False
       print(f"✅ {details}")
   
   # Step 2: Start services
   print(f"\nPhase 2: Starting services for {environment}")
   try:
       if environment == "production":
           subprocess.run(["docker-compose", "-f", "docker-compose.yml", "up", "-d"], check=True)
       else:
           subprocess.run(["docker-compose", "-f", "docker-compose.dev.yml", "up", "-d"], check=True)
       
       # Wait for services to start
       print("Waiting for services to start...")
       await asyncio.sleep(30)
       
   except subprocess.CalledProcessError as e:
       print(f"❌ Failed to start services: {e}")
       return False
   
   # Step 3: Full validation
   print(f"\nPhase 3: Full deployment validation")
   deployment_ready = await validator.run_full_validation()
   
   if deployment_ready:
       print(f"\n🎉 Deployment to {environment} completed successfully!")
       return True
   else:
       print(f"\n❌ Deployment validation failed. Rolling back...")
       # Rollback logic would go here
       return False

# CLI interface
if __name__ == "__main__":
   import argparse
   
   parser = argparse.ArgumentParser(description="Production validation and deployment for AI Agent")
   parser.add_argument("--validate-only", action="store_true", help="Run validation only")
   parser.add_argument("--deploy", choices=["development", "production"], help="Deploy to environment")
   parser.add_argument("--base-url", default="http://localhost:8000", help="Base URL for API testing")
   
   args = parser.parse_args()
   
   async def main():
       if args.deploy:
           success = await deploy_with_validation(args.deploy)
           sys.exit(0 if success else 1)
       else:
           validator = ProductionValidator(args.base_url)
           success = await validator.run_full_validation()
           sys.exit(0 if success else 1)
   
   asyncio.run(main())