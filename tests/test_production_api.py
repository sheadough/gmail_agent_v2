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