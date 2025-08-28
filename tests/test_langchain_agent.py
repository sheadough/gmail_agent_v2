# examples/test_langchain_agent.py
import asyncio
import sys
from pathlib import Path

# Add src to Python path
sys.path.append(str(Path(__file__).parent.parent / "src"))
from src.integrations.langchain.integrated_agent import IntegratedAgent
from src.integrations.langchain.agent_executor import LangChainAgent
from src.integrations.databases.vector_store import create_vector_store
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_langchain_agent():
    """Test the LangChain agent with various scenarios"""
    print("🤖 Testing LangChain Agent Integration")
    print("=" * 50)
    
    # Create vector store (for memory functionality)
    vector_store = create_vector_store("test_agent_memory")
    
    # Initialize agent
    agent = LangChain
    
    # Test scenarios
    test_cases = [
        {
            "name": "Simple Greeting",
            "message": "Hello! Can you introduce yourself?",
            "expected_tools": []
        },
        {
            "name": "Mathematical Calculation", 
            "message": "What's the square root of 144 plus 25% of 80?",
            "expected_tools": ["calculator"]
        },
        {
            "name": "Web Search Request",
            "message": "What's the latest news about artificial intelligence?",
            "expected_tools": ["web_search"]
        },
        {
            "name": "Weather Inquiry",
            "message": "What's the current weather in New York City?",
            "expected_tools": ["get_weather"]
        },
        {
            "name": "Email Request",
            "message": "Send an email to team@company.com with subject 'Meeting Update' saying the meeting is moved to 3 PM",
            "expected_tools": ["send_email"]
        },
        {
            "name": "Multi-tool Complex Task",
            "message": "Calculate 15% of 500, then search for information about that number in mathematics, and tell me the weather in London",
            "expected_tools": ["calculator", "web_search", "get_weather"]
        }
    ]
    
    results = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📋 Test {i}: {test_case['name']}")
        print(f"User: {test_case['message']}")
        
        try:
            # Process the message
            response = await agent.process_message(test_case['message'], f"test_user_{i}")
            
            # Display results
            print(f"Agent: {response['response']}")
            print(f"Tools used: {response['tools_used']}")
            print(f"Processing time: {response['processing_time']:.2f}s")
            print(f"Success: {'✅' if response['success'] else '❌'}")
            
            # Check if expected tools were used
            expected_tools = set(test_case['expected_tools'])
            actual_tools = set(response['tools_used'])
            
            if expected_tools and expected_tools.issubset(actual_tools):
                print("✅ Expected tools were used correctly")
            elif expected_tools and not expected_tools.intersection(actual_tools):
                print("⚠️ Expected tools were not used")
            else:
                print("ℹ️ Tool usage as expected")
            
            # Show tool execution details
            if response['tool_details']:
                print("🔧 Tool execution details:")
                for detail in response['tool_details']:
                    print(f"   - {detail['tool']}: {detail.get('duration', 0):.2f}s")
            
            results.append({
                'test': test_case['name'],
                'success': response['success'],
                'tools_used': response['tools_used'],
                'processing_time': response['processing_time']
            })
            
        except Exception as e:
            print(f"❌ Test failed with error: {e}")

            results.append({
                        'test': test_case['name'],
                        'success': False,
                        'error': str(e),
                        'tools_used': [],
                        'processing_time': 0
                    })
                
                # Wait between tests to avoid rate limiting
                await asyncio.sleep(1)
            
            # Display summary
            print("\n" + "=" * 50)
            print("📊 TEST SUMMARY")
            print("=" * 50)
            
            successful_tests = sum(1 for result in results if result['success'])
            total_tests = len(results)
            
            print(f"Tests completed: {total_tests}")
            print(f"Successful: {successful_tests}")
            print(f"Failed: {total_tests - successful_tests}")
            print(f"Success rate: {(successful_tests / total_tests) * 100:.1f}%")
            
            # Tool usage statistics
            all_tools_used = []
            for result in results:
                all_tools_used.extend(result.get('tools_used', []))
            
            if all_tools_used:
                from collections import Counter
                tool_counts = Counter(all_tools_used)
                print(f"\n🔧 Tool Usage Statistics:")
                for tool, count in tool_counts.most_common():
                    print(f"   - {tool}: {count} times")
            
            # Average processing time
            processing_times = [r['processing_time'] for r in results if r['success']]
            if processing_times:
                avg_time = sum(processing_times) / len(processing_times)
                print(f"\n⏱️ Average processing time: {avg_time:.2f}s")
            
            # Agent statistics
            print(f"\n📈 Agent Statistics:")
            stats = agent.get_agent_stats()
            print(f"   - Total interactions: {stats['total_interactions']}")
            print(f"   - Success rate: {stats['success_rate']:.1f}%")
            print(f"   - Available tools: {len(stats['available_tools'])}")
            
            return results

            async def interactive_test():
            """Interactive test mode for manual testing"""
            print("\n🎮 Interactive Test Mode")
            print("Type 'quit' to exit, 'help' for agent capabilities")
            print("-" * 40)
            
            # Initialize agent
            vector_store = create_vector_store("interactive_test_memory")
            agent = LangChainAgent(vector_store)
            
            while True:
                try:
                    user_input = input("\nYou: ").strip()
                    
                    if user_input.lower() == 'quit':
                        break
                    elif user_input.lower() == 'help':
                        help_text = await agent.explain_capabilities()
                        print(f"\nAgent: {help_text}")
                        continue
                    elif not user_input:
                        continue
                    
                    # Process the message
                    response = await agent.process_message(user_input, "interactive_user")
                    
                    print(f"\nAgent: {response['response']}")
                    
                    if response['tools_used']:
                        print(f"🔧 Tools used: {', '.join(response['tools_used'])}")
                    
                    print(f"⏱️ Response time: {response['processing_time']:.2f}s")
                    
                except KeyboardInterrupt:
                    print("\n\nGoodbye!")
                    break
                except Exception as e:
                    print(f"\n❌ Error: {e}")

            if __name__ == "__main__":
            print("Choose test mode:")
            print("1. Automated tests")
            print("2. Interactive mode")
            
            choice = input("Enter choice (1 or 2): ").strip()
            
            if choice == "1":
                asyncio.run(test_langchain_agent())
            elif choice == "2":
                asyncio.run(interactive_test())
            else:
                print("Invalid choice")