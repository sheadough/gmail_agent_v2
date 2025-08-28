# src/integrations/langchain/workflows.py
from typing import Dict, Any, List
from datetime import datetime
import asyncio
import logging
from src.integrations.langchain.agent_executor import LangChainAgent

logger = logging.getLogger(__name__)

class AgentWorkflows:
    """Pre-defined workflows for complex multi-step tasks"""
    
    def __init__(self, agent: LangChainAgent):
        self.agent = agent
    
    async def research_and_summarize_workflow(self, topic: str, user_id: str = "default") -> Dict[str, Any]:
        """Workflow: Research a topic and create a comprehensive summary"""
        workflow_steps = [
            f"Search for comprehensive information about {topic}",
            f"Search for recent news and developments about {topic}", 
            f"Based on the search results, create a detailed summary covering: 1) What is {topic}, 2) Current developments, 3) Key benefits/challenges, 4) Future outlook"
        ]
        
        results = []
        for step in workflow_steps:
            result = await self.agent.process_message(step, user_id)
            results.append(result)
            await asyncio.sleep(0.5)  # Brief pause between steps
        
        return {
            "workflow": "research_and_summarize",
            "topic": topic,
            "steps_completed": len(results),
            "final_response": results[-1]["response"] if results else "",
            "total_tools_used": list(set(tool for result in results for tool in result["tools_used"])),
            "total_processing_time": sum(result["processing_time"] for result in results)
        }
    
    async def daily_briefing_workflow(self, location: str = "New York", user_id: str = "default") -> Dict[str, Any]:
        """Workflow: Generate a daily briefing with weather and news"""
        workflow_steps = [
            f"What's the current weather in {location}?",
            "What are the top news stories today?",
            "Based on the weather and news information, create a brief daily summary"
        ]
        
        results = []
        for step in workflow_steps:
            result = await self.agent.process_message(step, user_id)
            results.append(result)
            await asyncio.sleep(0.5)
        
        return {
            "workflow": "daily_briefing",
            "location": location,
            "briefing": results[-1]["response"] if results else "",
            "tools_used": list(set(tool for result in results for tool in result["tools_used"])),
            "processing_time": sum(result["processing_time"] for result in results)
        }
    
    async def problem_solving_workflow(self, problem: str, user_id: str = "default") -> Dict[str, Any]:
        """Workflow: Systematic problem solving approach"""
        workflow_steps = [
            f"Break down this problem into smaller components: {problem}",
            f"For each component identified, search for relevant information and solutions",
            f"Calculate any numerical aspects if applicable for: {problem}",
            f"Synthesize all the information into a comprehensive solution for: {problem}"
        ]
        
        results = []
        for step in workflow_steps:
            result = await self.agent.process_message(step, user_id)
            results.append(result)
            await asyncio.sleep(0.5)
        
        return {
            "workflow": "problem_solving",
            "problem": problem,
            "solution": results[-1]["response"] if results else "",
            "analysis_steps": len(results),
            "tools_used": list(set(tool for result in results for tool in result["tools_used"])),
            "processing_time": sum(result["processing_time"] for result in results)
        }

# Example usage script
async def demo_workflows():
    """Demonstrate the agent workflows"""
    print("🔄 Demonstrating Agent Workflows")
    print("=" * 50)
    
    from src.integrations.databases.vector_store import create_vector_store
    
    # Initialize components
    vector_store = create_vector_store("workflow_demo")
    agent = LangChainAgent(vector_store)
    workflows = AgentWorkflows(agent)
    
    # Demo scenarios
    demos = [
        {
            "name": "Research Workflow",
            "function": workflows.research_and_summarize_workflow,
            "args": ("machine learning",),
            "description": "Research and summarize information about machine learning"
        },
        {
            "name": "Daily Briefing",
            "function": workflows.daily_briefing_workflow,
            "args": ("San Francisco",),
            "description": "Generate a daily briefing for San Francisco"
        },
        {
            "name": "Problem Solving",
            "function": workflows.problem_solving_workflow,
            "args": ("How to reduce energy consumption in a data center",),
            "description": "Systematically solve an energy efficiency problem"
        }
    ]
    
    for demo in demos:
        print(f"\n🎯 {demo['name']}: {demo['description']}")
        print("-" * 30)
        
        try:
            result = await demo['function'](*demo['args'])
            
            print(f"✅ Workflow completed in {result['processing_time']:.2f}s")
            print(f"🔧 Tools used: {', '.join(result['total_tools_used']) if 'total_tools_used' in result else ', '.join(result['tools_used'])}")
            print(f"📝 Result preview: {result.get('final_response', result.get('briefing', result.get('solution', '')))[:200]}...")
            
        except Exception as e:
            print(f"❌ Workflow failed: {e}")
        
        await asyncio.sleep(2)  # Pause between demos

if __name__ == "__main__":
    asyncio.run(demo_workflows())