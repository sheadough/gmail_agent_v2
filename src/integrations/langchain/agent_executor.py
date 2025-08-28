# src/integrations/langchain/agent_executor.py
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langchain.callbacks.base import BaseCallbackHandler
from typing import List, Dict, Any, Optional
import asyncio
import json
import time
from datetime import datetime
from config.settings import settings
from src.integrations.langchain.agent_tools import create_agent_toolset
import logging

logger = logging.getLogger(__name__)

class AgentCallbackHandler(BaseCallbackHandler):
    """Custom callback handler to track agent execution"""
    
    def __init__(self):
        self.execution_log = []
        self.current_step = {}
    
    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs) -> None:
        """Called when a tool starts executing"""
        tool_name = serialized.get("name", "Unknown")
        self.current_step = {
            "tool": tool_name,
            "input": input_str,
            "start_time": time.time()
        }
        logger.debug(f"Tool started: {tool_name} with input: {input_str}")
    
    def on_tool_end(self, output: str, **kwargs) -> None:
        """Called when a tool finishes executing"""
        if self.current_step:
            self.current_step.update({
                "output": output,
                "end_time": time.time(),
                "duration": time.time() - self.current_step.get("start_time", time.time())
            })
            self.execution_log.append(self.current_step.copy())
            logger.debug(f"Tool completed: {self.current_step['tool']} in {self.current_step['duration']:.2f}s")
        self.current_step = {}
    
    def on_tool_error(self, error: Exception, **kwargs) -> None:
        """Called when a tool encounters an error"""
        if self.current_step:
            self.current_step.update({
                "error": str(error),
                "end_time": time.time(),
                "duration": time.time() - self.current_step.get("start_time", time.time())
            })
            self.execution_log.append(self.current_step.copy())
            logger.error(f"Tool error in {self.current_step['tool']}: {error}")
        self.current_step = {}

class LangChainAgent:
    """Advanced LangChain agent with tool orchestration"""
    
    def __init__(self, vector_store=None, custom_tools=None):
        # Initialize LLM
        self.llm = ChatOpenAI(
            model=settings.openai_model,
            temperature=settings.openai_temperature,
            openai_api_key=settings.openai_api_key
        )
        
        # Initialize tools
        self.tools = custom_tools or create_agent_toolset(vector_store)
        self.vector_store = vector_store
        
        # Initialize memory
        self.memory = ConversationBufferWindowMemory(
            k=settings.max_conversation_turns,
            memory_key="chat_history",
            return_messages=True
        )
        
        # Initialize callback handler
        self.callback_handler = AgentCallbackHandler()
        
        # Create agent executor
        self.agent_executor = self._create_agent_executor()
        
        # Statistics
        self.stats = {
            "total_interactions": 0,
            "successful_interactions": 0,
            "tool_usage": {},
            "average_response_time": 0.0,
            "error_count": 0
        }
        
        logger.info(f"LangChain agent initialized with {len(self.tools)} tools")
    
    def _create_agent_executor(self) -> AgentExecutor:
        """Create the agent executor with custom prompt"""
        
        # Create a comprehensive system prompt
        system_prompt = """You are a helpful and intelligent AI assistant with access to various tools.

**Your capabilities:**
- 🧮 **Calculator**: Perform mathematical calculations and solve equations
- 🔍 **Web Search**: Find current information, news, and real-time data  
- 📧 **Email**: Send emails to specified recipients
- 🌤️ **Weather**: Get current weather information for any location
- 🧠 **Memory Search**: Recall information from previous conversations

**Guidelines for effective tool usage:**

1. **When to use tools:**
   - Use calculator for ANY mathematical operations, even simple ones
   - Use web search for current events, recent information, or data not in your training
   - Use email when asked to send messages or notifications
   - Use weather tool for any weather-related questions
   - Use memory search to find relevant past conversations

2. **How to use tools effectively:**
   - Always explain why you're using a specific tool
   - Use clear, specific inputs to tools
   - Interpret and summarize tool outputs for the user
   - If a tool fails, try alternative approaches

3. **Communication style:**
   - Be conversational and friendly
   - Explain your reasoning process
   - Ask for clarification when needed
   - Provide comprehensive, helpful responses

**Current context:**
- Date: {current_date}
- Time: {current_time}

Remember: You have access to these tools for a reason - use them when they would provide more accurate, current, or helpful information than your base knowledge."""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        # Create the agent
        agent = create_openai_functions_agent(self.llm, self.tools, prompt)
        
        # Create executor with configuration
        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            memory=self.memory,
            callbacks=[self.callback_handler],
            verbose=settings.debug,
            max_iterations=10,
            max_execution_time=120,  # 2 minute timeout
            handle_parsing_errors=True,
            return_intermediate_steps=True
        )
    
    async def process_message(self, message: str, user_id: str = "default") -> Dict[str, Any]:
        """Process a user message through the agent"""
        start_time = time.time()
        self.stats["total_interactions"] += 1
        
        try:
            # Clear previous execution log
            self.callback_handler.execution_log = []
            
            # Prepare input with current context
            current_time = datetime.now()
            agent_input = {
                "input": message,
                "current_date": current_time.strftime("%Y-%m-%d"),
                "current_time": current_time.strftime("%H:%M:%S")
            }
            
            logger.info(f"Processing message from {user_id}: {message[:100]}...")
            
            # Execute the agent
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                self.agent_executor.invoke,
                agent_input
            )
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Update statistics
            self.stats["successful_interactions"] += 1
            self._update_stats(processing_time)
            
            # Extract tool usage information
            tools_used = self._extract_tools_used()
            for tool in tools_used:
                self.stats["tool_usage"][tool] = self.stats["tool_usage"].get(tool, 0) + 1
            
            return {
                "response": response["output"],
                "tools_used": tools_used,
                "tool_details": self.callback_handler.execution_log.copy(),
                "processing_time": processing_time,
                "intermediate_steps": len(response.get("intermediate_steps", [])),
                "success": True,
                "timestamp": datetime.now().isoformat(),
                "user_id": user_id
            }
            
        except Exception as e:
            self.stats["error_count"] += 1
            processing_time = time.time() - start_time
            
            logger.error(f"Agent execution failed: {e}")
            
            return {
                "response": "I apologize, but I encountered an error while processing your request. Please try rephrasing your question or ask something else.",
                "error": str(e),
                "tools_used": [],
                "tool_details": [],
                "processing_time": processing_time,
                "success": False,
                "timestamp": datetime.now().isoformat(),
                "user_id": user_id
            }
    
    def _extract_tools_used(self) -> List[str]:
        """Extract which tools were used from execution log"""
        return list(set(step["tool"] for step in self.callback_handler.execution_log))
    
    def _update_stats(self, processing_time: float):
        """Update processing statistics"""
        # Update average response time
        total = self.stats["successful_interactions"]
        current_avg = self.stats["average_response_time"]
        self.stats["average_response_time"] = (
            (current_avg * (total - 1) + processing_time) / total
        )
    
    def add_tool(self, tool) -> bool:
        """Add a new tool to the agent"""
        try:
            self.tools.append(tool)
            self.agent_executor = self._create_agent_executor()
            logger.info(f"Added tool: {tool.name}")
            return True
        except Exception as e:
            logger.error(f"Failed to add tool {tool.name}: {e}")
            return False
    
    def remove_tool(self, tool_name: str) -> bool:
        """Remove a tool from the agent"""
        try:
            self.tools = [tool for tool in self.tools if tool.name != tool_name]
            self.agent_executor = self._create_agent_executor()
            logger.info(f"Removed tool: {tool_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to remove tool {tool_name}: {e}")
            return False
    
    def get_conversation_history(self) -> List[BaseMessage]:
        """Get the conversation history"""
        return self.memory.chat_memory.messages
    
    def clear_memory(self):
        """Clear conversation memory"""
        self.memory.clear()
        logger.info("Cleared conversation memory")
    
    def get_agent_stats(self) -> Dict[str, Any]:
        """Get comprehensive agent statistics"""
        success_rate = 0.0
        if self.stats["total_interactions"] > 0:
            success_rate = (self.stats["successful_interactions"] / self.stats["total_interactions"]) * 100
        
        return {
            **self.stats,
            "available_tools": [tool.name for tool in self.tools],
            "memory_size": len(self.memory.chat_memory.messages),
            "success_rate": success_rate
        }
    
    async def explain_capabilities(self) -> str:
        """Generate a dynamic explanation of agent capabilities"""
        capabilities = []
        
        for tool in self.tools:
            tool_desc = f"**{tool.name}**: {tool.description.split('.')[0]}"
            capabilities.append(tool_desc)
        
        explanation = f"""I'm an AI assistant with access to {len(self.tools)} specialized tools:

{chr(10).join(capabilities)}

I can help you with calculations, finding current information, sending emails, checking weather, and more. Just ask me naturally, and I'll use the appropriate tools to help you!

**Examples of what you can ask:**
- "What's 15% of 2,847?" (I'll use the calculator)
- "What's the latest news about AI?" (I'll search the web)
- "Send an email to john@company.com about the meeting" (I'll send the email)
- "What's the weather in Tokyo?" (I'll get current weather)
- "What did we discuss about Python yesterday?" (I'll search my memory)

How can I help you today?"""
        
        return explanation