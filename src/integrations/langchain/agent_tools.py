# src/integrations/langchain/agent_tools.py
from langchain.tools import BaseTool, StructuredTool
from langchain.pydantic_v1 import BaseModel, Field
from typing import Optional, Type, List, Dict, Any
import json
import math
import re
import requests
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# Input schemas for our tools
class CalculatorInput(BaseModel):
    """Input schema for calculator tool"""
    expression: str = Field(description="Mathematical expression to calculate (e.g., '2 + 2', 'sqrt(16)', 'sin(3.14/2)')")

class SearchInput(BaseModel):
    """Input schema for search tool"""
    query: str = Field(description="Search query to find information")
    max_results: int = Field(default=5, description="Maximum number of results to return")

class EmailInput(BaseModel):
    """Input schema for email tool"""
    recipient: str = Field(description="Email recipient address")
    subject: str = Field(description="Email subject line")
    message: str = Field(description="Email message content")

class WeatherInput(BaseModel):
    """Input schema for weather tool"""
    location: str = Field(description="City or location name (e.g., 'New York', 'London')")

class MemoryInput(BaseModel):
    """Input schema for memory search tool"""
    query: str = Field(description="Query to search in agent's memory")
    memory_type: Optional[str] = Field(default=None, description="Type of memory to search (conversation, fact, etc.)")

# Tool Implementations
class AdvancedCalculatorTool(BaseTool):
    """Enhanced calculator with support for complex mathematical operations"""
    
    name = "calculator"
    description = """Performs mathematical calculations including:
    - Basic arithmetic: +, -, *, /, ** (power)
    - Functions: sqrt, sin, cos, tan, log, log10, exp, abs
    - Constants: pi, e
    - Example: calculator('2 + 2') or calculator('sqrt(16) + sin(pi/2)')"""
    args_schema: Type[BaseModel] = CalculatorInput
    
    def _run(self, expression: str) -> str:
        """Execute mathematical calculation safely"""
        try:
            # Clean and prepare the expression
            expression = expression.replace('^', '**')  # Convert ^ to **
            
            # Define safe functions and constants
            safe_dict = {
                "__builtins__": {},
                # Math functions
                "sqrt": math.sqrt,
                "sin": math.sin,
                "cos": math.cos,
                "tan": math.tan,
                "log": math.log,
                "log10": math.log10,
                "exp": math.exp,
                "abs": abs,
                "round": round,
                "min": min,
                "max": max,
                "sum": sum,
                "pow": pow,
                # Constants
                "pi": math.pi,
                "e": math.e
            }
            
            # Evaluate the expression safely
            result = eval(expression, safe_dict)
            
            # Format the result nicely
            if isinstance(result, float):
                if result.is_integer():
                    result = int(result)
                else:
                    result = round(result, 6)  # Round to 6 decimal places
            
            return f"The result of '{expression}' is {result}"
            
        except ZeroDivisionError:
            return "Error: Division by zero is not allowed"
        except OverflowError:
            return "Error: The result is too large to calculate"
        except ValueError as e:
            return f"Error: Invalid mathematical operation - {str(e)}"
        except Exception as e:
            return f"Calculation error: {str(e)}. Please check your expression syntax."
    
    async def _arun(self, expression: str) -> str:
        """Async version of the tool"""
        return self._run(expression)

class WebSearchTool(BaseTool):
    """Web search tool for finding current information"""
    
    name = "web_search"
    description = """Searches the web for current information about any topic.
    Use this for recent events, current data, news, or information not in training data.
    Example: web_search('latest news about AI') or web_search('current weather in Tokyo')"""
    args_schema: Type[BaseModel] = SearchInput
    
    def _run(self, query: str, max_results: int = 5) -> str:
        """Execute web search (simulated for tutorial)"""
        try:
            # In a real implementation, you would integrate with:
            # - DuckDuckGo API (free)
            # - Google Custom Search API
            # - Serper API
            # - Tavily AI Search API
            
            # For this tutorial, we'll simulate realistic search results
            search_results = self._simulate_search_results(query, max_results)
            
            if not search_results:
                return f"No search results found for '{query}'"
            
            # Format results for the agent
            formatted_results = [f"**Search Results for '{query}':**\n"]
            
            for i, result in enumerate(search_results, 1):
                formatted_results.append(
                    f"{i}. **{result['title']}**\n"
                    f"   {result['snippet']}\n"
                    f"   Source: {result['url']}\n"
                )
            
            return "\n".join(formatted_results)
            
        except Exception as e:
            return f"Search error: {str(e)}"
    
    def _simulate_search_results(self, query: str, max_results: int) -> List[Dict[str, str]]:
        """Simulate web search results for tutorial purposes"""
        # Create realistic search results based on query patterns
        results = []
        
        if "weather" in query.lower():
            results.append({
                "title": f"Current Weather for {query.replace('weather', '').strip()}",
                "snippet": "Get current weather conditions, temperature, humidity, and forecast. Updated every 15 minutes.",
                "url": "https://weather.com/current-conditions"
            })
        elif "news" in query.lower() or "latest" in query.lower():
            results.append({
                "title": f"Latest News: {query}",
                "snippet": "Breaking news and recent developments. Stay informed with real-time updates.",
                "url": "https://news.example.com/latest"
            })
        elif "price" in query.lower() or "cost" in query.lower():
            results.append({
                "title": f"Current Prices: {query}",
                "snippet": "Compare prices from multiple sources. Find the best deals and current market rates.",
                "url": "https://price-comparison.com"
            })
        else:
            # Generic search results
            results.extend([
                {
                    "title": f"Complete Guide to {query}",
                    "snippet": f"Comprehensive information about {query}. Learn everything you need to know with expert insights and detailed explanations.",
                    "url": f"https://guide.example.com/{query.replace(' ', '-')}"
                },
                {
                    "title": f"{query} - Wikipedia",
                    "snippet": f"Encyclopedia article about {query}. Free, reliable information from the world's largest encyclopedia.",
                    "url": f"https://wikipedia.org/wiki/{query.replace(' ', '_')}"
                }
            ])
        
        return results[:max_results]
    
    async def _arun(self, query: str, max_results: int = 5) -> str:
        return self._run(query, max_results)

class EmailTool(BaseTool):
    """Tool for sending emails"""
    
    name = "send_email"
    description = """Sends an email to a specified recipient.
    Use when user asks to send an email, notify someone, or communicate via email.
    Example: send_email('john@example.com', 'Meeting Reminder', 'Don't forget about our 2pm meeting')"""
    args_schema: Type[BaseModel] = EmailInput
    
    def _run(self, recipient: str, subject: str, message: str) -> str:
        """Send email (simulated for tutorial)"""
        try:
            # Validate email format
            if not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', recipient):
                return f"Error: '{recipient}' is not a valid email address"
            
            # In a real implementation, integrate with:
            # - SendGrid API
            # - AWS SES
            # - Gmail API
            # - Mailgun
            # - SMTP server
            
            # Simulate email sending
            email_id = f"email_{int(datetime.now().timestamp())}"
            
            # Log the email for debugging
            logger.info(f"Email sent - ID: {email_id}, To: {recipient}, Subject: {subject}")
            
            return (f"✅ Email sent successfully!\n"
                   f"To: {recipient}\n"
                   f"Subject: {subject}\n"
                   f"Message: {message[:100]}{'...' if len(message) > 100 else ''}\n"
                   f"Email ID: {email_id}")
            
        except Exception as e:
            return f"Email sending failed: {str(e)}"
    
    async def _arun(self, recipient: str, subject: str, message: str) -> str:
        return self._run(recipient, subject, message)

class WeatherTool(BaseTool):
    """Tool for getting weather information"""
    
    name = "get_weather"
    description = """Gets current weather information for any location.
    Use when user asks about weather conditions, temperature, or forecast.
    Example: get_weather('New York') or get_weather('London, UK')"""
    args_schema: Type[BaseModel] = WeatherInput
    
    def _run(self, location: str) -> str:
        """Get weather information (simulated for tutorial)"""
        try:
            # In a real implementation, integrate with:
            # - OpenWeatherMap API
            # - WeatherAPI
            # - AccuWeather API
            # - National Weather Service API
            
            # Simulate realistic weather data
            import random
            
            weather_conditions = [
                "Clear", "Partly Cloudy", "Cloudy", "Light Rain", 
                "Heavy Rain", "Thunderstorms", "Snow", "Fog"
            ]
            
            # Generate realistic weather based on location patterns
            temp_ranges = {
                "new york": (15, 25),
                "london": (8, 18),
                "tokyo": (18, 28),
                "sydney": (20, 30),
                "moscow": (-5, 10),
                "miami": (25, 35)
            }
            
            location_lower = location.lower()
            temp_range = (10, 25)  # Default range
            
            for city, range_val in temp_ranges.items():
                if city in location_lower:
                    temp_range = range_val
                    break
            
            temperature = random.randint(temp_range[0], temp_range[1])
            condition = random.choice(weather_conditions)
            humidity = random.randint(30, 90)
            wind_speed = random.randint(5, 25)
            
            weather_info = (
                f"🌤️ **Weather in {location}**\n"
                f"Temperature: {temperature}°C ({int(temperature * 9/5 + 32)}°F)\n"
                f"Condition: {condition}\n"
                f"Humidity: {humidity}%\n"
                f"Wind Speed: {wind_speed} km/h\n"
                f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
            
            return weather_info
            
        except Exception as e:
            return f"Weather lookup failed: {str(e)}"
    
    async def _arun(self, location: str) -> str:
        return self._run(location)

class MemorySearchTool(BaseTool):
    """Tool for searching agent's memory"""
    
    name = "memory_search"
    description = """Searches the agent's memory for relevant past conversations or information.
    Use when you need to recall previous interactions, user preferences, or stored information.
    Example: memory_search('user preferences') or memory_search('previous questions about Python')"""
    args_schema: Type[BaseModel] = MemoryInput
    
    def __init__(self, vector_store=None):
        super().__init__()
        self.vector_store = vector_store
    
    def _run(self, query: str, memory_type: Optional[str] = None) -> str:
        """Search agent memory"""
        try:
            if not self.vector_store:
                return "Memory search is not available - vector store not configured"
            
            # For tutorial purposes, simulate memory search results
            # In real implementation, this would use: self.vector_store.similarity_search(query)
            
            simulated_memories = [
                {
                    "content": f"Previous conversation: User asked about {query} and I provided helpful information.",
                    "timestamp": "2024-01-15 10:30:00",
                    "relevance": 0.85,
                    "type": "conversation"
                },
                {
                    "content": f"User preference noted: Shows interest in {query} topics.",
                    "timestamp": "2024-01-14 15:45:00", 
                    "relevance": 0.72,
                    "type": "preference"
                }
            ]
            
            if not simulated_memories:
                return f"No relevant memories found for '{query}'"
            
            # Format memories for agent
            formatted_memories = [f"**Memory Search Results for '{query}':**\n"]
            
            for i, memory in enumerate(simulated_memories, 1):
                formatted_memories.append(
                    f"{i}. **{memory['type'].title()} Memory** (Relevance: {memory['relevance']:.2f})\n"
                    f"   {memory['content']}\n"
                    f"   From: {memory['timestamp']}\n"
                )
            
            return "\n".join(formatted_memories)
            
        except Exception as e:
            return f"Memory search error: {str(e)}"
    
    async def _arun(self, query: str, memory_type: Optional[str] = None) -> str:
        return self._run(query, memory_type)

def create_agent_toolset(vector_store=None) -> List[BaseTool]:
    """Create the complete set of tools for our agent"""
    tools = [
        AdvancedCalculatorTool(),
        WebSearchTool(),
        EmailTool(),
        WeatherTool()
    ]
    
    # Add memory search if vector store is available
    if vector_store:
        tools.append(MemorySearchTool(vector_store))
    
    logger.info(f"Created agent toolset with {len(tools)} tools")
    return tools

