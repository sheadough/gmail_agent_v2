# src/utils/config.py - Configuration Management System
from typing import Dict, Any, Optional, Union
import yaml
import os
from pathlib import Path
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)

@dataclass
class AgentConfig:
    """Configuration container for agent setup"""
    
    def __init__(self, config: Union[Dict[str, Any], str, Path] = None):
        if isinstance(config, (str, Path)):
            # Load from file
            self.data = self._load_config_file(config)
        elif isinstance(config, dict):
            self.data = config
        else:
            self.data = {}
            
        # Set defaults
        self._apply_defaults()
    
    def _load_config_file(self, config_path: Union[str, Path]) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        config_path = Path(config_path)
        
        if not config_path.exists():
            logger.error(f"Config file not found: {config_path}")
            return {}
            
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logger.error(f"Failed to load config file {config_path}: {e}")
            return {}
    
    def _apply_defaults(self):
        """Apply default configuration values"""
        defaults = {
            'agent': {
                'name': 'AI Agent',
                'version': '1.0.0'
            },
            'components': {},
            'integrations': {},
            'interfaces': ['api'],
            'features': []
        }
        
        # Deep merge defaults with loaded config
        self.data = self._deep_merge(defaults, self.data)
    
    def _deep_merge(self, base: Dict, overlay: Dict) -> Dict:
        """Deep merge two dictionaries"""
        result = base.copy()
        for key, value in overlay.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result
    
    def get_components(self) -> Dict[str, Any]:
        """Get component configurations"""
        return self.data.get('components', {})
    
    def get_integrations(self) -> Dict[str, Any]:
        """Get integration configurations"""
        return self.data.get('integrations', {})
    
    def get_interfaces(self) -> list:
        """Get enabled interfaces"""
        return self.data.get('interfaces', [])
    
    def get_features(self) -> list:
        """Get enabled features"""
        return self.data.get('features', [])
    
    def get(self, key: str, default=None):
        """Get configuration value by key"""
        return self.data.get(key, default)

class TemplateManager:
    """Manages agent templates and configurations"""
    
    def __init__(self, templates_dir: str = "templates"):
        self.templates_dir = Path(templates_dir)
    
    def list_templates(self) -> Dict[str, Dict[str, Any]]:
        """List available agent templates"""
        templates = {}
        
        if not self.templates_dir.exists():
            return templates
            
        for template_dir in self.templates_dir.iterdir():
            if template_dir.is_dir():
                template_info = self._load_template_info(template_dir)
                if template_info:
                    templates[template_dir.name] = template_info
                    
        return templates
    
    def _load_template_info(self, template_dir: Path) -> Optional[Dict[str, Any]]:
        """Load template information"""
        config_file = template_dir / "config.yaml"
        if not config_file.exists():
            return None
            
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
                
            # Add template metadata
            readme_file = template_dir / "README.md"
            if readme_file.exists():
                with open(readme_file, 'r') as f:
                    config['description_long'] = f.read()
                    
            return config
            
        except Exception as e:
            logger.error(f"Failed to load template {template_dir.name}: {e}")
            return None
    
    def load_template_config(self, template_name: str) -> Optional[AgentConfig]:
        """Load configuration for specific template"""
        template_dir = self.templates_dir / template_name
        config_file = template_dir / "config.yaml"
        
        if not config_file.exists():
            logger.error(f"Template config not found: {config_file}")
            return None
            
        return AgentConfig(config_file)
    
    def create_agent_from_template(self, template_name: str, customizations: Dict[str, Any] = None):
        """Create agent instance from template"""
        from ..core.agent import Agent
        
        config = self.load_template_config(template_name)
        if not config:
            raise ValueError(f"Template not found: {template_name}")
        
        # Apply customizations
        if customizations:
            config.data = config._deep_merge(config.data, customizations)
            
        return Agent(config=config)

# Template configuration examples embedded in code for reference
TEMPLATE_CONFIGS = {
    'basic_chatbot': {
        'agent': {
            'name': 'Basic Chatbot',
            'version': '1.0.0'
        },
        'components': {
            'perception': 'basic',
            'memory': 'local', 
            'actions': 'basic'
        },
        'integrations': {
            'openai': {
                'enabled': True,
                'model': 'gpt-3.5-turbo'
            }
        },
        'interfaces': ['api', 'cli'],
        'features': ['logging']
    },
    
    'rag_assistant': {
        'agent': {
            'name': 'RAG Assistant',
            'version': '1.0.0'
        },
        'components': {
            'perception': 'llm_enhanced',
            'memory': {
                'implementation': 'vector',
                'config': {
                    'vector_store_type': 'chroma',
                    'collection_name': 'rag_documents'
                }
            },
            'decision': 'llm_based',
            'actions': 'basic'
        },
        'integrations': {
            'openai': {
                'enabled': True,
                'model': 'gpt-4'
            },
            'langchain': {
                'enabled': False  # Using core components instead
            }
        },
        'interfaces': ['api', 'web'],
        'features': ['logging', 'metrics', 'document_upload']
    },
    
    'tool_agent': {
        'agent': {
            'name': 'Tool-Using Agent',
            'version': '1.0.0'  
        },
        'components': {
            'perception': 'llm_enhanced',
            'memory': 'vector',
            'decision': 'llm_based',
            'actions': 'tool_calling'
        },
        'integrations': {
            'openai': {
                'enabled': True,
                'model': 'gpt-4'
            },
            'langchain': {
                'enabled': True,
                'tools': ['web_search', 'calculator', 'email'],
                'fallback_to_core': True
            }
        },
        'interfaces': ['api', 'cli'],
        'features': ['logging', 'metrics', 'tool_monitoring']
    },
    
    'production_agent': {
        'agent': {
            'name': 'Production AI Agent',
            'version': '1.0.0'
        },
        'components': {
            'perception': 'llm_enhanced',
            'memory': {
                'implementation': 'vector',
                'config': {
                    'vector_store_type': 'chroma',
                    'persistent_storage': True,
                    'backup_enabled': True
                }
            },
            'decision': 'llm_based', 
            'actions': 'tool_calling',
            'learning': 'rl'
        },
        'integrations': {
            'openai': {
                'enabled': True,
                'model': 'gpt-4',
                'fallback_model': 'gpt-3.5-turbo'
            },
            'langchain': {
                'enabled': True,
                'tools': ['web_search', 'calculator', 'email', 'file_ops'],
                'rate_limiting': True
            }
        },
        'interfaces': ['api', 'web'],
        'features': ['logging', 'metrics', 'monitoring', 'health_checks', 'rate_limiting'],
        'deployment': {
            'nginx': True,
            'ssl': True,
            'monitoring': ['prometheus', 'grafana']
        }
    }
}

def create_template_files():
    """Utility to create template files from embedded configs"""
    templates_dir = Path("templates")
    templates_dir.mkdir(exist_ok=True)
    
    for template_name, config in TEMPLATE_CONFIGS.items():
        template_dir = templates_dir / template_name
        template_dir.mkdir(exist_ok=True)
        
        # Write config.yaml
        config_file = template_dir / "config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
            
        # Write README.md
        readme_file = template_dir / "README.md"
        with open(readme_file, 'w') as f:
            f.write(f"# {config['agent']['name']}\n\n")
            f.write(f"Version: {config['agent']['version']}\n\n")
            f.write("## Components\n")
            for component, impl in config['components'].items():
                f.write(f"- **{component}**: {impl}\n")
            f.write("\n## Usage\n")
            f.write(f"```bash\n./create_agent.py {template_name} my_agent\n```\n")
    
    print(f"Created {len(TEMPLATE_CONFIGS)} template configurations in {templates_dir}")

if __name__ == "__main__":
    create_template_files()