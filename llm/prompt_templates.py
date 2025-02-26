# llm/prompt_templates.py

from typing import Dict, Any, Optional, List, Union
from string import Template
import json
import yaml
from pathlib import Path
import logging
import pkg_resources
import os

class PromptError(Exception):
    """Raised when prompt operations fail."""
    pass

class PromptTemplate:
    """Manages prompt templates with variable substitution."""
    
    def __init__(self, template: Union[str, Dict[str, str]]):
        if isinstance(template, dict):
            self.system = Template(template.get('system', ''))
            self.query = Template(template.get('query', ''))
        else:
            self.system = Template('')
            self.query = Template(template)
        
    def format(self, **kwargs) -> Union[str, Dict[str, str]]:
        """Format template with provided variables."""
        try:
            if self.system.template:
                return {
                    'system': self.system.safe_substitute(**kwargs),
                    'query': self.query.safe_substitute(**kwargs)
                }
            return self.query.safe_substitute(**kwargs)
        except KeyError as e:
            raise PromptError(f"Missing required variable: {str(e)}")
        except Exception as e:
            raise PromptError(f"Template formatting error: {str(e)}")

class RAGPromptManager:
    """Manages RAG-specific prompt templates."""
    
    DEFAULT_TEMPLATES = {
        'basic': 'basic_rag',
        'conversational': 'conversational_rag',
        'analytical': 'analytical_rag',
        'multi_document': 'multi_document_rag',
        'code': 'code_rag',
        'summary': 'summary_rag',
        'qa': 'qa_rag'
    }
    
    def __init__(self):
        self.templates = self._load_default_templates()
        
    def _load_default_templates(self) -> Dict[str, PromptTemplate]:
        """Load default RAG templates."""
        try:
            # Get the current file's directory
            current_dir = Path(__file__).parent
            template_path = current_dir / 'templates' / 'rag_templates.yaml'
            
            with open(template_path, 'r') as f:
                templates_data = yaml.safe_load(f)
                
            return {
                name: PromptTemplate(template)
                for name, template in templates_data.items()
            }
        except Exception as e:
            logging.error(f"Failed to load default RAG templates: {str(e)}")
            return {}
    
    def get_template(self, template_type: str = 'basic') -> PromptTemplate:
        """Get a RAG template by type."""
        template_name = self.DEFAULT_TEMPLATES.get(template_type)
        if not template_name or template_name not in self.templates:
            raise PromptError(f"Template type not found: {template_type}")
        return self.templates[template_name]
    
    def format_prompt(
        self,
        template_type: str = 'basic',
        **kwargs
    ) -> Dict[str, str]:
        """Format a RAG template with variables."""
        template = self.get_template(template_type)
        return template.format(**kwargs)

class PromptManager:
    """Manages a collection of prompt templates."""
    
    def __init__(
        self,
        templates_dir: Optional[Path] = None,
        load_defaults: bool = True
    ):
        self.templates: Dict[str, PromptTemplate] = {}
        self.rag = RAGPromptManager() if load_defaults else None
        if templates_dir:
            self.load_templates(templates_dir)

    def load_templates(self, templates_dir: Path):
        """Load templates from YAML files in directory."""
        for template_file in templates_dir.glob("*.yaml"):
            try:
                with open(template_file) as f:
                    templates_data = yaml.safe_load(f)
                    
                for template_name, template_text in templates_data.items():
                    self.templates[template_name] = PromptTemplate(template_text)
                    
            except Exception as e:
                logging.error(f"Failed to load template file {template_file}: {str(e)}")

    def get_template(self, template_name: str) -> PromptTemplate:
        """Get a prompt template by name."""
        if template_name not in self.templates:
            raise PromptError(f"Template not found: {template_name}")
        return self.templates[template_name]

    def format_prompt(
        self,
        template_name: str,
        rag_type: Optional[str] = None,
        **kwargs
    ) -> Union[str, Dict[str, str]]:
        """Format a specific template with variables."""
        if rag_type and self.rag:
            return self.rag.format_prompt(rag_type, **kwargs)
        template = self.get_template(template_name)
        return template.format(**kwargs)

    def list_rag_templates(self) -> List[str]:
        """List available RAG template types."""
        if self.rag:
            return list(self.rag.DEFAULT_TEMPLATES.keys())
        return []

    def get_rag_template(self, template_type: str) -> Optional[PromptTemplate]:
        """Get a specific RAG template."""
        if self.rag:
            return self.rag.get_template(template_type)
        return None