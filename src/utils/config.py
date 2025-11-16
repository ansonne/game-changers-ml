import os
import yaml

from typing import Dict, Any

class Config:
    """Singleton configuration manager"""

    _instance = None
    _config = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._load_config()
        return cls._instance
    
    @classmethod
    def _load_config(cls):
        config_path = os.path.join(os.path.dirname(__file__), '../../config.yaml')
        with open(config_path, 'r', encoding='utf-8') as file:
            cls._config = yaml.safe_load(file)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key"""
        
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value