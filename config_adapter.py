# config_adapter.py
class ConfigAdapter:
    """
    Adapter class to provide attribute access to Flask config
    """
    
    def __init__(self, config):
        """
        Initialize the config adapter
        
        Args:
            config: Flask config dictionary
        """
        self.config = config
    
    def __getattr__(self, name):
        """
        Get attribute from config
        
        Args:
            name: Attribute name
            
        Returns:
            Attribute value or None if not found
        """
        if name in self.config:
            return self.config[name]
        elif name.upper() in self.config:
            return self.config[name.upper()]
        else:
            return None