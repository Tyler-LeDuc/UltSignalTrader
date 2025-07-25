"""Logging utilities for UltSignalTrader.

Provides a centralized logging configuration with both console
and rotating file handlers.
"""

import logging
import os
from logging.handlers import RotatingFileHandler
from typing import Optional


class Logger:
    """Custom logger with console and file output."""
    
    _loggers = {}
    
    @classmethod
    def get_logger(cls, name: str, config: Optional[dict] = None) -> logging.Logger:
        """Get or create a logger instance.
        
        Args:
            name: Logger name (usually __name__ of the module)
            config: Optional logging configuration dict
            
        Returns:
            Configured logger instance
        """
        if name in cls._loggers:
            return cls._loggers[name]
        
        # Create logger
        logger = logging.getLogger(name)
        
        # Use provided config or defaults
        if config is None:
            from ..config import config as app_config
            level = app_config.logging.level
            file_path = app_config.logging.file_path
            max_bytes = app_config.logging.max_bytes
            backup_count = app_config.logging.backup_count
        else:
            level = config.get('level', 'INFO')
            file_path = config.get('file_path', 'logs/ultsignaltrader.log')
            max_bytes = config.get('max_bytes', 10_485_760)
            backup_count = config.get('backup_count', 5)
        
        # Set level
        logger.setLevel(getattr(logging, level))
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        simple_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(simple_formatter)
        logger.addHandler(console_handler)
        
        # File handler with rotation
        if file_path:
            # Create log directory if it doesn't exist
            log_dir = os.path.dirname(file_path)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir)
            
            file_handler = RotatingFileHandler(
                file_path,
                maxBytes=max_bytes,
                backupCount=backup_count
            )
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(detailed_formatter)
            logger.addHandler(file_handler)
        
        # Cache logger
        cls._loggers[name] = logger
        
        return logger
    
    @classmethod
    def setup_global_logging(cls, config: Optional[dict] = None):
        """Setup global logging configuration.
        
        Args:
            config: Optional logging configuration dict
        """
        if config is None:
            from ..config import config as app_config
            level = app_config.logging.level
        else:
            level = config.get('level', 'INFO')
        
        # Configure root logger
        logging.basicConfig(
            level=getattr(logging, level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Suppress noisy libraries
        logging.getLogger('urllib3').setLevel(logging.WARNING)
        logging.getLogger('ccxt').setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Convenience function to get a logger.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Configured logger instance
    """
    return Logger.get_logger(name)