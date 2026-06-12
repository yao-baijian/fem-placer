import sys
import os


class Logger:
    _instance = None
    _log_levels = {
        'ERROR': 1,
        'WARNING': 2,
        'INFO': 3,
        'DEBUG': 4
    }
    
    def __new__(cls, level='INFO', show_class_name=True):
        if cls._instance is None:
            cls._instance = super(Logger, cls).__new__(cls)
            cls._instance._initialize(level, show_class_name)
        return cls._instance
    
    def _initialize(self, level, show_class_name):
        self._set_level(level)
        self.show_class_name = show_class_name
        self.enabled = True
        self._log_file = None
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = Logger()
        return cls._instance
    
    def set_log_file(self, path: str):
        """Write all future log output to *path* in addition to stdout."""
        self._log_file = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Write a header
        with open(path, 'w', encoding='utf-8') as f:
            f.write(f"=== FEM Placer Log ===\n")
    
    def _write_file(self, formatted: str):
        if self._log_file:
            try:
                with open(self._log_file, 'a', encoding='utf-8') as f:
                    f.write(formatted + '\n')
            except Exception:
                pass

    def _set_level(self, level: str):
        level = level.upper()
        if level in self._log_levels:
            self.level = level
            self.level_value = self._log_levels[level]
        else:
            self.level = 'INFO'
            self.level_value = self._log_levels['INFO']
    
    def _should_log(self, message_level: str) -> bool:
        if not self.enabled:
            return False
        message_level_value = self._log_levels.get(message_level.upper(), 0)
        return message_level_value <= self.level_value
    
    def _format_message(self, message: str, message_level: str = 'INFO') -> str:
        caller_class = None
        if self.show_class_name:
            try:
                frame = sys._getframe(3)
                while frame:
                    self_obj = frame.f_locals.get('self')
                    if self_obj and self_obj.__class__.__name__ != 'Logger':
                        caller_class = self_obj.__class__.__name__
                        break
                    frame = frame.f_back
            except:
                pass
        
        # 使用实际的消息级别
        if caller_class:
            return f"{message_level.upper()} <{caller_class}>: {message}"
        else:
            return f"{message_level.upper()}: {message}"
    
    def log(self, message: str, level: str = 'INFO'):
        if self._should_log(level):
            formatted = self._format_message(message, level)
            print(formatted)
            self._write_file(formatted)
    
    def error(self, message: str):
        self.log(message, 'ERROR')
    
    def warning(self, message: str):
        self.log(message, 'WARNING')
    
    def info(self, message: str):
        self.log(message, 'INFO')
    
    def debug(self, message: str):
        self.log(message, 'DEBUG')
    
    def set_level(self, level: str):
        self._set_level(level)
    
    def enable(self):
        self.enabled = True
    
    def disable(self):
        self.enabled = False

def ERROR(*args):
    logger = Logger.get_instance()
    message = ' '.join(str(arg) for arg in args)
    logger.error(message)

def WARNING(*args):
    logger = Logger.get_instance()
    message = ' '.join(str(arg) for arg in args)
    logger.warning(message)

def INFO(*args):
    logger = Logger.get_instance()
    message = ' '.join(str(arg) for arg in args)
    logger.info(message)

def DEBUG(*args):
    logger = Logger.get_instance()
    message = ' '.join(str(arg) for arg in args)
    logger.debug(message)

def SET_LEVEL(level: str):
    logger = Logger.get_instance()
    logger.set_level(level)

def ENABLE():
    logger = Logger.get_instance()
    logger.enable()

def DISABLE():
    logger = Logger.get_instance()
    logger.disable()