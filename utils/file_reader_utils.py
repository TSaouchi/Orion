import json
import yaml
import configparser
import csv
import os
from abc import ABC, abstractmethod

class ReaderStrategy(ABC):
    """
    Abstract base class for all file reader strategies
    """
    
    @abstractmethod
    def read(self, file_path: str):
        """
        Read and parse the file at the given path
        """
        pass

class JSONReaderStrategy(ReaderStrategy):
    def read(self, file_path: str) -> dict:
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return json.load(file)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format in '{file_path}': {str(e)}")

class YAMLReaderStrategy(ReaderStrategy):
    def read(self, file_path: str) -> dict:
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return yaml.safe_load(file)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML format in '{file_path}': {str(e)}")

class INIReaderStrategy(ReaderStrategy):
    def read(self, file_path: str) -> dict:
        try:
            config = configparser.ConfigParser(comment_prefixes=(';', '#'))
            config.read(file_path, encoding='utf-8')
            # Convert to dictionary
            return {section: dict(config[section]) for section in config.sections()}
        except configparser.Error as e:
            raise ValueError(f"Invalid INI format in '{file_path}': {str(e)}")

class TXTReaderStrategy(ReaderStrategy):
    def read(self, file_path: str) -> list:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.readlines()

class CSVReaderStrategy(ReaderStrategy):
    def read(self, file_path: str) -> list:
        try:
            with open(file_path, 'r', encoding='utf-8', newline='') as file:
                csv_reader = csv.DictReader(file)
                return list(csv_reader)
        except csv.Error as e:
            raise ValueError(f"Invalid CSV format in '{file_path}': {str(e)}")

class DATReaderStrategy(ReaderStrategy):
    def read(self, file_path: str) -> bytes:
        with open(file_path, 'rb') as file:
            return file.read()

class BinaryReaderStrategy(ReaderStrategy):
    def read(self, file_path: str) -> bytes:
        with open(file_path, 'rb') as file:
            return file.read()

class FileReader:
    """
    Main class that determines the appropriate strategy and reads the file
    """

    # Map file extensions to reader strategies
    _strategies = {
        '.json': JSONReaderStrategy(),
        '.yaml': YAMLReaderStrategy(),
        '.yml': YAMLReaderStrategy(),
        '.ini': INIReaderStrategy(),
        '.cfg': INIReaderStrategy(),
        '.txt': TXTReaderStrategy(),
        '.csv': CSVReaderStrategy(),
        '.dat': DATReaderStrategy(),
    }

    def __init__(self, file_path: str):
        """
        Initialize with the path to the file to be read
        """
        self.file_path = file_path
        self._strategy = self._get_strategy()

    def _get_strategy(self) -> ReaderStrategy:
        """Determine the appropriate reader strategy based on file extension"""
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"File not found: {self.file_path}")

        _, ext = os.path.splitext(self.file_path)

        if ext in self._strategies:
            return self._strategies[ext.lower()]
        else:
            # Default to binary reader for unknown extensions
            return BinaryReaderStrategy()

    def read(self):
        """
        Read and parse the file using the appropriate strategy
        """
        try:
            return self._strategy.read(self.file_path)
        except Exception as e:
            raise RuntimeError(f"Error reading file '{self.file_path}': {str(e)}")

    @classmethod
    def register_strategy(cls, extension: str, strategy: ReaderStrategy) -> None:
        """
        Register a new file reading strategy
        """
        cls._strategies[extension.lower()] = strategy