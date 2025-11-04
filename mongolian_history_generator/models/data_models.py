"""
Data models for the Mongolian History Generator.

Defines the core data structures used throughout the application.
"""

from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any
import json
import re


@dataclass
class HistoricalEntry:
    """Represents a single historical entry with title, date, and content."""
    
    title: str
    date: str
    content: str
    
    def validate(self) -> bool:
        """
        Validates entry format and content requirements.
        
        Returns:
            bool: True if entry is valid, False otherwise
        """
        return (
            self._validate_title() and
            self._validate_date() and
            self._validate_content()
        )
    
    def _validate_title(self) -> bool:
        """Validate title field is present and not empty."""
        return bool(self.title and self.title.strip())
    
    def _validate_date(self) -> bool:
        """Validate date format follows YYYY or YYYY-MM-DD pattern."""
        if not self.date:
            return False
        
        # Check YYYY format
        if re.match(r'^\d{4}$', self.date):
            return True
        
        # Check YYYY-MM-DD format
        if re.match(r'^\d{4}-\d{2}-\d{2}$', self.date):
            return True
        
        return False
    
    def _validate_content(self) -> bool:
        """
        Validate content length (80-150 words) and paragraph structure (1-3 paragraphs).
        
        Returns:
            bool: True if content meets requirements, False otherwise
        """
        if not self.content or not self.content.strip():
            return False
        
        # Count words
        words = len(self.content.split())
        if words < 80 or words > 150:
            return False
        
        # Count paragraphs (split by double newlines or single newlines)
        paragraphs = [p.strip() for p in self.content.split('\n\n') if p.strip()]
        if not paragraphs:
            # Try single newline split if no double newlines
            paragraphs = [p.strip() for p in self.content.split('\n') if p.strip()]
        
        # If still no paragraphs, treat as single paragraph
        if not paragraphs:
            paragraphs = [self.content.strip()]
        
        return 1 <= len(paragraphs) <= 3
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'HistoricalEntry':
        """Create instance from dictionary."""
        return cls(
            title=data['title'],
            date=data['date'],
            content=data['content']
        )


@dataclass
class GenerationResult:
    """Represents the result of processing a single topic."""
    
    topic: str
    entries: List[HistoricalEntry]
    success: bool
    error_message: Optional[str] = None
    tokens_used: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'topic': self.topic,
            'entries': [entry.to_dict() for entry in self.entries],
            'success': self.success,
            'error_message': self.error_message,
            'tokens_used': self.tokens_used
        }


@dataclass
class SummaryReport:
    """Represents a summary report of the entire generation process."""
    
    total_topics: int
    successful_generations: int
    failed_generations: int
    total_entries: int
    total_tokens_used: int
    errors: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)