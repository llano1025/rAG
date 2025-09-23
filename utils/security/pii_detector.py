import re
from typing import List, Dict, Set, Pattern, Optional, Union
import spacy
from dataclasses import dataclass, field
import logging
from pathlib import Path
import json

@dataclass
class PIIPattern:
    name: str
    pattern: Union[str, Pattern]
    description: str
    sensitivity_level: int = 1  # 1-5, with 5 being most sensitive

@dataclass
class PIIConfig:
    patterns_path: Optional[Path] = None
    custom_patterns: List[PIIPattern] = field(default_factory=list)
    use_spacy: bool = False
    spacy_model: str = "en_core_web_sm"
    min_confidence: float = 0.85

class PIIDetector:
    """Detects and handles Personally Identifiable Information (PII) in text."""

    DEFAULT_PATTERNS = [
        PIIPattern(
            name="email",
            pattern=r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            description="Email addresses",
            sensitivity_level=3
        ),
        PIIPattern(
            name="ssn",
            pattern=r'\b\d{3}[-.]?\d{2}[-.]?\d{4}\b',
            description="Social Security Numbers",
            sensitivity_level=5
        ),
        PIIPattern(
            name="credit_card",
            pattern=r'\b(?:\d[ -]*?){13,16}\b',
            description="Credit Card Numbers",
            sensitivity_level=5
        ),
        PIIPattern(
            name="phone_number",
            pattern=r'\b(?:\+\d{1,2}\s)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b',
            description="Phone Numbers",
            sensitivity_level=3
        ),
        PIIPattern(
            name="ip_address",
            pattern=r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
            description="IP Addresses",
            sensitivity_level=2
        ),
    ]

    def __init__(self, config: PIIConfig):
        self.config = config
        self.patterns: List[PIIPattern] = []
        self._nlp = None
        self._initialize_detector()

    def _initialize_detector(self) -> None:
        """Initialize the PII detector with patterns and NLP model."""
        try:
            # Load default patterns
            self.patterns.extend(self.DEFAULT_PATTERNS)

            # Load custom patterns from file if specified
            if self.config.patterns_path:
                self._load_patterns_from_file()

            # Add any custom patterns provided directly
            self.patterns.extend(self.config.custom_patterns)

            # Compile regex patterns
            for pattern in self.patterns:
                if isinstance(pattern.pattern, str):
                    pattern.pattern = re.compile(pattern.pattern)

            # Initialize spaCy if needed
            if self.config.use_spacy:
                self._initialize_spacy()

        except Exception as e:
            logging.error(f"Failed to initialize PII detector: {str(e)}")
            raise

    def _initialize_spacy(self) -> None:
        """Initialize the spaCy NLP model."""
        try:
            self._nlp = spacy.load(self.config.spacy_model)
        except Exception as e:
            logging.error(f"Failed to load spaCy model: {str(e)}")
            self.config.use_spacy = False

    def _load_patterns_from_file(self) -> None:
        """Load custom PII patterns from a JSON file."""
        try:
            with open(self.config.patterns_path) as f:
                patterns_data = json.load(f)
                for pattern_data in patterns_data:
                    self.patterns.append(PIIPattern(**pattern_data))
        except Exception as e:
            logging.error(f"Failed to load patterns from file: {str(e)}")
            raise

    def detect(self, text: str) -> List[Dict]:
        """
        Detect PII in the given text.
        
        Args:
            text: The text to analyze
            
        Returns:
            List[Dict]: List of detected PII instances with type, value, position, and confidence
        """
        findings = []
        
        # Regex-based detection
        for pattern in self.patterns:
            matches = pattern.pattern.finditer(text)
            for match in matches:
                findings.append({
                    'type': pattern.name,
                    'value': match.group(),
                    'start': match.start(),
                    'end': match.end(),
                    'sensitivity': pattern.sensitivity_level,
                    'confidence': 1.0,
                    'method': 'regex'
                })

        # NER-based detection using spaCy
        if self.config.use_spacy and self._nlp:
            doc = self._nlp(text)
            for ent in doc.ents:
                if ent.label_ in {'PERSON', 'ORG', 'GPE', 'LOC'}:
                    if ent.label_prob >= self.config.min_confidence:
                        findings.append({
                            'type': ent.label_,
                            'value': ent.text,
                            'start': ent.start_char,
                            'end': ent.end_char,
                            'sensitivity': 3,  # Default sensitivity for NER findings
                            'confidence': ent.label_prob,
                            'method': 'ner'
                        })

        return findings

    def redact(self, text: str, replacement: str = '[REDACTED]') -> str:
        """
        Redact detected PII from the text.
        
        Args:
            text: Text to redact
            replacement: String to use for redaction
            
        Returns:
            str: Text with PII redacted
        """
        findings = self.detect(text)
        findings.sort(key=lambda x: x['start'], reverse=True)
        
        result = text
        for finding in findings:
            result = (
                result[:finding['start']] +
                replacement +
                result[finding['end']:]
            )
        return result

    def analyze_document(self, document_path: Union[str, Path]) -> Dict:
        """
        Analyze a document file for PII.
        
        Args:
            document_path: Path to the document to analyze
            
        Returns:
            Dict: Analysis results including detected PII and statistics
        """
        try:
            with open(document_path, 'r') as f:
                content = f.read()

            findings = self.detect(content)
            
            # Group findings by type
            findings_by_type = {}
            for finding in findings:
                pii_type = finding['type']
                if pii_type not in findings_by_type:
                    findings_by_type[pii_type] = []
                findings_by_type[pii_type].append(finding)

            # Calculate statistics
            stats = {
                'total_findings': len(findings),
                'types_found': len(findings_by_type),
                'findings_by_type': {
                    pii_type: len(findings)
                    for pii_type, findings in findings_by_type.items()
                },
                'highest_sensitivity': max(
                    (f['sensitivity'] for f in findings),
                    default=0
                )
            }

            return {
                'document_path': str(document_path),
                'findings': findings,
                'statistics': stats
            }

        except Exception as e:
            logging.error(f"Document analysis failed: {str(e)}")
            raise