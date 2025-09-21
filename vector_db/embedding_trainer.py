"""
Custom Embedding Model Training for RAG System
Provides capabilities to train domain-specific embedding models.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, TrainingArguments, Trainer
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
import pickle
from pathlib import Path
import aiofiles

from database.connection import get_db
from utils.monitoring.metrics import MetricsCollector

logger = logging.getLogger(__name__)


class TrainingMethod(Enum):
    """Training methods for embedding models"""
    CONTRASTIVE = "contrastive"
    TRIPLET = "triplet"
    COSINE_SIMILARITY = "cosine_similarity"
    MULTIPLE_NEGATIVES_RANKING = "multiple_negatives_ranking"
    SENTENCE_TRANSFORMER_FINE_TUNE = "sentence_transformer_fine_tune"


class TrainingStatus(Enum):
    """Status of training job"""
    PENDING = "pending"
    PREPARING_DATA = "preparing_data"
    TRAINING = "training"
    EVALUATING = "evaluating"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TrainingConfig:
    """Configuration for embedding model training"""
    model_name: str
    base_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    training_method: TrainingMethod = TrainingMethod.SENTENCE_TRANSFORMER_FINE_TUNE
    max_seq_length: int = 512
    train_batch_size: int = 16
    eval_batch_size: int = 16
    num_epochs: int = 3
    learning_rate: float = 2e-5
    warmup_steps: int = 100
    evaluation_steps: int = 500
    save_steps: int = 1000
    output_dir: str = "models/custom_embeddings"
    use_gpu: bool = True
    gradient_accumulation_steps: int = 1
    fp16: bool = False
    dataloader_num_workers: int = 0


@dataclass
class TrainingExample:
    """Training example for embedding model"""
    text_a: str
    text_b: str
    label: float  # 0.0 for dissimilar, 1.0 for similar
    source: str = "generated"


@dataclass
class TrainingJob:
    """Training job information"""
    job_id: str
    config: TrainingConfig
    status: TrainingStatus = TrainingStatus.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress: float = 0.0
    current_epoch: int = 0
    training_loss: Optional[float] = None
    evaluation_score: Optional[float] = None
    error_message: Optional[str] = None
    model_path: Optional[str] = None


class DocumentSimilarityDataset(Dataset):
    """Dataset for document similarity training"""
    
    def __init__(self, examples: List[TrainingExample], tokenizer, max_length: int = 512):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Tokenize both texts
        inputs_a = self.tokenizer(
            example.text_a,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        inputs_b = self.tokenizer(
            example.text_b,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids_a': inputs_a['input_ids'].squeeze(),
            'attention_mask_a': inputs_a['attention_mask'].squeeze(),
            'input_ids_b': inputs_b['input_ids'].squeeze(),
            'attention_mask_b': inputs_b['attention_mask'].squeeze(),
            'labels': torch.tensor(example.label, dtype=torch.float)
        }


class CustomEmbeddingTrainer:
    """Custom embedding model trainer"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.metrics = MetricsCollector()
        self.training_jobs: Dict[str, TrainingJob] = {}
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() and config.use_gpu else 'cpu')
        logger.info(f"Using device: {self.device}")
    
    async def start_training(self, training_data: List[TrainingExample], 
                           validation_data: Optional[List[TrainingExample]] = None) -> str:
        """Start training a custom embedding model"""
        
        # Generate job ID
        job_id = f"training_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        # Create training job
        job = TrainingJob(
            job_id=job_id,
            config=self.config
        )
        
        self.training_jobs[job_id] = job
        
        # Start training in background
        asyncio.create_task(self._run_training(job, training_data, validation_data))
        
        return job_id
    
    async def _run_training(self, job: TrainingJob, training_data: List[TrainingExample],
                          validation_data: Optional[List[TrainingExample]] = None):
        """Run the training process"""
        try:
            job.status = TrainingStatus.PREPARING_DATA
            job.started_at = datetime.utcnow()
            
            logger.info(f"Starting training job {job.job_id}")
            
            # Prepare data
            await self._prepare_training_data(job, training_data, validation_data)
            
            # Start training
            job.status = TrainingStatus.TRAINING
            await self._train_model(job, training_data, validation_data)
            
            # Evaluate model
            job.status = TrainingStatus.EVALUATING
            await self._evaluate_model(job, validation_data)
            
            # Complete training
            job.status = TrainingStatus.COMPLETED
            job.completed_at = datetime.utcnow()
            job.progress = 100.0
            
            logger.info(f"Training job {job.job_id} completed successfully")
            
        except Exception as e:
            job.status = TrainingStatus.FAILED
            job.error_message = str(e)
            job.completed_at = datetime.utcnow()
            logger.error(f"Training job {job.job_id} failed: {e}")
    
    async def _prepare_training_data(self, job: TrainingJob, training_data: List[TrainingExample],
                                   validation_data: Optional[List[TrainingExample]]):
        """Prepare training data"""
        logger.info(f"Preparing {len(training_data)} training examples")
        
        # Convert to SentenceTransformer format
        train_examples = []
        for example in training_data:
            train_examples.append(InputExample(
                texts=[example.text_a, example.text_b],
                label=example.label
            ))
        
        # Save training examples
        output_dir = Path(job.config.output_dir) / job.job_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_dir / 'training_examples.pkl', 'wb') as f:
            pickle.dump(train_examples, f)
        
        if validation_data:
            val_examples = []
            for example in validation_data:
                val_examples.append(InputExample(
                    texts=[example.text_a, example.text_b],
                    label=example.label
                ))
            
            with open(output_dir / 'validation_examples.pkl', 'wb') as f:
                pickle.dump(val_examples, f)
        
        job.progress = 20.0
    
    async def _train_model(self, job: TrainingJob, training_data: List[TrainingExample],
                         validation_data: Optional[List[TrainingExample]]):
        """Train the embedding model"""
        
        # Load base model
        model = SentenceTransformer(job.config.base_model, device=self.device)
        
        # Prepare training data
        train_examples = []
        for example in training_data:
            train_examples.append(InputExample(
                texts=[example.text_a, example.text_b],
                label=example.label
            ))
        
        # Create data loader
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=job.config.train_batch_size)
        
        # Choose loss function based on training method
        if job.config.training_method == TrainingMethod.COSINE_SIMILARITY:
            train_loss = losses.CosineSimilarityLoss(model)
        elif job.config.training_method == TrainingMethod.CONTRASTIVE:
            train_loss = losses.ContrastiveLoss(model)
        elif job.config.training_method == TrainingMethod.MULTIPLE_NEGATIVES_RANKING:
            train_loss = losses.MultipleNegativesRankingLoss(model)
        else:
            train_loss = losses.CosineSimilarityLoss(model)  # Default
        
        # Prepare evaluator
        evaluator = None
        if validation_data:
            val_examples = []
            for example in validation_data:
                val_examples.append(InputExample(
                    texts=[example.text_a, example.text_b],
                    label=example.label
                ))
            
            evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
                val_examples, batch_size=job.config.eval_batch_size, name='validation'
            )
        
        # Set up output directory
        output_path = Path(job.config.output_dir) / job.job_id
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Training callback to update progress
        class ProgressCallback:
            def __init__(self, job, total_steps):
                self.job = job
                self.total_steps = total_steps
                self.current_step = 0
            
            def __call__(self, score, epoch, steps):
                self.current_step = steps
                self.job.current_epoch = epoch
                self.job.progress = 20.0 + (steps / self.total_steps) * 60.0  # 20-80% for training
                if hasattr(score, 'spearman_cosine'):
                    self.job.evaluation_score = score.spearman_cosine
        
        total_steps = len(train_dataloader) * job.config.num_epochs
        progress_callback = ProgressCallback(job, total_steps)
        
        # Train the model
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            evaluator=evaluator,
            epochs=job.config.num_epochs,
            evaluation_steps=job.config.evaluation_steps,
            warmup_steps=job.config.warmup_steps,
            output_path=str(output_path),
            save_best_model=True,
            callback=progress_callback
        )
        
        # Save the final model path
        job.model_path = str(output_path)
        
        logger.info(f"Model training completed for job {job.job_id}")
    
    async def _evaluate_model(self, job: TrainingJob, validation_data: Optional[List[TrainingExample]]):
        """Evaluate the trained model"""
        if not validation_data or not job.model_path:
            job.progress = 100.0
            return
        
        try:
            # Load trained model
            model = SentenceTransformer(job.model_path)
            
            # Prepare evaluation data
            sentences1 = [example.text_a for example in validation_data]
            sentences2 = [example.text_b for example in validation_data]
            labels = [example.label for example in validation_data]
            
            # Generate embeddings
            embeddings1 = model.encode(sentences1)
            embeddings2 = model.encode(sentences2)
            
            # Calculate cosine similarities
            cosine_scores = []
            for emb1, emb2 in zip(embeddings1, embeddings2):
                cosine_score = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
                cosine_scores.append(cosine_score)
            
            # Calculate correlation with ground truth
            correlation = np.corrcoef(cosine_scores, labels)[0, 1]
            job.evaluation_score = correlation
            
            # Save evaluation results
            eval_results = {
                'correlation': correlation,
                'predictions': cosine_scores,
                'ground_truth': labels,
                'evaluation_date': datetime.utcnow().isoformat()
            }
            
            output_dir = Path(job.model_path)
            with open(output_dir / 'evaluation_results.json', 'w') as f:
                json.dump(eval_results, f, indent=2)
            
            logger.info(f"Model evaluation completed. Correlation: {correlation:.4f}")
            
        except Exception as e:
            logger.error(f"Error during model evaluation: {e}")
        
        job.progress = 100.0
    
    async def generate_training_data_from_documents(self, num_examples: int = 1000) -> List[TrainingExample]:
        """Generate training data from existing documents"""
        training_examples = []
        
        try:
            async with get_db() as session:
                # Get random pairs of documents
                result = await session.execute(
                    """SELECT d1.title, d1.content, d2.title, d2.content
                       FROM documents d1, documents d2
                       WHERE d1.id != d2.id 
                         AND d1.status = 'completed' 
                         AND d2.status = 'completed'
                         AND length(d1.content) > 100
                         AND length(d2.content) > 100
                       ORDER BY RANDOM()
                       LIMIT ?""",
                    (num_examples * 2,)
                )
                
                document_pairs = result.fetchall()
                
                for i in range(0, len(document_pairs) - 1, 2):
                    doc1 = document_pairs[i]
                    doc2 = document_pairs[i + 1]
                    
                    # Create similar pair (same document with slight variations)
                    text1 = doc1[1][:512]  # Truncate for training
                    text2 = doc1[1][50:562] if len(doc1[1]) > 562 else doc1[1]  # Slightly offset
                    
                    if len(text2) > 100:  # Ensure minimum length
                        training_examples.append(TrainingExample(
                            text_a=text1,
                            text_b=text2,
                            label=0.8,  # High similarity
                            source="document_overlap"
                        ))
                    
                    # Create dissimilar pair
                    text3 = doc1[1][:512]
                    text4 = doc2[1][:512]
                    
                    training_examples.append(TrainingExample(
                        text_a=text3,
                        text_b=text4,
                        label=0.1,  # Low similarity
                        source="different_documents"
                    ))
                    
                    if len(training_examples) >= num_examples:
                        break
            
            logger.info(f"Generated {len(training_examples)} training examples from documents")
            return training_examples[:num_examples]
            
        except Exception as e:
            logger.error(f"Error generating training data: {e}")
            return []
    
    async def get_job_status(self, job_id: str) -> Optional[Dict]:
        """Get status of a training job"""
        if job_id not in self.training_jobs:
            return None
        
        job = self.training_jobs[job_id]
        
        return {
            'job_id': job.job_id,
            'status': job.status.value,
            'progress': job.progress,
            'current_epoch': job.current_epoch,
            'created_at': job.created_at.isoformat(),
            'started_at': job.started_at.isoformat() if job.started_at else None,
            'completed_at': job.completed_at.isoformat() if job.completed_at else None,
            'training_loss': job.training_loss,
            'evaluation_score': job.evaluation_score,
            'error_message': job.error_message,
            'model_path': job.model_path,
            'config': {
                'model_name': job.config.model_name,
                'base_model': job.config.base_model,
                'training_method': job.config.training_method.value,
                'num_epochs': job.config.num_epochs,
                'learning_rate': job.config.learning_rate
            }
        }
    
    async def list_trained_models(self) -> List[Dict]:
        """List all trained models"""
        models = []
        
        models_dir = Path(self.config.output_dir)
        if models_dir.exists():
            for model_dir in models_dir.iterdir():
                if model_dir.is_dir():
                    # Check if it's a valid model directory
                    if (model_dir / 'config.json').exists():
                        try:
                            # Load model info
                            with open(model_dir / 'config.json', 'r') as f:
                                config_data = json.load(f)
                            
                            # Load evaluation results if available
                            eval_results = {}
                            eval_file = model_dir / 'evaluation_results.json'
                            if eval_file.exists():
                                with open(eval_file, 'r') as f:
                                    eval_results = json.load(f)
                            
                            models.append({
                                'model_id': model_dir.name,
                                'model_path': str(model_dir),
                                'created_at': datetime.fromtimestamp(model_dir.stat().st_ctime).isoformat(),
                                'evaluation_score': eval_results.get('correlation'),
                                'base_model': config_data.get('base_model', 'unknown')
                            })
                            
                        except Exception as e:
                            logger.warning(f"Error loading model info for {model_dir}: {e}")
        
        return sorted(models, key=lambda x: x['created_at'], reverse=True)
    
    async def load_custom_model(self, model_path: str) -> SentenceTransformer:
        """Load a custom trained model"""
        try:
            model = SentenceTransformer(model_path, device=self.device)
            logger.info(f"Loaded custom model from {model_path}")
            return model
        except Exception as e:
            logger.error(f"Error loading custom model: {e}")
            raise
    
    def cancel_training(self, job_id: str) -> bool:
        """Cancel a running training job"""
        if job_id not in self.training_jobs:
            return False
        
        job = self.training_jobs[job_id]
        if job.status in [TrainingStatus.PENDING, TrainingStatus.PREPARING_DATA, TrainingStatus.TRAINING]:
            job.status = TrainingStatus.CANCELLED
            job.completed_at = datetime.utcnow()
            logger.info(f"Training job {job_id} cancelled")
            return True
        
        return False


# Factory function
def create_embedding_trainer(config: TrainingConfig) -> CustomEmbeddingTrainer:
    """Create an embedding trainer instance"""
    return CustomEmbeddingTrainer(config)