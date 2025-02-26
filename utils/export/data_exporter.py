from typing import List, Dict, Any, Optional
import json
import csv
import pandas as pd
from datetime import datetime
import logging
from pathlib import Path
import aiofiles
import asyncio
from fastapi import HTTPException

logger = logging.getLogger(__name__)

class DataExporter:
    """
    Handles exporting data in various formats and managing export jobs.
    Supports async operations and different export formats.
    """
    
    def __init__(self, export_dir: str = "exports"):
        self.export_dir = Path(export_dir)
        self.export_dir.mkdir(exist_ok=True)
        self.supported_formats = ['json', 'csv', 'excel']
        
    async def export_data(
        self,
        data: List[Dict[Any, Any]],
        format: str,
        filename: Optional[str] = None,
        include_metadata: bool = True
    ) -> str:
        """
        Export data to specified format asynchronously.
        
        Args:
            data: List of dictionaries containing the data to export
            format: Export format (json, csv, excel)
            filename: Optional custom filename
            include_metadata: Whether to include export metadata
            
        Returns:
            str: Path to exported file
        """
        if format not in self.supported_formats:
            raise ValueError(f"Unsupported format: {format}")
            
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"export_{timestamp}.{format}"
            
        export_path = self.export_dir / filename
        
        try:
            if include_metadata:
                metadata = {
                    "exported_at": datetime.now().isoformat(),
                    "record_count": len(data),
                    "format": format
                }
                data = self._add_metadata(data, metadata)
            
            if format == 'json':
                await self._export_json(data, export_path)
            elif format == 'csv':
                await self._export_csv(data, export_path)
            elif format == 'excel':
                await self._export_excel(data, export_path)
                
            logger.info(f"Successfully exported data to {export_path}")
            return str(export_path)
            
        except Exception as e:
            logger.error(f"Error exporting data: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to export data: {str(e)}"
            )

    async def _export_json(self, data: List[Dict], path: Path) -> None:
        """Export data to JSON file"""
        async with aiofiles.open(path, 'w') as f:
            await f.write(json.dumps(data, indent=2))

    async def _export_csv(self, data: List[Dict], path: Path) -> None:
        """Export data to CSV file"""
        if not data:
            raise ValueError("No data to export")
            
        fieldnames = list(data[0].keys())
        
        async with aiofiles.open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            await f.write(','.join(fieldnames) + '\n')
            for row in data:
                await f.write(','.join(str(row.get(field, '')) for field in fieldnames) + '\n')

    async def _export_excel(self, data: List[Dict], path: Path) -> None:
        """Export data to Excel file"""
        df = pd.DataFrame(data)
        await asyncio.to_thread(df.to_excel, path, index=False)

    def _add_metadata(self, data: List[Dict], metadata: Dict) -> List[Dict]:
        """Add metadata to the export data"""
        return [{**item, "_export_metadata": metadata} for item in data]

    async def get_export_status(self, export_id: str) -> Dict:
        """
        Get status of an export job
        
        Args:
            export_id: ID of the export job
            
        Returns:
            Dict containing export job status
        """
        # Implementation for tracking export job status
        pass

    async def cleanup_old_exports(self, days: int = 7) -> None:
        """
        Clean up export files older than specified days
        
        Args:
            days: Number of days to keep exports
        """
        cutoff = datetime.now().timestamp() - (days * 24 * 60 * 60)
        
        for file_path in self.export_dir.glob("*"):
            if file_path.stat().st_mtime < cutoff:
                try:
                    file_path.unlink()
                    logger.info(f"Cleaned up old export: {file_path}")
                except Exception as e:
                    logger.error(f"Error cleaning up {file_path}: {str(e)}")

    async def get_available_exports(self) -> List[Dict]:
        """
        Get list of available export files
        
        Returns:
            List of dictionaries containing export file information
        """
        exports = []
        for file_path in self.export_dir.glob("*"):
            stats = file_path.stat()
            exports.append({
                "filename": file_path.name,
                "size": stats.st_size,
                "created_at": datetime.fromtimestamp(stats.st_mtime).isoformat(),
                "format": file_path.suffix[1:]
            })
        return sorted(exports, key=lambda x: x["created_at"], reverse=True)