"""
Data loading module for WhiskeyHub ML pipeline.

This module provides the DataLoader class that handles loading and merging
CSV data files from the WhiskeyHub database export.
"""

import pandas as pd
import os
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional
from sklearn.model_selection import train_test_split

# Set up logging
logger = logging.getLogger(__name__)


class DataLoader:
    """
    DataLoader handles loading, merging, and basic preprocessing of WhiskeyHub data.
    
    This class refactors the functionality from scripts/db_connect.py into a
    reusable, configurable, and testable component.
    """
    
    def __init__(
        self, 
        data_path: str = None,
        results_path: str = None,
        encoding: str = 'utf-8-sig',
        random_state: int = 42
    ):
        """
        Initialize DataLoader with configuration.
        
        Args:
            data_path: Path to directory containing CSV files
            results_path: Path to save processed data
            encoding: Encoding for CSV files (utf-8-sig handles BOM)
            random_state: Random state for reproducible splits
        """
        # Set default paths relative to project root
        project_root = Path(__file__).parent.parent.parent
        
        self.data_path = data_path or str(project_root / "data" / "WhiskeyHubMySQL_6_13_2025_pt2")
        self.results_path = results_path or str(project_root / "results")
        self.encoding = encoding
        self.random_state = random_state
        
        # Data storage
        self.raw_data: Dict[str, pd.DataFrame] = {}
        self.merged_data: Optional[pd.DataFrame] = None
        
        # File names
        self.file_names = {
            'flights': 'flights.csv',
            'flight_pours': 'flight_pours.csv', 
            'flight_notes': 'flight_notes.csv',
            'whiskeys': 'whishkeys.csv'  # Note: original typo preserved
        }
        
        logger.info(f"DataLoader initialized with data_path: {self.data_path}")
    
    def load_raw_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load all raw CSV files into memory.
        
        Returns:
            Dictionary mapping table names to DataFrames
        """
        logger.info("Loading raw CSV files...")
        
        for table_name, file_name in self.file_names.items():
            file_path = os.path.join(self.data_path, file_name)
            
            try:
                df = pd.read_csv(file_path, encoding=self.encoding)
                self.raw_data[table_name] = df
                logger.info(f"✅ Loaded {table_name}: {len(df)} rows")
                
            except Exception as e:
                logger.error(f"❌ Error loading {table_name}: {e}")
                self.raw_data[table_name] = pd.DataFrame()
        
        return self.raw_data
    
    def validate_raw_data(self) -> Dict[str, bool]:
        """
        Validate that raw data is suitable for merging.
        
        Returns:
            Dictionary mapping table names to validation status
        """
        validation_results = {}
        
        for table_name, df in self.raw_data.items():
            is_valid = len(df) > 0
            validation_results[table_name] = is_valid
            
            if not is_valid:
                logger.warning(f"⚠️ {table_name} is empty or failed to load")
        
        # Check critical tables for merging
        critical_tables = ['flight_pours', 'flight_notes']
        can_merge = all(validation_results.get(table, False) for table in critical_tables)
        
        if not can_merge:
            logger.error("❌ Cannot perform merge - missing critical tables")
        else:
            logger.info("✅ Data validation passed - ready for merging")
        
        return validation_results
    
    def merge_data(self) -> pd.DataFrame:
        """
        Merge raw data tables following the established logic.
        
        This preserves the exact merge logic from scripts/db_connect.py
        that has been validated to work correctly.
        
        Returns:
            Merged DataFrame
        """
        if not self.raw_data:
            raise ValueError("No raw data loaded. Call load_raw_data() first.")
        
        pours = self.raw_data.get('flight_pours', pd.DataFrame())
        notes = self.raw_data.get('flight_notes', pd.DataFrame())
        flights = self.raw_data.get('flights', pd.DataFrame())
        whiskeys = self.raw_data.get('whiskeys', pd.DataFrame())
        
        if len(pours) == 0 or len(notes) == 0:
            raise ValueError("Cannot merge - missing critical data (pours or notes)")
        
        logger.info("🔄 Merging tables...")
        
        # Step 1: Merge pours with notes (core merge)
        df = pours.merge(
            notes, 
            left_on="id", 
            right_on="flight_pour_id", 
            how="inner", 
            suffixes=('_pour', '_note')
        )
        logger.info(f"  - Pours + Notes: {len(df)} rows")
        
        # Step 2: Merge with flights (if available)
        if len(flights) > 0:
            # Dynamic column selection (preserves existing logic)
            flight_col = 'flight_id_pour' if 'flight_id_pour' in df.columns else 'flight_id'
            df = df.merge(
                flights, 
                left_on=flight_col, 
                right_on="id", 
                how="left", 
                suffixes=('', '_flight')
            )
            logger.info(f"  - + Flights: {len(df)} rows")
        
        # Step 3: Merge with whiskeys (if available)
        if len(whiskeys) > 0:
            df = df.merge(
                whiskeys, 
                left_on="whiskey_id", 
                right_on="id", 
                how="left", 
                suffixes=('', '_whiskey')
            )
            logger.info(f"  - + Whiskeys: {len(df)} rows")
        
        self.merged_data = df
        logger.info(f"✅ Merge completed: {len(df)} rows, {len(df.columns)} columns")
        
        return df
    
    def save_merged_data(self, filename: str = "full_joined.csv") -> str:
        """
        Save merged data to results directory.
        
        Args:
            filename: Name of output file
            
        Returns:
            Path to saved file
        """
        if self.merged_data is None:
            raise ValueError("No merged data to save. Call merge_data() first.")
        
        # Ensure results directory exists
        os.makedirs(self.results_path, exist_ok=True)
        
        output_path = os.path.join(self.results_path, filename)
        self.merged_data.to_csv(output_path, index=False)
        
        logger.info(f"✅ Merged data saved to {output_path}")
        return output_path
    
    def load_merged_data(self, filename: str = "full_joined.csv") -> pd.DataFrame:
        """
        Load previously saved merged data.
        
        Args:
            filename: Name of file to load
            
        Returns:
            Merged DataFrame
        """
        file_path = os.path.join(self.results_path, filename)
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Merged data file not found: {file_path}")
        
        self.merged_data = pd.read_csv(file_path)
        logger.info(f"✅ Loaded merged data from {file_path}: {len(self.merged_data)} rows")
        
        return self.merged_data
    
    def train_test_split(
        self, 
        test_size: float = 0.2, 
        stratify_column: str = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split merged data into train and test sets.
        
        Args:
            test_size: Fraction of data for testing
            stratify_column: Column to stratify split on (optional)
            
        Returns:
            Tuple of (train_df, test_df)
        """
        if self.merged_data is None:
            raise ValueError("No merged data available. Call merge_data() first.")
        
        stratify = self.merged_data[stratify_column] if stratify_column else None
        
        train_df, test_df = train_test_split(
            self.merged_data,
            test_size=test_size,
            random_state=self.random_state,
            stratify=stratify
        )
        
        logger.info(f"✅ Data split: {len(train_df)} train, {len(test_df)} test")
        return train_df, test_df
    
    def get_data_info(self) -> Dict:
        """
        Get summary information about the loaded data.
        
        Returns:
            Dictionary with data statistics
        """
        info = {
            'raw_data_loaded': bool(self.raw_data),
            'merged_data_available': self.merged_data is not None,
        }
        
        if self.raw_data:
            info['raw_data_counts'] = {
                name: len(df) for name, df in self.raw_data.items()
            }
        
        if self.merged_data is not None:
            info.update({
                'merged_rows': len(self.merged_data),
                'merged_columns': len(self.merged_data.columns),
                'column_names': list(self.merged_data.columns),
            })
        
        return info
    
    def load_and_merge(self) -> pd.DataFrame:
        """
        Convenience method to load raw data and merge in one call.
        
        Returns:
            Merged DataFrame
        """
        self.load_raw_data()
        validation = self.validate_raw_data()
        
        if not all(validation[table] for table in ['flight_pours', 'flight_notes']):
            raise ValueError("Critical data validation failed")
        
        return self.merge_data()


# Convenience function for backward compatibility
def load_whiskeyhub_data(data_path: str = None, results_path: str = None) -> pd.DataFrame:
    """
    Convenience function that replicates the behavior of scripts/db_connect.py
    
    Args:
        data_path: Path to CSV data directory
        results_path: Path to save results
        
    Returns:
        Merged DataFrame
    """
    loader = DataLoader(data_path=data_path, results_path=results_path)
    merged_data = loader.load_and_merge()
    loader.save_merged_data()
    return merged_data