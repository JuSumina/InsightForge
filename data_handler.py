import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import logging
from config import Config

logger = logging.getLogger(__name__)

class SampleDataGenerator:
    """Generate sample sales data for testing"""
    
    @staticmethod
    def generate_sales_data(size: int = None) -> pd.DataFrame:
        """Generate sample sales data matching CSV structure"""
        if size is None:
            size = Config.sample_data_size
        
        np.random.seed(42)
        
        data = []
        start_date = pd.Timestamp('2023-01-01')
        
        for i in range(size):
            record = {
                'date': start_date + pd.Timedelta(days=np.random.randint(0, 730)),
                'product': np.random.choice(Config.sample_products),
                'region': np.random.choice(Config.sample_regions),
                'sales': np.random.randint(100, 5000),
                'customer_age': np.random.randint(18, 70),
                'customer_gender': np.random.choice(Config.sample_genders),
                'customer_satisfaction': np.round(np.random.uniform(1, 5), 1)
            }
            data.append(record)
        
        return pd.DataFrame(data)

class DataProcessor:
    """Handles all data preparation and analysis"""
    
    def __init__(self):
        self.data: Optional[pd.DataFrame] = None
        self.processed_data: Dict[str, Any] = {}
        
    def load_csv_data(self, file_path: str = None) -> pd.DataFrame:
        """Load sales data from CSV file or generate sample data"""
        if file_path is None:
            file_path = Config.default_csv_path
            
        try:
            # Try to load CSV file
            self.data = pd.read_csv(file_path)
            self._standardize_columns()
            self._create_derived_features()
            logger.info(f"Successfully loaded {len(self.data)} records from {file_path}")
            
        except FileNotFoundError:
            logger.warning(f"File {file_path} not found. Using sample data.")
            self.data = SampleDataGenerator.generate_sales_data()
            self._standardize_columns()
            self._create_derived_features()
            
        except Exception as e:
            logger.error(f"Error loading CSV: {e}. Using sample data.")
            self.data = SampleDataGenerator.generate_sales_data()
            self._standardize_columns()
            self._create_derived_features()
            
        return self.data
    
    def _standardize_columns(self):
        """Standardize column names and types"""
        # Handle different possible column name formats
        column_mapping = {}
        for col in self.data.columns:
            if col.lower() in ['date']:
                column_mapping[col] = 'date'
            elif col.lower() in ['product']:
                column_mapping[col] = 'product'
            elif col.lower() in ['region']:
                column_mapping[col] = 'region'
            elif col.lower() in ['sales']:
                column_mapping[col] = 'sales'
            elif col.lower() in ['customer_age', 'age']:
                column_mapping[col] = 'customer_age'
            elif col.lower() in ['customer_gender', 'gender']:
                column_mapping[col] = 'customer_gender'
            elif col.lower() in ['customer_satisfaction', 'satisfaction']:
                column_mapping[col] = 'customer_satisfaction'
        
        self.data = self.data.rename(columns=column_mapping)
        
        # Convert date column
        if 'date' in self.data.columns:
            self.data['date'] = pd.to_datetime(self.data['date'])
    
    def _create_derived_features(self):
        """Create additional features for analysis"""
        if 'date' not in self.data.columns:
            return
            
        # Time-based features
        self.data['month']   = self.data['date'].dt.to_period('M')
        self.data['quarter'] = self.data['date'].dt.to_period('Q')
        self.data['year'] = self.data['date'].dt.year
        self.data['month_name'] = self.data['date'].dt.strftime('%B')
        self.data['day_of_week'] = self.data['date'].dt.day_name()
        
        # Age groups
        if 'customer_age' in self.data.columns:
            self.data['age_group'] = pd.cut(
                self.data['customer_age'], 
                bins=[0, 25, 35, 45, 55, 100], 
                labels=['18-25', '26-35', '36-45', '46-55', '55+']
            )
        
        # Satisfaction categories
        if 'customer_satisfaction' in self.data.columns:
            self.data['satisfaction_category'] = pd.cut(
                self.data['customer_satisfaction'],
                bins=[0, 3, 4, 5],
                labels=['Low (0-3)', 'Medium (3-4)', 'High (4-5)']
            )
    
    def perform_analysis(self) -> Dict[str, Any]:
        """Perform comprehensive business analysis"""
        if self.data is None:
            raise ValueError("Data not loaded. Call load_csv_data() first.")
        
        analysis = {
            'sales_performance': self._analyze_sales_performance(),
            'product_analysis': self._analyze_products(),
            'regional_analysis': self._analyze_regions(),
            'customer_demographics': self._analyze_demographics(),
            'satisfaction_analysis': self._analyze_satisfaction(),
            'statistical_measures': self._calculate_statistics(),
            'temporal_insights': self._analyze_temporal_patterns()
        }
        
        self.processed_data = analysis
        return analysis
    
    def _analyze_sales_performance(self) -> Dict[str, Any]:
        """Analyze overall sales performance"""
        sales_col = 'sales'
        if sales_col not in self.data.columns:
            return {}
            
        result = {
            'total_sales': float(self.data[sales_col].sum()),
            'total_transactions': int(len(self.data)),
            'avg_transaction_value': float(self.data[sales_col].mean()),
            'median_transaction_value': float(self.data[sales_col].median())
        }
        
        # Add time-based trends if date column exists
        if 'month' in self.data.columns:
            s = self.data.groupby('month')[sales_col].sum().astype(float)
            s.index = s.index.astype(str)
            result['monthly_trends'] = s.to_dict()
        if 'quarter' in self.data.columns:
            s = self.data.groupby('quarter')[sales_col].sum().astype(float)
            s.index = s.index.astype(str)
            result['quarterly_trends'] = s.to_dict()
            
        return result
    
    def _analyze_products(self) -> Dict[str, Any]:
        """Analyze product performance"""
        if 'product' not in self.data.columns or 'sales' not in self.data.columns:
            return {}
            
        product_metrics = self.data.groupby('product').agg({
            'sales': ['sum', 'mean', 'count']
        }).round(2)
        
        result = {
            'total_sales_by_product': product_metrics[('sales', 'sum')].astype(float).to_dict(),
            'avg_sales_by_product': product_metrics[('sales', 'mean')].astype(float).to_dict(),
            'transaction_count_by_product': product_metrics[('sales', 'count')].astype(int).to_dict()
        }
        
        # Add satisfaction if available
        if 'customer_satisfaction' in self.data.columns:
            satisfaction_by_product = self.data.groupby('product', observed=True)['customer_satisfaction'].mean().astype(float).to_dict()
            result['avg_satisfaction_by_product'] = satisfaction_by_product
            
        return result
    
    def _analyze_regions(self) -> Dict[str, Any]:
        """Analyze regional performance"""
        if 'region' not in self.data.columns or 'sales' not in self.data.columns:
            return {}
            
        regional_metrics = self.data.groupby('region').agg({
            'sales': ['sum', 'mean', 'count']
        }).round(2)
        
        result = {
            'total_sales_by_region': regional_metrics[('sales', 'sum')].astype(float).to_dict(),
            'avg_sales_by_region': regional_metrics[('sales', 'mean')].astype(float).to_dict(),
            'transaction_count_by_region': regional_metrics[('sales', 'count')].astype(int).to_dict()
        }
        
        # Add satisfaction if available
        if 'customer_satisfaction' in self.data.columns:
            satisfaction_by_region = self.data.groupby('region', observed=True)['customer_satisfaction'].mean().astype(float).to_dict()
            result['avg_satisfaction_by_region'] = satisfaction_by_region
            
        return result
    
    def _analyze_demographics(self) -> Dict[str, Any]:
        """Analyze customer demographics"""
        result = {}
        
        if 'age_group' in self.data.columns and 'sales' in self.data.columns:
            result['sales_by_age_group'] = self.data.groupby('age_group', observed=True)['sales'].sum().astype(float).to_dict()
            
        if 'customer_gender' in self.data.columns and 'sales' in self.data.columns:
            result['sales_by_gender'] = self.data.groupby('customer_gender', observed=True)['sales'].sum().astype(float).to_dict()
            
        return result
    
    def _analyze_satisfaction(self) -> Dict[str, Any]:
        """Analyze customer satisfaction"""
        if 'customer_satisfaction' not in self.data.columns:
            return {}
            
        result = {
            'overall_satisfaction_score': float(self.data['customer_satisfaction'].mean()),
            'satisfaction_distribution': self.data['customer_satisfaction'].value_counts().sort_index().astype(int).to_dict()
        }
        
        # Satisfaction by product
        if 'product' in self.data.columns:
            result['satisfaction_by_product'] = self.data.groupby('product', observed=True)['customer_satisfaction'].mean().astype(float).to_dict()
            
        return result
    
    def _calculate_statistics(self) -> Dict[str, Any]:
        """Calculate statistical measures"""
        result = {}
        
        if 'sales' in self.data.columns:
            sales_stats = {
                'sales_median': float(self.data['sales'].median()),
                'sales_std': float(self.data['sales'].std()),
                'sales_min': float(self.data['sales'].min()),
                'sales_max': float(self.data['sales'].max()),
                'sales_25th_percentile': float(self.data['sales'].quantile(0.25)),
                'sales_75th_percentile': float(self.data['sales'].quantile(0.75))
            }
            result.update(sales_stats)
        
        if 'customer_age' in self.data.columns:
            result['avg_customer_age'] = float(self.data['customer_age'].mean())
            
        if 'customer_satisfaction' in self.data.columns:
            result['avg_satisfaction'] = float(self.data['customer_satisfaction'].mean())
            
        return result
    
    def _analyze_temporal_patterns(self) -> Dict[str, Any]:
        """Analyze temporal patterns"""
        result = {}
        
        if 'day_of_week' in self.data.columns and 'sales' in self.data.columns:
            result['avg_sales_by_day'] = self.data.groupby('day_of_week', observed=True)['sales'].mean().astype(float).to_dict()
            peak_day = self.data.groupby('day_of_week', observed=True)['sales'].mean().idxmax()
            result['peak_sales_day'] = peak_day
            
        if 'month_name' in self.data.columns and 'sales' in self.data.columns:
            result['seasonal_patterns'] = self.data.groupby('month_name', observed=True)['sales'].mean().astype(float).to_dict()
            
        return result