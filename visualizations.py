import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from typing import Dict, Any
from config import Config

class DataVisualizer:
    """Create all interactive data visualizations"""
    
    def __init__(self, data: pd.DataFrame, processed_data: Dict[str, Any]):
        self.data = data
        self.processed_data = processed_data
        self.config = Config.get_chart_config()
    
    def create_sales_trend_chart(self) -> go.Figure:
        """Create sales trend over time"""
        if 'date' not in self.data.columns or 'sales' not in self.data.columns:
            return self._create_empty_chart("Sales data not available")
            
        monthly_sales = self.data.groupby(
            self.data['date'].dt.to_period('M')
        )['sales'].sum().reset_index()
        monthly_sales['date'] = monthly_sales['date'].astype(str)
        
        fig = px.line(
            monthly_sales,
            x='date',
            y='sales',
            title='Sales Trend Over Time',
            labels={'sales': 'Sales ($)', 'date': 'Month'}
        )
        fig.update_traces(line=dict(width=3))
        fig.update_layout(
            height=self.config['height'],
            hovermode='x unified',
            xaxis_title="Month",
            yaxis_title="Sales ($)"
        )
        return fig
    
    def create_product_performance_chart(self) -> go.Figure:
        """Create product performance comparison"""
        if 'product' not in self.data.columns or 'sales' not in self.data.columns:
            return self._create_empty_chart("Product data not available")
            
        product_data = self.data.groupby('product')['sales'].sum().sort_values(ascending=True)
        
        fig = px.bar(
            x=product_data.values,
            y=product_data.index,
            orientation='h',
            title='Product Performance by Sales',
            labels={'x': 'Sales ($)', 'y': 'Product'},
            color=product_data.values,
            color_continuous_scale='Blues'
        )
        fig.update_layout(
            height=self.config['height'],
            showlegend=False
        )
        return fig
    
    def create_regional_analysis_chart(self) -> go.Figure:
        """Create regional analysis pie chart"""
        if 'region' not in self.data.columns or 'sales' not in self.data.columns:
            return self._create_empty_chart("Regional data not available")
            
        regional_data = self.data.groupby('region')['sales'].sum()
        
        fig = px.pie(
            values=regional_data.values,
            names=regional_data.index,
            title='Sales Distribution by Region',
            hole=0.3,
            color_discrete_sequence=self.config['color_palette']
        )
        fig.update_traces(
            textposition='inside',
            textinfo='percent+label'
        )
        fig.update_layout(height=self.config['height'])
        return fig
    
    def create_customer_demographics_chart(self) -> go.Figure:
        """Create customer demographics analysis"""
        if 'age_group' not in self.data.columns or 'sales' not in self.data.columns:
            return self._create_empty_chart("Demographics data not available")
        
        if 'customer_gender' in self.data.columns:
            # Gender and age analysis
            demo_data = self.data.groupby(['age_group', 'customer_gender'])['sales'].sum().reset_index()
            
            fig = px.bar(
                demo_data,
                x='age_group',
                y='sales',
                color='customer_gender',
                title='Sales by Age Group and Gender',
                labels={'sales': 'Total Sales ($)', 'age_group': 'Age Group'},
                barmode='group',
                color_discrete_sequence=self.config['color_palette']
            )
        else:
            # Age analysis only
            age_data = self.data.groupby('age_group')['sales'].sum().reset_index()
            
            fig = px.bar(
                age_data,
                x='age_group',
                y='sales',
                title='Sales by Age Group',
                labels={'sales': 'Total Sales ($)', 'age_group': 'Age Group'},
                color='sales',
                color_continuous_scale='Viridis'
            )
        
        fig.update_layout(height=self.config['height'])
        return fig
    
    def create_satisfaction_analysis_chart(self) -> go.Figure:
        """Create customer satisfaction analysis"""
        if 'customer_satisfaction' not in self.data.columns or 'sales' not in self.data.columns:
            return self._create_empty_chart("Satisfaction data not available")
            
        satisfaction_data = self.data.groupby('customer_satisfaction')['sales'].sum().reset_index()
        
        fig = px.scatter(
            satisfaction_data,
            x='customer_satisfaction',
            y='sales',
            size='sales',
            title='Sales vs Customer Satisfaction',
            labels={
                'customer_satisfaction': 'Customer Satisfaction Score',
                'sales': 'Total Sales ($)'
            },
            color='sales',
            color_continuous_scale='RdYlGn'
        )
        fig.update_layout(height=self.config['height'])
        return fig
    
    def create_temporal_patterns_chart(self) -> go.Figure:
        """Create temporal patterns analysis"""
        if 'day_of_week' not in self.data.columns or 'sales' not in self.data.columns:
            return self._create_empty_chart("Temporal data not available")
            
        # Define day order
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        daily_patterns = self.data.groupby('day_of_week')['sales'].mean()
        
        # Reorder by day of week
        daily_patterns = daily_patterns.reindex([day for day in day_order if day in daily_patterns.index])
        
        fig = px.bar(
            x=daily_patterns.index,
            y=daily_patterns.values,
            title='Average Sales by Day of Week',
            labels={'x': 'Day of Week', 'y': 'Average Sales ($)'},
            color=daily_patterns.values,
            color_continuous_scale='Blues'
        )
        fig.update_layout(height=self.config['height'], showlegend=False)
        return fig
    
    def create_correlation_heatmap(self) -> go.Figure:
        """Create correlation heatmap for numerical variables"""
        numerical_cols = []
        if 'sales' in self.data.columns:
            numerical_cols.append('sales')
        if 'customer_age' in self.data.columns:
            numerical_cols.append('customer_age')
        if 'customer_satisfaction' in self.data.columns:
            numerical_cols.append('customer_satisfaction')
            
        if len(numerical_cols) < 2:
            return self._create_empty_chart("Insufficient numerical data for correlation")
            
        correlation_matrix = self.data[numerical_cols].corr()
        
        fig = px.imshow(
            correlation_matrix,
            text_auto=True,
            aspect="auto",
            title="Correlation Matrix: Key Variables",
            color_continuous_scale='RdBu'
        )
        fig.update_layout(height=self.config['height'])
        return fig
    
    def _create_empty_chart(self, message: str) -> go.Figure:
        """Create an empty chart with a message"""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            xanchor='center', yanchor='middle',
            showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(
            height=self.config['height'],
            title="Chart Not Available"
        )
        return fig