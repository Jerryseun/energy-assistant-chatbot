# metrics_handler.py
from dataclasses import dataclass, field
from collections import deque
from datetime import datetime
from typing import List, Dict
import pandas as pd
import plotly.graph_objects as go

@dataclass
class ChatMetrics:
    response_times: deque = field(default_factory=lambda: deque(maxlen=100))
    token_counts: deque = field(default_factory=lambda: deque(maxlen=100))
    timestamps: deque = field(default_factory=lambda: deque(maxlen=100))
    total_queries: int = 0
    successful_queries: int = 0
    error_count: int = 0

    def add_interaction(self, response_time: float, token_count: int, success: bool):
        self.response_times.append(response_time)
        self.token_counts.append(token_count)
        self.timestamps.append(datetime.now())
        self.total_queries += 1
        if success:
            self.successful_queries += 1
        else:
            self.error_count += 1

    @property
    def success_rate(self) -> float:
        return (self.successful_queries / max(1, self.total_queries)) * 100

    @property
    def average_response_time(self) -> float:
        return sum(self.response_times) / max(1, len(self.response_times))

    @property
    def average_token_count(self) -> float:
        return sum(self.token_counts) / max(1, len(self.token_counts))

    def get_metrics_df(self) -> pd.DataFrame:
        """Get metrics as DataFrame for plotting"""
        return pd.DataFrame({
            'timestamp': list(self.timestamps),
            'response_time': list(self.response_times),
            'token_count': list(self.token_counts)
        })

    def create_metrics_chart(self) -> go.Figure:
        """Create interactive metrics chart"""
        df = self.get_metrics_df()
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['response_time'],
            name='Response Time',
            line=dict(color='#0066cc', width=2),
            hovertemplate='%{y:.2f}s'
        ))
        
        fig.update_layout(
            title='Response Time Trend',
            xaxis_title='Time',
            yaxis_title='Response Time (s)',
            hovermode='x unified',
            height=300,
            margin=dict(l=0, r=0, t=40, b=0),
            template='plotly_white'
        )
        
        return fig