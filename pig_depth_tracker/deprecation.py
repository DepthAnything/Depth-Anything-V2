# pig_depth_tracker/depreciation.py
import numpy as np
from scipy import stats
from typing import Dict, List, Optional
from collections import deque
import pandas as pd

class DepreciationAnalyzer:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {
            'temporal_window': 15,  # frames to consider for temporal analysis
            'depth_change_threshold': 0.05,  # significant depth change
            'asymmetry_threshold': 0.25,  # significant asymmetry
            'min_consistent_frames': 5  # min frames for consistent pattern
        }
        self.reference_model = None
        self.temporal_buffer = deque(maxlen=self.config['temporal_window'])
        
    def build_reference_model(self, depth_analyses: List[Dict]):
        """Build reference model from initial healthy frames"""
        if not depth_analyses:
            return None
            
        # Collect all depth values
        all_depths = np.concatenate([da['depth_values'] for da in depth_analyses if da])
        
        # Calculate reference statistics
        self.reference_model = {
            'mean': float(np.mean(all_depths)),
            'std': float(np.std(all_depths)),
            'percentiles': {
                '5': float(np.percentile(all_depths, 5)),
                '25': float(np.percentile(all_depths, 25)),
                '50': float(np.percentile(all_depths, 50)),
                '75': float(np.percentile(all_depths, 75)),
                '95': float(np.percentile(all_depths, 95))
            },
            'regional_means': {
                'head': float(np.mean([da.get('head_mean', np.nan) for da in depth_analyses])),
                'middle': float(np.mean([da.get('middle_mean', np.nan) for da in depth_analyses])),
                'rear': float(np.mean([da.get('rear_mean', np.nan) for da in depth_analyses]))
            }
        }
        return self.reference_model
    
    def detect_depreciation(self, current_analysis: Dict, frame_idx: int) -> Dict:
        """Main detection method combining multiple analyses"""
        if not current_analysis:
            return {'depreciation_score': 0.0, 'confidence': 0.0}
            
        # Update temporal buffer
        self.temporal_buffer.append(current_analysis)
        
        # Run individual analyses
        depth_dev = self._analyze_depth_deviation(current_analysis)
        temp_consistency = self._check_temporal_consistency(current_analysis, frame_idx)
        regional = self._analyze_regions(current_analysis)
        symmetry = self._analyze_symmetry(current_analysis)
        
        # Combine results into overall score (0-1)
        depreciation_score = 0.4 * depth_dev['score'] + \
                           0.3 * temp_consistency['score'] + \
                           0.2 * regional['score'] + \
                           0.1 * symmetry['score']
        
        # Build results dictionary
        results = {
            'depreciation_score': float(depreciation_score),
            'confidence': min(1.0, len(self.temporal_buffer)/self.config['temporal_window']),
            'depth_deviation': depth_dev,
            'temporal_consistency': temp_consistency,
            'regional_analysis': regional,
            'symmetry_analysis': symmetry,
            'frame_index': frame_idx
        }
        
        return results
    
    def _analyze_depth_deviation(self, analysis: Dict) -> Dict:
        """Compare current depth to reference model"""
        if not self.reference_model:
            return {'score': 0.0, 'message': 'No reference model'}
            
        # Overall depth comparison
        depth_diff = abs(analysis['mean_depth'] - self.reference_model['mean'])
        depth_score = min(1.0, depth_diff / (self.reference_model['std'] * 2))
        
        # Percentile comparison
        p5_diff = abs(analysis['percentiles']['5'] - self.reference_model['percentiles']['5'])
        p95_diff = abs(analysis['percentiles']['95'] - self.reference_model['percentiles']['95'])
        percentile_score = min(1.0, (p5_diff + p95_diff) / (self.reference_model['std'] * 3))
        
        return {
            'score': max(depth_score, percentile_score),
            'depth_difference': depth_diff,
            'percentile_differences': {'p5': p5_diff, 'p95': p95_diff},
            'message': f"Depth deviation: {depth_diff:.3f} (ref: {self.reference_model['mean']:.3f})"
        }
    
    def _check_temporal_consistency(self, analysis: Dict, frame_idx: int) -> Dict:
        """Check if anomalies persist over time"""
        if len(self.temporal_buffer) < 2:
            return {'score': 0.0, 'message': 'Insufficient temporal data'}
            
        # Calculate rolling mean and std
        means = [a['mean_depth'] for a in self.temporal_buffer if a]
        stds = [a['std_depth'] for a in self.temporal_buffer if a]
        
        rolling_mean = np.mean(means)
        rolling_std = np.std(means)
        
        # Check for consistent deviation
        current_dev = abs(analysis['mean_depth'] - rolling_mean)
        consistent_frames = sum(1 for m in means if abs(m - rolling_mean) > self.config['depth_change_threshold']))
        
        consistency_score = min(1.0, consistent_frames / self.config['min_consistent_frames'])
        
        return {
            'score': consistency_score,
            'rolling_mean': rolling_mean,
            'rolling_std': rolling_std,
            'current_deviation': current_dev,
            'consistent_frames': consistent_frames,
            'message': f"{consistent_frames} consistent anomalous frames"
        }
    
    def _analyze_regions(self, analysis: Dict) -> Dict:
        """Analyze regional depth patterns"""
        if not self.reference_model:
            return {'score': 0.0, 'message': 'No reference model'}
            
        regional_scores = {}
        max_score = 0.0
        
        for region in ['head', 'middle', 'rear']:
            current = analysis.get(f'{region}_mean')
            reference = self.reference_model['regional_means'].get(region)
            
            if current is None or reference is None:
                continue
                
            diff = abs(current - reference)
            score = min(1.0, diff / (self.reference_model['std'] * 1.5))
            regional_scores[region] = score
            max_score = max(max_score, score)
            
        return {
            'score': max_score,
            'regional_differences': regional_scores,
            'message': f"Max regional difference: {max_score:.2f}"
        }
    
    def _analyze_symmetry(self, analysis: Dict) -> Dict:
        """Analyze left-right symmetry"""
        symmetry_score = analysis.get('symmetry_score', 0.0)
        return {
            'score': min(1.0, symmetry_score / self.config['asymmetry_threshold']),
            'raw_score': symmetry_score,
            'message': f"Symmetry score: {symmetry_score:.3f} (threshold: {self.config['asymmetry_threshold']})"
        }
    
    def generate_report(self, all_results: List[Dict]) -> pd.DataFrame:
        """Generate comprehensive report from all frame analyses"""
        report_data = []
        
        for result in all_results:
            if not result:
                continue
                
            row = {
                'frame': result['frame_index'],
                'depreciation_score': result['depreciation_score'],
                'confidence': result['confidence'],
                'depth_deviation': result['depth_deviation']['score'],
                'temporal_consistency': result['temporal_consistency']['score'],
                'regional_analysis': result['regional_analysis']['score'],
                'symmetry': result['symmetry_analysis']['score'],
                **{f"region_{k}": v for k, v in result['regional_analysis']['regional_differences'].items()}
            }
            report_data.append(row)
            
        return pd.DataFrame(report_data)