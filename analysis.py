from case.signature.signature import SignatureVerifier
from case.ink.ink import InkAnalyzer
from case.metadata.metadata import MetadataInspector
from case.paper.paper import PaperAnalyzer
from case.ela.ela import ELAAnalyzer
from case.verdict import DocumentVerdict
import datetime
import logging

class ForensicAnalysis:
    def __init__(self):
        """Initialize all forensic analysis modules with trained models"""
        self.logger = logging.getLogger('ForensicAnalysis')
        self.models_loaded = False
        
        try:
            # Initialize all analysis modules with their respective trained models
            self.signature = SignatureVerifier(model_path='ai_models/signature_model.pkl')
            self.ink = InkAnalyzer(model_path='ai_models/ink_model.pkl')
            self.meta = MetadataInspector(model_path='ai_models/metadata_model.pkl')
            self.paper = PaperAnalyzer(model_path='ai_models/paper_model.pkl')
            self.ela = ELAAnalyzer(model_path='ai_models/ela_model.pkl')
            self.verdict = DocumentVerdict()
            
            self.models_loaded = True
            self.logger.info("All forensic models loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Model initialization failed: {str(e)}")
            raise RuntimeError(f"Analysis initialization failed: {str(e)}")

    def full_analysis(self, file_path):
        """
        Perform complete forensic analysis on a document
        Args:
            file_path: Path to the document to analyze
        Returns:
            Dictionary containing all analysis results and final verdict
        """
        if not self.models_loaded:
            return {
                'error': "Models not loaded properly",
                'verdict': 'ANALYSIS_FAILED'
            }

        try:
            analysis_start = datetime.datetime.now()
            self.logger.info(f"Starting analysis of {file_path}")
            
            # Run all analyses with individual error handling
            results = {
                'report_id': f"FSCAN-{analysis_start.strftime('%Y%m%d-%H%M%S')}",
                'document': file_path,
                'analysis_date': str(analysis_start),
                'signature_analysis': self._safe_analyze(self.signature.verify, file_path),
                'ink_analysis': self._safe_analyze(self.ink.analyze, file_path),
                'metadata_analysis': self._safe_analyze(self.meta.analyze, file_path),
                'paper_analysis': self._safe_analyze(self.paper.analyze, file_path),
                'ela_analysis': self._safe_analyze(self.ela.analyze, file_path)
            }
            
            # Generate final verdict using all available scores
            verdict_scores = {
                'signature': results['signature_analysis'].get('authenticity_score', 0),
                'ink': results['ink_analysis'].get('consistency_score', 0),
                'metadata': results['metadata_analysis'].get('tamper_probability', 0),
                'paper': results['paper_analysis'].get('authenticity_score', 0),
                'ela': results['ela_analysis'].get('manipulation_score', 0)
            }
            
            final_verdict = self.verdict.generate(**verdict_scores)
            
            # Format results for frontend
            formatted_results = {
                'report_id': results['report_id'],
                'analysis_date': results['analysis_date'],
                'signature_analysis': {
                    'score': results['signature_analysis'].get('authenticity_score', 0),
                    'verdict': results['signature_analysis'].get('verdict', 'UNKNOWN'),
                    'metrics': {
                        'stroke_variation': results['signature_analysis'].get('stroke_variation', 0),
                        'pressure_consistency': results['signature_analysis'].get('pressure_consistency', 0),
                        'speed_variation': results['signature_analysis'].get('speed_variation', 0)
                    }
                },
                'ink_analysis': {
                    'score': results['ink_analysis'].get('consistency_score', 0),
                    'verdict': results['ink_analysis'].get('verdict', 'UNKNOWN'),
                    'metrics': {
                        'ink_types': results['ink_analysis'].get('ink_types', 1),
                        'density_variation': results['ink_analysis'].get('density_variation', 0),
                        'chemical_match': results['ink_analysis'].get('chemical_match', 0)
                    }
                },
                'digital_analysis': {
                    'score': 1 - ((results['metadata_analysis'].get('tamper_probability', 0) + 
                                 results['ela_analysis'].get('manipulation_score', 0)) / 2),
                    'verdict': results['metadata_analysis'].get('verdict', 'UNKNOWN'),
                    'metrics': {
                        'tamper_prob': results['metadata_analysis'].get('tamper_probability', 0),
                        'manipulation_score': results['ela_analysis'].get('manipulation_score', 0),
                        'compression': 1 if results['ela_analysis'].get('compression_artifacts', False) else 0
                    }
                },
                'paper_analysis': {
                    'score': results['paper_analysis'].get('authenticity_score', 0),
                    'verdict': results['paper_analysis'].get('verdict', 'UNKNOWN'),
                    'metrics': {
                        'fiber_match': 0 if results['paper_analysis'].get('fiber_inconsistency', 1) > 0.4 else 1,
                        'watermark_match': 0 if results['paper_analysis'].get('watermark_mismatch', False) else 1,
                        'age_consistency': results['paper_analysis'].get('age_consistency', 0)
                    }
                },
                'final_verdict': {
                    'verdict': final_verdict,
                    'confidence': (
                        (
                            results['signature_analysis'].get('authenticity_score', 0) +
                            results['ink_analysis'].get('consistency_score', 0) +
                            (1 - results['metadata_analysis'].get('tamper_probability', 0)) +
                            results['paper_analysis'].get('authenticity_score', 0)
                        ) / 4
                    ),
                    'indicators': self._get_indicators(results)
                }
            }
            
            analysis_time = (datetime.datetime.now() - analysis_start).total_seconds()
            formatted_results['analysis_duration_sec'] = round(analysis_time, 2)
            self.logger.info(f"Analysis completed in {analysis_time:.2f} seconds")
            
            return formatted_results
            
        except Exception as e:
            self.logger.error(f"Analysis failed: {str(e)}")
            return {
                'error': str(e),
                'verdict': 'ANALYSIS_FAILED'
            }
    
    def _safe_analyze(self, analyzer_func, file_path):
        """Wrapper for safe analysis with error handling"""
        try:
            return analyzer_func(file_path)
        except Exception as e:
            self.logger.warning(f"Partial analysis failed: {str(e)}")
            return {
                'error': str(e),
                'verdict': 'ANALYSIS_PARTIAL_FAILURE'
            }
    
    def _get_indicators(self, results):
        """Compile forensic indicators from all analyses"""
        indicators = []
        
        try:
            # Signature indicators
            sig = results.get('signature_analysis', {})
            if sig.get('verdict') == "FORGED":
                indicators.append("Signature shows signs of forgery")
            elif sig.get('verdict') == "SUSPECT":
                indicators.append("Signature shows minor anomalies")
                
            # Ink indicators
            ink = results.get('ink_analysis', {})
            if ink.get('ink_types', 0) > 1:
                indicators.append(f"Multiple ink types detected ({ink['ink_types']})")
            if ink.get('density_variation', 0) > 0.3:  # Fixed typo: variation -> variation
                indicators.append("High ink density variation detected")
                
            # Metadata indicators
            meta = results.get('metadata_analysis', {})
            if meta.get('verdict') == "TAMPERED":
                indicators.append("Metadata shows signs of tampering")
            if meta.get('software_indicators'):
                indicators.append(f"Editing software detected: {', '.join(meta['software_indicators'])}")
                
            # Paper indicators
            paper = results.get('paper_analysis', {})
            if paper.get('fiber_inconsistency', 0) > 0.4:
                indicators.append("Paper fiber inconsistencies detected")
            if paper.get('watermark_mismatch', False):
                indicators.append("Watermark doesn't match expected pattern")
                
            # ELA indicators
            ela = results.get('ela_analysis', {})
            if ela.get('manipulation_score', 0) > 0.7:
                indicators.append("High probability of digital manipulation")
            if ela.get('compression_artifacts', False):
                indicators.append("Inconsistent compression artifacts detected")
                
        except Exception as e:
            self.logger.error(f"Indicator compilation failed: {str(e)}")
            
        return indicators