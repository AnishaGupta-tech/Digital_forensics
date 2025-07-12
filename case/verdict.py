import joblib
from jinja2 import Template

class DocumentVerdict:
    def __init__(self):
        self.model = joblib.load('ai_models/ensemble_verdict.pkl')
        
    def generate(self, sig_score, ink_score, meta_score):
        prob = self.model.predict_proba([[sig_score, ink_score, meta_score]])[0][1]
        
        return {
            'forgery_probability': float(prob),
            'confidence': int((1 - prob) * 100),
            'official_verdict': "DOCUMENT_AUTHENTIC" if prob < 0.1 else "DOCUMENT_FORGED",
            'legal_implications': self._get_legal_implications(prob),
            'expert_opinion': self._generate_expert_opinion(prob)
        }
    
    def _generate_expert_opinion(self, prob):
        if prob < 0.1:
            return "The document shows no signs of forgery based on comprehensive forensic analysis."
        elif prob < 0.5:
            return "The document exhibits some anomalous characteristics requiring further examination."
        else:
            return "The document shows clear signs of forgery across multiple forensic parameters."
    
    def generate_pdf_report(self, analysis_data):
        template = Template(open('templates/forensic_report.html').read())
        html = template.render(**analysis_data)
        
        # Use weasyprint or similar to convert to PDF
        from weasyprint import HTML
        HTML(string=html).write_pdf(f"reports/{analysis_data['report_id']}.pdf")
        
        return f"reports/{analysis_data['report_id']}.pdf"