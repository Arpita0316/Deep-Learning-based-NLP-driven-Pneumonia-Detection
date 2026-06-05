"""
src/nlp_engine.py — NLP-Driven Clinical Explanation Engine

Generates structured radiology-style reports combining:
- Rule-based medical language templates (confidence/uncertainty-aware)
- Dynamic clinical recommendations
- Uncertainty quantification narration
"""

import random
import textwrap
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ClinicalReport:
    patient_id:            str
    timestamp:             str
    study_type:            str
    ai_diagnosis:          str
    confidence_pct:        str
    uncertainty_pct:       str
    predictive_entropy:    str
    confidence_tier:       str
    uncertainty_level:     str
    radiological_findings: str
    recommendations:       List[str]
    uncertainty_note:      str
    disclaimer:            str = (
        "AI-generated report. For research/educational use only. "
        "Not intended to replace qualified clinical judgment."
    )


class ClinicalExplanationEngine:
    """
    NLP Clinical Explanation Engine.

    Usage:
        engine = ClinicalExplanationEngine()
        report = engine.generate_report(
            prediction='PNEUMONIA',
            confidence=0.94,
            uncertainty=0.03,
            entropy=0.18,
            patient_id='PT-001'
        )
        print(engine.format_report(report))
    """

    FINDINGS_TEMPLATES = {
        'high_pneumonia': [
            (
                "Chest radiograph demonstrates consolidative opacification with air bronchograms "
                "consistent with lobar or segmental pneumonia. Increased interstitial markings "
                "are noted in the affected region. The cardiac silhouette is within normal limits."
            ),
            (
                "There is patchy airspace disease with heterogeneous opacification, suggestive of "
                "pneumonic consolidation. The distribution of opacity is consistent with community-"
                "acquired pneumonia. Possible pleural reaction cannot be excluded."
            ),
            (
                "Radiographic findings reveal perihilar infiltrates with airspace consolidation. "
                "The lung parenchyma shows heterogeneous opacification consistent with an acute "
                "infectious pulmonary process. No pneumothorax identified."
            ),
            (
                "Dense consolidation is identified occupying a lobar or sublobar distribution. "
                "The density and morphology of the opacity are consistent with bacterial or viral "
                "pneumonia in the appropriate clinical setting."
            ),
        ],
        'mid_pneumonia': [
            (
                "Subtle increased opacity is identified in the lung parenchyma. The findings are "
                "indeterminate and may represent an early or evolving pneumonic process versus "
                "atelectasis. Clinical correlation is strongly advised."
            ),
            (
                "Mild perihilar haziness is observed bilaterally. While non-specific, these findings "
                "in the appropriate clinical context may represent early pneumonic infiltrate. "
                "Comparison with prior radiographs, if available, is recommended."
            ),
        ],
        'high_normal': [
            (
                "The lungs are clear bilaterally with no focal consolidation, pleural effusion, "
                "or pneumothorax. The cardiomediastinal silhouette is within normal limits. "
                "Bony thorax appears intact without acute osseous abnormality."
            ),
            (
                "No acute cardiopulmonary abnormality is identified. Lung fields appear "
                "well-aerated without evidence of airspace infiltrate or pleural effusion. "
                "Trachea is midline. Cardiac size is normal."
            ),
            (
                "Clear lung fields bilaterally. No consolidation, atelectasis, or pleural disease "
                "identified. Pulmonary vascularity appears normal. The hemidiaphragms are sharp."
            ),
        ],
        'uncertain': [
            (
                "The radiographic findings are equivocal. High model uncertainty was detected for "
                "this image, limiting diagnostic confidence. Recommend expert radiologist review "
                "and thorough clinical correlation before any management decisions are made."
            ),
        ]
    }

    RECOMMENDATIONS = {
        'PNEUMONIA': {
            'high': [
                "Initiate antibiotic therapy per institutional antimicrobial stewardship protocol.",
                "Obtain sputum Gram stain and culture prior to antibiotic administration if feasible.",
                "Monitor oxygen saturation; initiate supplemental oxygen therapy if SpO₂ < 94%.",
                "Consider CT chest for further characterisation if clinical response is inadequate.",
                "Repeat chest X-ray in 4–6 weeks post-treatment to confirm radiographic resolution.",
                "Assess pneumonia severity index (PSI/PORT or CURB-65) to guide hospitalisation decision.",
            ],
            'mid': [
                "Clinical correlation with presenting symptoms (fever, productive cough, dyspnoea) required.",
                "Consider repeat chest radiograph in 24–48 hours if symptoms persist or worsen.",
                "Laboratory workup including CBC with differential, CRP, and procalcitonin may be informative.",
                "Pulse oximetry monitoring recommended.",
            ],
            'low': [
                "Expert radiologist review strongly recommended before any clinical action.",
                "Repeat imaging may be warranted to clarify equivocal findings.",
            ]
        },
        'NORMAL': {
            'high': [
                "No acute radiographic abnormality detected.",
                "Routine clinical follow-up as indicated by patient symptoms.",
                "If respiratory symptoms persist, consider alternative diagnoses and further workup.",
            ],
            'mid': [
                "No definitive radiographic abnormality identified.",
                "Clinical correlation with presenting symptoms is advised.",
                "Consider repeat imaging or further investigation if clinically warranted.",
            ],
            'low': [
                "Expert review recommended given moderate model uncertainty.",
                "Consider repeat or advanced imaging as clinically indicated.",
            ]
        }
    }

    UNCERTAINTY_WARNINGS = {
        'low':    "✅ Model confidence is HIGH. Prediction reliability is strong.",
        'medium': "⚡ Model confidence is MODERATE. Clinical correlation is recommended.",
        'high':   (
            "⚠️  HIGH UNCERTAINTY DETECTED. This prediction should NOT be used for "
            "clinical decision-making without expert radiologist review."
        )
    }

    def generate_report(
        self,
        prediction: str,
        confidence: float,
        uncertainty: float,
        entropy: float,
        patient_id: str = 'N/A'
    ) -> ClinicalReport:
        """
        Generate a ClinicalReport object.

        Args:
            prediction : 'NORMAL' or 'PNEUMONIA'
            confidence : float in [0, 1]  — predicted class probability
            uncertainty: float in [0, 1]  — MC dropout std for predicted class
            entropy    : float            — predictive entropy
            patient_id : str              — optional patient identifier
        """
        conf_tier = (
            'high' if confidence > 0.85 else
            'mid'  if confidence > 0.65 else
            'low'
        )
        unc_tier = (
            'low'    if uncertainty < 0.05 else
            'medium' if uncertainty < 0.15 else
            'high'
        )

        # Select findings
        if unc_tier == 'high':
            finding_key = 'uncertain'
        elif prediction == 'PNEUMONIA' and conf_tier in ('high', 'mid'):
            finding_key = f'{conf_tier}_pneumonia'
        else:
            finding_key = 'high_normal'

        templates = self.FINDINGS_TEMPLATES.get(
            finding_key, self.FINDINGS_TEMPLATES['uncertain']
        )
        findings = random.choice(templates)
        recs     = self.RECOMMENDATIONS.get(prediction, {}).get(conf_tier, [])

        return ClinicalReport(
            patient_id=patient_id,
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            study_type='Chest Radiograph — Posteroanterior (PA) View',
            ai_diagnosis=prediction,
            confidence_pct=f'{confidence * 100:.2f}%',
            uncertainty_pct=f'{uncertainty * 100:.2f}%',
            predictive_entropy=f'{entropy:.4f}',
            confidence_tier=conf_tier.upper(),
            uncertainty_level=unc_tier.upper(),
            radiological_findings=findings,
            recommendations=recs,
            uncertainty_note=self.UNCERTAINTY_WARNINGS[unc_tier],
        )

    def format_report(self, report: ClinicalReport) -> str:
        """Render report as a formatted clinical document string."""
        W = 68
        divider = '─' * W

        lines = [
            divider,
            '  AUTOMATED CHEST X-RAY ANALYSIS REPORT',
            '  PneumoniaNet — EfficientNetB3 + NLP Clinical Engine',
            divider,
            f"  Patient ID   : {report.patient_id}",
            f"  Study        : {report.study_type}",
            f"  Report Date  : {report.timestamp}",
            divider,
            '  DIAGNOSTIC RESULT',
            divider,
            f"  AI Diagnosis : {report.ai_diagnosis}",
            f"  Confidence   : {report.confidence_pct}  (Tier: {report.confidence_tier})",
            f"  Uncertainty  : {report.uncertainty_pct}  (Level: {report.uncertainty_level})",
            f"  Entropy      : {report.predictive_entropy}",
            '',
            f"  {report.uncertainty_note}",
            divider,
            '  RADIOLOGICAL FINDINGS',
            divider,
        ]

        # Wrap findings text
        for line in textwrap.wrap(report.radiological_findings, width=W - 4):
            lines.append(f'  {line}')

        lines += [divider, '  CLINICAL RECOMMENDATIONS', divider]

        for i, rec in enumerate(report.recommendations, 1):
            wrapped = textwrap.wrap(rec, width=W - 7)
            lines.append(f'  {i}. {wrapped[0]}')
            for continuation in wrapped[1:]:
                lines.append(f'     {continuation}')

        lines += [
            divider,
            '  ⚠️  DISCLAIMER',
        ]
        for line in textwrap.wrap(report.disclaimer, width=W - 4):
            lines.append(f'  {line}')
        lines.append(divider)

        return '\n'.join(lines)

    def to_dict(self, report: ClinicalReport) -> dict:
        """Serialize report to a plain dictionary."""
        return {
            'patient_id':            report.patient_id,
            'timestamp':             report.timestamp,
            'study_type':            report.study_type,
            'ai_diagnosis':          report.ai_diagnosis,
            'confidence_pct':        report.confidence_pct,
            'uncertainty_pct':       report.uncertainty_pct,
            'predictive_entropy':    report.predictive_entropy,
            'confidence_tier':       report.confidence_tier,
            'uncertainty_level':     report.uncertainty_level,
            'radiological_findings': report.radiological_findings,
            'recommendations':       report.recommendations,
            'uncertainty_note':      report.uncertainty_note,
            'disclaimer':            report.disclaimer,
        }
