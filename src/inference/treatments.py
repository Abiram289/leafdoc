"""
Agronomic treatment recommendation engine for PlantVillage disease classes.
Maps disease diagnoses and severity ratings to actionable cultural, chemical,
and biological remedies.
"""
import json
from pathlib import Path
from typing import Any, Dict, Optional


class TreatmentRecommender:
    """
    Recommends agronomic treatments and management practices based on
    fine-grained disease classification and severity estimation.
    """

    def __init__(self, treatments_path: Optional[str] = None):
        if treatments_path is None:
            # Default to data/treatments.json relative to project root
            base_dir = Path(__file__).resolve().parent.parent.parent
            treatments_path = base_dir / "data" / "treatments.json"

        self.treatments_path = Path(treatments_path)
        if not self.treatments_path.exists():
            raise FileNotFoundError(f"Treatments database not found at: {self.treatments_path}")

        with open(self.treatments_path, "r", encoding="utf-8") as f:
            self.database: Dict[str, Dict[str, Any]] = json.load(f)

    @property
    def treatments(self) -> Dict[str, Dict[str, Any]]:
        """Dictionary access alias to treatments database."""
        return self.database

    def get_recommendation(self, fine_label: str, severity: Optional[str] = None) -> Dict[str, Any]:
        """
        Retrieves treatment recommendations for a given fine-grained class label.
        
        Args:
            fine_label: 38-class label (e.g. 'Tomato___Early_blight' or 'Apple___healthy')
            severity: Optional severity rating ('Mild', 'Moderate', 'Severe', or 'Healthy')
            
        Returns:
            Dictionary containing disease details, controls, and severity-specific actions.
        """
        if fine_label not in self.database:
            # Fallback for unknown classes
            species = fine_label.split("___")[0].replace("_", " ") if "___" in fine_label else "Unknown"
            disease = fine_label.split("___")[1].replace("_", " ") if "___" in fine_label else fine_label
            return {
                "fine_label": fine_label,
                "species": species,
                "condition_name": disease,
                "pathogen_type": "Unknown",
                "description": f"Diagnosis: {disease} on {species}.",
                "cultural_controls": ["Isolate affected plants and improve canopy ventilation."],
                "chemical_controls": ["Consult a local agricultural extension officer for approved fungicides."],
                "biological_controls": ["Apply bio-fungicides such as Bacillus subtilis or neem oil."],
                "preventive_measures": ["Practice crop rotation and avoid overhead watering."],
                "severity_level": severity or "Unknown",
                "immediate_action": "Inspect surrounding plants and monitor progress.",
            }

        data = self.database[fine_label].copy()
        data["fine_label"] = fine_label
        
        # Normalize severity action
        severity_actions = data.get("severity_actions", {})
        if severity and severity in severity_actions:
            data["immediate_action"] = severity_actions[severity]
        elif "Healthy" in severity_actions and "healthy" in fine_label.lower():
            data["immediate_action"] = severity_actions["Healthy"]
        elif "Moderate" in severity_actions:
            data["immediate_action"] = severity_actions["Moderate"]
        else:
            data["immediate_action"] = list(severity_actions.values())[0] if severity_actions else "Monitor plant."

        data["severity_level"] = severity or ("Healthy" if "healthy" in fine_label.lower() else "Moderate")
        return data

    def format_text_summary(self, fine_label: str, severity: Optional[str] = None) -> str:
        """Formats the recommendation as a readable multi-line string for CLI output."""
        rec = self.get_recommendation(fine_label, severity)
        lines = []
        lines.append(f"Condition : {rec['condition_name']} ({rec['species']})")
        lines.append(f"Pathogen  : {rec['pathogen_type']}")
        lines.append(f"Severity  : {rec['severity_level']}")
        lines.append(f"Summary   : {rec['description']}")
        lines.append(f"\nImmediate Action Plan [{rec['severity_level']}]:")
        lines.append(f"  * {rec['immediate_action']}")
        
        if rec.get("cultural_controls"):
            lines.append("\nCultural Management:")
            for item in rec["cultural_controls"]:
                lines.append(f"  - {item}")

        if rec.get("chemical_controls") and rec["pathogen_type"] != "None":
            lines.append("\nChemical Controls:")
            for item in rec["chemical_controls"]:
                lines.append(f"  - {item}")

        if rec.get("biological_controls") and rec["pathogen_type"] != "None":
            lines.append("\nBiological & Organic Controls:")
            for item in rec["biological_controls"]:
                lines.append(f"  - {item}")

        if rec.get("preventive_measures"):
            lines.append("\nPreventive Measures:")
            for item in rec["preventive_measures"]:
                lines.append(f"  - {item}")

        return "\n".join(lines)
