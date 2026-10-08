from typing import Dict, Any


class ExplanationGenerator:
    """
    Synthesizes controlled structured classification + recommendations
    into natural, warm, personalized explanatory text in French.
    """

    def generate_natural_explanation(self, classification: Dict[str, Any], recommendations: Dict[str, Any]) -> str:
        primary = classification.get('primary_skin_type', 'Combination')
        sensitivity = classification.get('sensitivity', 'Low')

        # French skin type labels
        types_fr = {
            'Combination': 'Mixte',
            'Oily': 'Grasse',
            'Dry': 'Sèche',
            'Normal': 'Normale',
        }
        sensi_fr = {
            'High': 'Élevée',
            'Medium': 'Moyenne',
            'Low': 'Faible',
        }
        primary_fr = types_fr.get(primary, primary)
        sensit_fr = sensi_fr.get(sensitivity, sensitivity)

        intro = f"Sur la base de notre analyse faciale bi-angulaire et de votre questionnaire de routine personnel, votre profil de peau est identifié comme **{primary_fr}** avec une **Sensibilité {sensit_fr}**."

        if primary == 'Combination':
            details = "Notre cartographie des zones faciales a observé un brillant plus élevé et une production active de sébum le long de votre zone T (front et nez), en contraste avec des zones de joues plus sèches ou équilibrées. C'est un type de peau courant qui bénéficie de soins ciblés par zone."
        elif primary == 'Oily':
            details = "Le scan visuel a détecté un reflet élevé sur plusieurs zones du visage ainsi qu'une structure de pores visible. Une hydratation légère et non grasse aidera à garder votre peau fraîche sans obstruer les pores."
        elif primary == 'Dry':
            details = "Le scan a indiqué un reflet d'humidité réduit et une variance de texture de surface fine sur vos joues et votre front. Des crèmes riches en lipides et des nettoyants doux non moussants aideront à restaurer la douceur naturelle de votre peau."
        else:
            details = "Votre peau affiche une répartition bien équilibrée de l'humidité et du sébum naturel sur toutes les zones du visage. Maintenir cet équilibre avec une protection antioxydante et des soins solaires quotidiens est votre objectif principal."

        sens_note = ""
        if sensitivity == 'High':
            sens_note = " Comme votre peau montre des signes de sensibilité élevée, nous recommandons des ingrédients botaniques apaisants comme le Centella Asiatica et de d'éviter les parfums artificiels ou les exfoliations agressives."
        elif sensitivity == 'Medium':
            sens_note = " Votre peau présente une réactivité modérée, il est donc recommandé d'introduire progressivement de nouveaux ingrédients actifs."

        return f"{intro} {details}{sens_note}"
