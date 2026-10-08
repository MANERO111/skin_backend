from typing import Dict, Any, List


class RecommendationEngine:
    """
    Generates controlled cosmetic skincare recommendations based on skin profile.
    Separates controlled logic (What is recommended) from natural language generation.
    """

    def generate_recommendations(self, primary_type: str, sensitivity: str, metrics: Dict[str, int]) -> Dict[str, Any]:
        ingredients_to_seek: List[str] = []
        ingredients_to_avoid: List[str] = []
        morning_routine: List[Dict[str, str]] = []
        evening_routine: List[Dict[str, str]] = []
        guidance = ""

        # Base routine according to Primary Skin Type
        if primary_type == 'Combination':
            guidance = "Focus on balancing moisture: hydrators for drier cheek areas while controlling excess T-zone shine."
            ingredients_to_seek = ['Niacinamide (2-5%)', 'Hyaluronic Acid', 'Green Tea Extract', 'Zinc PCA', 'Lightweight Glycerin']
            ingredients_to_avoid = ['Heavy Mineral Oils', 'High Occlusive Petroleum Jel', 'Harsh Alcohol-based Toners']

            morning_routine = [
                {'step': 'Cleanser', 'name': 'Gentle Foaming or Gel Cleanser', 'desc': 'Cleanses without stripping natural moisture barrier.'},
                {'step': 'Treatment', 'name': 'Niacinamide & Hydrating Serum', 'desc': 'Balances sebum in T-zone while plumping cheeks.'},
                {'step': 'Moisturizer', 'name': 'Oil-free Lotion / Gel-Cream', 'desc': 'Provides weightless moisture balance.'},
                {'step': 'Sunscreen', 'name': 'Broad-Spectrum SPF 30+ Fluid', 'desc': 'Non-comedogenic UV protection.'}
            ]

            evening_routine = [
                {'step': 'First Cleanse', 'name': 'Micellar Water or Light Cleansing Oil', 'desc': 'Dissolves daily impurities and sunscreen.'},
                {'step': 'Second Cleanse', 'name': 'Gentle Hydrating Cleanser', 'desc': 'Ensures thorough clean skin canvas.'},
                {'step': 'Exfoliant / Serum', 'name': 'BHA (Salicylic Acid 1-2%) 2x Weekly', 'desc': 'Keeps pores clear in forehead & nose.'},
                {'step': 'Moisturizer', 'name': 'Barrier Repair Lotion', 'desc': 'Nourishes cheek zones overnight.'}
            ]

        elif primary_type == 'Oily':
            guidance = "Aim for sebum regulation and lightweight hydration without clogging pores or over-stripping skin natural oils."
            ingredients_to_seek = ['Salicylic Acid (BHA)', 'Zinc PCA', 'Niacinamide', 'Tea Tree Oil (diluted)', 'Centella Asiatica']
            ingredients_to_avoid = ['Coconut Oil', 'Heavy Emollients', 'Artificial Fragrances', 'Comedogenic Waxes']

            morning_routine = [
                {'step': 'Cleanser', 'name': 'Purifying Gel Cleanser with BHA', 'desc': 'Removes overnight sebum buildup.'},
                {'step': 'Toner / Essence', 'name': 'Pore-Refining Niacinamide Toner', 'desc': 'Tightens pore appearance and controls shine.'},
                {'step': 'Moisturizer', 'name': 'Matte Water Gel Moisturizer', 'desc': 'Hydrates without heavy residue.'},
                {'step': 'Sunscreen', 'name': 'Oil-Control Matte SPF 50', 'desc': 'Keeps skin matte throughout the day.'}
            ]

            evening_routine = [
                {'step': 'Cleanser', 'name': 'Deep Clarifying Cleanser', 'desc': 'Deeply purifies pore channels.'},
                {'step': 'Active Treatment', 'name': 'Salicylic Acid 2% Serum', 'desc': 'Prevents congestion and blemishes.'},
                {'step': 'Night Hydration', 'name': 'Ultra-Lightweight Gel Moisturizer', 'desc': 'Replenishes water balance without oil.'}
            ]

        elif primary_type == 'Dry':
            guidance = "Prioritize intense barrier support, rich lipids, and occlusive hydration to restore soft, comfortable skin elasticity."
            ingredients_to_seek = ['Ceramides (1, 3, 6-II)', 'Squalane', 'Hyaluronic Acid', 'Shea Butter', 'Panthenol (Pro-Vitamin B5)']
            ingredients_to_avoid = ['Physical Scrub Granules', 'Denatured Alcohol', 'Strong Salicylic Acid Wash']

            morning_routine = [
                {'step': 'Cleanser', 'name': 'Non-Foaming Cream Cleanser / Water Rinse', 'desc': 'Preserves delicate lipid barrier.'},
                {'step': 'Hydrating Essence', 'name': 'Multi-Depth Hyaluronic Acid Serum', 'desc': 'Binds moisture to dry epidermal layers.'},
                {'step': 'Moisturizer', 'name': 'Rich Ceramide Cream', 'desc': 'Locks in moisture and prevents transepidermal water loss.'},
                {'step': 'Sunscreen', 'name': 'Hydrating Cream SPF 30+', 'desc': 'Protects while supplying radiant glow.'}
            ]

            evening_routine = [
                {'step': 'Cleanser', 'name': 'Nourishing Balm Cleanser', 'desc': 'Melts away debris while soothing skin.'},
                {'step': 'Treatment', 'name': 'Panthenol & Squalane Facial Oil', 'desc': 'Deeply restores elasticity.'},
                {'step': 'Moisturizer', 'name': 'Intense Overnight Barrier Cream', 'desc': 'Soothes dry patches during sleep.'}
            ]

        else:  # Normal
            guidance = "Maintain your healthy, balanced skin condition with antioxidant protection and consistent daily hydration."
            ingredients_to_seek = ['Vitamin C (THD Ascorbate)', 'Hyaluronic Acid', 'Peptides', 'Green Tea Extract']
            ingredients_to_avoid = ['Over-exfoliating Acids', 'Harsh Soaps']

            morning_routine = [
                {'step': 'Cleanser', 'name': 'Balanced Gentle Wash', 'desc': 'Refreshes skin texture.'},
                {'step': 'Serum', 'name': 'Antioxidant Vitamin C Serum', 'desc': 'Shields against environmental stressors.'},
                {'step': 'Moisturizer', 'name': 'Daily Hydrating Lotion', 'desc': 'Keeps skin smooth and supple.'},
                {'step': 'Sunscreen', 'name': 'Invisible Daily SPF 30+', 'desc': 'Essential daily UV shield.'}
            ]

            evening_routine = [
                {'step': 'Cleanser', 'name': 'Gentle Hydrating Cleanser', 'desc': 'Removes daily buildup.'},
                {'step': 'Serum', 'name': 'Peptide & Barrier Support Serum', 'desc': 'Supports firm, healthy skin architecture.'},
                {'step': 'Moisturizer', 'name': 'Nourishing Night Cream', 'desc': 'Sustains optimal skin hydration.'}
            ]

        # Sensitivity Adjustments
        if sensitivity == 'High':
            guidance += " Since your sensitivity level is High, choose fragrance-free formulations and patch test new products."
            ingredients_to_seek = ['Centella Asiatica (Cica)', 'Madecassoside', 'Allantoin', 'Colloidal Oatmeal'] + ingredients_to_seek
            ingredients_to_avoid = ['Synthetic Fragrance', 'Essential Oils', 'High-concentration AHAs (Glycolic Acid > 5%)', 'Physical Scrubs'] + ingredients_to_avoid

        dos_and_donts = {
            'dos': [
                'Apply sunscreen every morning regardless of weather.',
                'Patch test any new product behind the ear for 24-48 hours.',
                'Maintain a consistent routine for 4-6 weeks to observe results.'
            ],
            'donts': [
                'Do not wash face with excessively hot water.',
                'Avoid aggressively scrubbing skin with physical abrasive towels.',
                'Do not introduce multiple active ingredients simultaneously.'
            ]
        }

        # Deduplicate list preserving order
        unique_seek = list(dict.fromkeys(ingredients_to_seek))
        unique_avoid = list(dict.fromkeys(ingredients_to_avoid))

        return {
            'summary_guidance': guidance,
            'ingredients_to_seek': unique_seek,
            'ingredients_to_avoid': unique_avoid,
            'morning_routine': morning_routine,
            'evening_routine': evening_routine,
            'dos_and_donts': dos_and_donts,
            'disclaimer': 'This guidance is intended strictly for cosmetic and general wellness care. It is not a medical diagnosis or treatment plan for skin conditions.'
        }
