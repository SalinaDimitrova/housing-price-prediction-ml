def apply_rules(house_data):
    """
    Експертни правила за интерпретация на прогнозата
    house_data: dict с характеристики на жилището
    """
    explanations = []

    # Доход
    if house_data.get("median_income", 0) > 6:
        explanations.append(
            "Високият среден доход в района оказва силно положително влияние върху цената."
        )

    # Стаи на домакинство
    if house_data.get("households", 1) > 0:
        rooms_per_household = (
            house_data.get("total_rooms", 0) / house_data["households"]
        )
        if rooms_per_household > 5:
            explanations.append(
                "Голям брой стаи на домакинство е индикатор за по-висок стандарт на живот."
            )

    # Възраст на сградата
    if house_data.get("housing_median_age", 100) < 20:
        explanations.append(
            "По-новите сгради обикновено имат по-висока пазарна стойност."
        )

    # Гъстота на населението
    if house_data.get("households", 1) > 0:
        population_density = (
            house_data.get("population", 0) / house_data["households"]
        )
        if population_density < 3:
            explanations.append(
                "По-ниската гъстота на населението повишава привлекателността на района."
            )

    return explanations
