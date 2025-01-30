
def credit_score_range_classification(credit_score):
    color = "white"
    if (credit_score >= 300) and (credit_score <= 579):
        color = "red"
    elif (credit_score >= 580) and (credit_score <= 669):
        color = "orange"
    elif (credit_score >= 670) and (credit_score <= 739):
        color = "yellow"
    elif (credit_score >= 740) and (credit_score <= 799):
        color = "lightgreen"
    else:
        color = "green"

    return color