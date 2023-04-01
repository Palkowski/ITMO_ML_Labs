def grade(score):
    if 0 <= score < 60:
        return "неудовлетворительно"
    elif 60 <= score <= 74:
        return "удовлетворительно"
    elif 74 < score <= 90:
        return "хорошо"
    elif 90 < score <= 100:
        return "отлично"


def solve(a, b):
    if a == b == 0:
        return "Any"
    elif a == 0 and b != 0:
        return "Error"
    else:
        return b / a
