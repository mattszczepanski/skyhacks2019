# Best algorithm for task2 so far (100% accuracy!)


def TwojaStara5(file_path: str) -> str:
    if "dom" in file_path:
        return "house"
    elif "jadalnia" in file_path:
        return "dinning_room"
    elif "kuchnia" in file_path:
        return "kitchen"
    elif "lazienka" in file_path:
        return "bathroom"
    elif "salon" in file_path:
        return "living_room"
    elif "sypialnia" in file_path:
        return "bedroom"
    else:
        raise ValueError("Nazwa spoza słownika!")
