LABEL_ORDER = ["left", "center", "right"]
LABEL_TO_ID = {label: index for index, label in enumerate(LABEL_ORDER)}
ID_TO_LABEL = {index: label for label, index in LABEL_TO_ID.items()}
BIAS_ID_TO_TEXT = {0: "left", 1: "center", 2: "right"}
