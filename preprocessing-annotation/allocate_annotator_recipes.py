
from os.path import join, dirname, abspath
import json

data_fn = join(dirname(dirname(abspath(__file__))), "data")
dinner = join(data_fn, "Intermediate", "recipes_Dinners_v1.json")


# Left annotations id for Philip to annotate
counter = 0
Annotator_P, Annotator_M, Annotator_H = list(), list(), list()
with open(dinner, "r") as js:
    dinner_d = json.load(js)
    for counter, recipe in enumerate(dinner_d):        
        if counter < 50:
            Annotator_P.append(recipe)
            Annotator_M.append(recipe)
        elif 100 > counter >= 50:
            Annotator_P.append(recipe)
            Annotator_H.append(recipe)
        else:
            break

with open(join(data_fn, "Intermediate", "Annotator_P.json"), "w") as js:
    json.dump(Annotator_P, js, indent=4)
with open(join(data_fn, "Intermediate", "Annotator_M.json"), "w") as js:
    json.dump(Annotator_M, js, indent=4)
with open(join(data_fn, "Intermediate", "Annotator_H.json"), "w") as js:
    json.dump(Annotator_H, js, indent=4)
