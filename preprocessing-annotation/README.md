
## Installation

You will need to install the Prodigy library.

I installed it as a [wheel](https://prodi.gy/docs/install#wheel) using the prodigy-1.11.7-cp310 version.

After you install it you would need to copy the `index.html` file in this repository and replace the `index.html` in Prodigy site-packages in your installed Python environmnet 
`.../venv/lib64/python3.10/site-packages/prodigy/static/index.html`.

One can look for referece the `requirements.txt`.

## Pre-Processing

The dataset used in our Annotation was the `data/Raw/foodbase/FooDBase_curated_corrected.xml` which is an edited version of the `FoodBase_curated.xml` to account for some tokenisation mismatches.

The `pre-processing.py` handles that dataset, making it appropriate for Prodigy Annotation. It labels the ACTION and INGR spans using spaCy's POS and the FoodBase annotations accordingly for the annotator's convenience. The output files will be found in `data/Intermediate/`.

The `allocate_annotator_recipes.py` take an above-produced file and allocates it to three annotators, similarly to how it was done in the study. The output files will be found in `data/Intermediate/`. This file acts as an example of how we allocated the recipes. The exact files for allocation are not shared because they would obfuscate the reader. The reason is there were multiple files due to annotators leaving and joining, the dev/ops period mentioned in the study and technical issues faced by the annotators resulted in re-counting which recipes were annotated and producing new files with the left annotations to annotate or ignoring them.

## Annotate

You are ready now to annotate!

Once you source your Python environmnet in which Prodigy was installed in, you can type the command:

`prodigy custom_recipe <database-name> <file.json> -F recipe.py`

You can make customisations of the GUI and port in your `/home/<your account>/.prodigy/prodigy.json`. Ours looked like this:


    {
        "custom_theme": {
            "cardMaxWidth": 2800,
            "buttonSize": 70,
            "f6f6f6": "#E4C4EF"
        } ,
        "theme": "eighties",
        "port": 7070
    }

You can find more information in [Prodigy docs](https://prodi.gy/docs) and [its support pages](https://support.prodi.gy/).

The annotations done by the three annotators of our study can be found in `data/Processed/Annotator_*`. Where inside are the json files derived from Prodigy and the CSV/PSV files from the Spreadshit feeling.

For more details on the Annotation Campaign please view the paper.