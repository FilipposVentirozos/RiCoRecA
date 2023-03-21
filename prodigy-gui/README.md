# Prodigy Annotation

## Installation

You will need to install the Prodigy library.

I installed it as a [wheel](https://prodi.gy/docs/install#wheel) using the prodigy-1.11.7-cp310 version.

After you install it you would need to copy the `index.html` file in this repository and replace the `index.html` in Prodigy site-packages in your installed Python environmnet 
`.../venv/lib64/python3.10/site-packages/prodigy/static/index.html`.


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