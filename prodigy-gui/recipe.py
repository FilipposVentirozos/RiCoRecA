import prodigy
from prodigy.components.preprocess import add_tokens
from prodigy.components import loaders
import spacy
# import requests
# from prodigy.components.loaders import JSON


@prodigy.recipe("custom_recipe",
                # dataset=prodigy.recipe_args['dataset'],
                dataset=("Dataset to save annotations to", "positional", None, str),
                source=("Data to annotate (file path or '-' to read from standard input)", "positional", None, str),
                # file_path=("Path to texts", "positional", None, str)
                )
def recipes_2_rel(dataset, source, lang="en"):
    # We can use the blocks to override certain config and content, and set
    # "text": None for the choice interface so it doesn't also render the text
    # html_template = (
    #     '<button class="custom-button" onClick="updateText()"'
    #     '>👇 View Graph'
    #     '</button>'
    #     '<br />'
    #     '<strong>Test</strong>')
    blocks = [
        {"view_id": "relations"},
        # {"view_id": "choice", "text": None},
        # {"view_id": ""}
        {"view_id": "html"}
        # {"view_id": "text_input", "field_id": "entities", "field_autofocus": True}
    ]
    # def get_stream():
    #     res = requests.get("https://cat-fact.herokuapp.com/facts").json()
    #     for fact in res["all"]:
    #         yield {"text": fact["text"], "options": options}

    nlp = spacy.blank(lang)           # blank spaCy pipeline for tokenization
    # Set up the stream. Using the preloaded stream instead of the JSON(source) allows to contiue on where we left off.
    stream = loaders.get_stream(
        source, None, None, rehash=True, dedup=True, input_key="text", is_binary=False
    )
    # stream = JSON(source)  # Alternative usage, if one wants to always start from the beginning.
    stream = add_tokens(nlp, stream)

    with open('static/index.html') as f:
        html_template = f.read()

    with open('static/script.js') as f:
        javascript = f.read()

    return {
        ""
        "dataset": dataset,          # the dataset to save annotations to
        "view_id": "blocks",         # set the view_id to "blocks"
        "stream": stream,            # the stream of incoming examples
        "config": {  # https://prodi.gy/docs/api-interfaces#relations-settings
            "relations_span_labels": ["ACTION",  # Removed HOW
                                      "INGR", "TOOL",  # Main NEs
                                      "MSR", "SETT",  # Answers the How?
                                      "COR_INGR", "COR_TOOL",  # Co-reference
                                      "PAR_INGR", "PAR_TOOL",  # Part of
                                      "STT_INGR", "STT_TOOL",  # State of
                                      # "Id_INGR", "Id_TOOL",  # Used as Identification of. Merge with the above Cor_*.
                                      "If", "Until", "Repeat",  # Code Idioms, removed the Or
                                      "WHY"],  # Answers as to why we do this step?
            "labels": ["Modifier", "Member", "Or", "Join", "Dependency"],
            "wrap_relations": True,
            'html_template': html_template,
            'javascript': javascript,
            "label_style": "list",  # "dropdown" makes the selection of the labels for the annotator in such way
            # https://support.prodi.gy/t/prodigy-multi-user-session-access/1644/2
            "feed_overlap": False,  # Not overlapping instances. Unique for each user.
            "force_stream_order": True,  # setting "force_stream_order" will make sure that all examples are re-sent
            # until they're answered, and always sent out in the same order:
            # "PRODIGY_ALLOWED_SESSIONS": "Al
            "blocks": blocks   # Add the blocks to the config
        }
    }


# The below is used for testing only, better to test from terminal
if __name__ == '__main__':
    dataset_path = "data/Intermediate/stable/recipes_0.0.json"
    recipes_2_rel("recipes_2", dataset_path)
#     prodigy.serve("recipes_2", dataset_path, recipes_2_rel)
