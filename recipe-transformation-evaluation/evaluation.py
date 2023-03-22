import copy
from pyvis.network import Network
from process import process
from os import listdir, makedirs
from os.path import isfile, join, isdir, normpath, dirname, abspath
import json
import re
from pprint import pprint
import spacy
from sklearn.model_selection import train_test_split, KFold
import pandas as pd
import rapidfuzz as fz
import static_variables
from collections import OrderedDict, defaultdict
import datetime
import plac
import spacy_alignments
import tqdm
import logging
log = logging.getLogger(__name__)

data_fn = join(dirname(dirname(abspath(__file__))), "data")

# https://www.davidsbatista.net/blog/2018/05/09/Named_Entity_Evaluation/
# We approximate the SemEval Type NER Metric
scores_NER, scores_Membership, scores_Primes, scores_RE = None, None, None, None
nlp = None

def initialise_scores():
    global scores_NER, scores_Membership, scores_Primes, scores_RE
    scores_NER = {"Correct": defaultdict(int), "Hypothesised": defaultdict(int), "Missed": defaultdict(int), "Incorrect": defaultdict(int)}
    scores_Membership = {"Correct": 0, "Hypothesised": 0, "Missed": 0} 
    scores_Primes = {"Correct": 0, "Hypothesised": 0, "Missed": 0} 
    scores_RE = {"Correct": defaultdict(int), "Hypothesised": defaultdict(int), "Missed": defaultdict(int)} 

def get_recipes(annotator, meal_type):
    recipes = dict()
    # Start with recipes
    # Prodigy Annotationsrecipe_vis
    with open(join(data_fn, "Processed", "Annotator_" + annotator, "prodigy_" + meal_type + ".json")) as fdrs:
        d_out = json.load(fdrs)
    # PSVs
    print("Total recipes")
    print(len(d_out))
    psvs = join(data_fn, "Processed", "Annotator_" + annotator, "PSV_" + annotator)
    onlyfiles = [f for f in listdir(psvs) if isfile(join(psvs, f))]
    omit = []
    not_passed = []
    for recipe in tqdm.tqdm(d_out):
        log.debug(recipe["meta"]["id"])        
        for file in onlyfiles:
            if file.split(".")[0] == recipe["meta"]["id"]:                
                try:
                    recipes[recipe["meta"]["id"]] = process(recipe, join(psvs, file))
                except:
                    not_passed.append(recipe["meta"]["id"])
                break
        else:
            omit.append(recipe["meta"]["id"])
            log.debug("not found")
    log.debug("Not passed")
    pprint(not_passed)
    log.debug("Not Found CSV")
    pprint(omit)
    return recipes

def parser():
    pass

def get_values(tok, idx=0):
    try:
        if tok.count('|') == 4:
            tok = re.sub(r'[\[\]]', "", tok)
            return tok.split("|")[idx].strip()
    except AttributeError:
        return None
def assign_values(func):
    def match_function(tok):
        match func:
            case x if "token" in func.__name__:
                return func(get_values(tok, idx=0))
            case x if "prime" in func.__name__:
                return func(get_values(tok, idx=1))
            case x if "mem" in func.__name__:
                return func(get_values(tok, idx=2))
            case x if "ner" in func.__name__:
                return func(get_values(tok, idx=3))
            case x if "re" in func.__name__:
                return func(get_values(tok, idx=4))
    return match_function

def return_tok(tok):
    if not tok:
        return "O"
    else:
        return tok

@assign_values
def get_token(tok):
    return return_tok(tok)

@assign_values
def get_mem(tok):
    return return_tok(tok)

@assign_values
def get_primes(tok):
    return return_tok(tok)

@assign_values
def get_ner(tok):
    return  return_tok(tok)
@assign_values
def get_re(tok):
    return return_tok(tok)

def recipe_vis(out, fn_ext, primes=True):

    def get_closest_cand(k, token):
        token = token.strip()
        nonlocal nodes
        for i in range(1, 1_000):
            # See Forwards
            try:
                if nodes[k + i]["title"].strip() == token.strip():  
                    return k + i
            except KeyError:
                pass
            # See Backwards
            try:
                if nodes[k - i]["title"].strip() == token.strip():
                    return k - i
            except KeyError:
                pass

    # Parse predicates
    nodes = dict()
    edges  = list()
    out = {int(k): v for k,v in out.items()}
    out = OrderedDict(sorted(out.items(), reverse=True))
    mem = None
    counter_id = 1_000
    additional_nodes = defaultdict(list)
    for k, tok in out.items():
        if tok == mem:
            continue
        # if get_ner(tok).lower() not in static_variables.code:
        #     continue
        # Maybe add the primes, instead for the Entities
        if get_ner(tok).lower().strip() not in static_variables.entities:
            nodes[k] = {"title": get_token(tok), "label": get_ner(tok)}
        else:
            # Alternative format
            for prime in get_primes(tok).split(','):
                if prime == "O":
                    break
                prime = prime.strip()
                counter_id += 1
                additional_nodes[k].append({"id": counter_id, "title": prime, "label": get_ner(tok)})
        mem = tok
    for k, tok in out.items():
        if tok == mem:
            continue
        ## Get Memberships
        for member in get_mem(tok).split(','):
            member = member.strip()
            # It's a Predicate with no upper belonging
            if get_ner(tok).strip().lower() in static_variables.code and get_mem(tok).strip() == get_token(tok).strip():
                break
            # Else add Membership
            else:
                if additional_nodes[k]:
                    for l in additional_nodes[k]:
                        relation = {"head": l["id"], "child": get_closest_cand(k, member), "label": "Member"}
                        edges.append(relation)
                else:
                    for member_ in member.split(","):
                        member_  = member_.strip()
                        relation = {"head": k, "child": get_closest_cand(k, member_), "label": "Member"}
                        edges.append(relation)
        # Add the rest of the relations
        for arc in get_re(tok).split(','):
            if arc == "O":
                break
            label = arc.split("=")[0].strip()
            child = arc.split("=")[1].strip()
            if label == "Dependency":
                relation = {"head": get_closest_cand(k, child), "child": k, "label": label}
                edges.append(relation)
            else:
                if  additional_nodes[k]:
                    for l in additional_nodes[k]:
                        relation = {"head": l["id"], "child": get_closest_cand(k, child), "label": label}
                        edges.append(relation)
                else:
                    relation = {"head": k, "child": get_closest_cand(k, child), "label": label}
                    edges.append(relation)
        # Avoid duplicates from span
        mem = tok

    # Populate the Graph
    g = Network('890px', '1900px', directed=True)
    node_ids, node_value, node_title, node_label, node_color, node_size, node_shape = list(), list(), list(), list(), list(), list(), list()
    for k, node in nodes.items():
        node_ids.append(k)  # Token end seems to be the identifier for the relations
        node_title.append(node["title"])
        label = node["label"]
        node_label.append(label)
        if label.lower() in static_variables.code:
            node_color.append("#E15233")
            node_size.append(20)
            node_shape.append("ellipse")
        elif label.lower() in static_variables.entities:
            if "ingr" in label.lower():
                node_color.append("#EFAE2C")
            else:
                node_color.append("#12A5E9")
            node_size.append(14)
            node_shape.append("dot")
        elif label.lower() in static_variables.states:
            if "ingr" in label.lower():
                node_color.append("#EFAE2C")
            else:
                node_color.append("#12A5E9")
            node_size.append(13)
            node_shape.append("triangle")
        elif label.lower() in static_variables.entities_aux:
            node_color.append("#3BE8AF")
            node_size.append(15)
            node_shape.append("box")
        else:
            node_color.append("#82A5B6")
            node_size.append(10)
            node_shape.append("hexagon")
    for l in additional_nodes.values():
        for node in l:
            node_ids.append(node["id"])  # Token end seems to be the identifier for the relations
            node_title.append(node["title"])
            label = node["label"]
            node_label.append(label)
            if label.lower() in static_variables.code:
                node_color.append("#E15233")
                node_size.append(20)
                node_shape.append("ellipse")
            elif label.lower() in static_variables.entities:
                if "ingr" in label.lower():
                    node_color.append("#EFAE2C")
                else:
                    node_color.append("#12A5E9")
                node_size.append(14)
                node_shape.append("dot")
            elif label.lower() in static_variables.states:
                if "ingr" in label.lower():
                    node_color.append("#EFAE2C")
                else:
                    node_color.append("#12A5E9")
                node_size.append(13)
                node_shape.append("triangle")
            elif label.lower() in static_variables.entities_aux:
                node_color.append("#3BE8AF")
                node_size.append(15)
                node_shape.append("box")
            else:
                node_color.append("#82A5B6")
                node_size.append(10)
                node_shape.append("hexagon")
    g.add_nodes(node_ids, title=node_title, label=node_label, value=node_size, color=node_color, shape=node_shape)
    for edge in edges:
        label = edge["label"]
        if label.lower() == "modifier":
            width = 2
            color = "#65C762"
        elif label.lower() == "member":
            width = 4
            color = "#62C7B5"
        elif label.lower() == "dependency":
            width = 5
            color = "#E38527"
        else:
            width = 1
            color = "#82A5B6"
        try:
            g.add_edge(edge["head"], edge["child"], title=label, width=width, color=color)
        except AssertionError:
            continue
    try:
        g.show("temp/" + fn_ext + ".html")
    except FileNotFoundError:
        pass

def get_score(gs, pr, **kwargs):
    global scores_NER
    log.debug(gs["text_in"])
    tokens_gs = dict()
    for token_id in range(len(gs["tokens"])):
        try:
            tokens_gs[token_id] = gs["tokens_label"][token_id]
        except KeyError:
            try:
                tokens_gs[token_id] = gs["tokens_label"][str(token_id)]
            except KeyError:
                tokens_gs[token_id] = gs["tokens"][token_id]

    tokens_pr = dict()
    for token_id in range(len(pr["tokens"])):
        try:
            tokens_pr[token_id] = pr["tokens_label"][token_id]
        except KeyError:
            try:
                tokens_pr[token_id] = pr["tokens_label"][str(token_id)]
            except KeyError:
                tokens_pr[token_id] = pr["tokens"][token_id]
    # recipe_vis(gs["tokens_label"], fn_ext=kwargs["key"] + "_gt") # Uncomment to visualise
    # recipe_vis(pr["tokens_label"], fn_ext=kwargs["key"] + "_pr") # Uncomment to visualise
    # NER
    def ner_score(gs_, pr_):
        global scores_NER
        match gs_, pr_:
            case x if gs_ == pr_ :
                scores_NER["Correct"][gs_] += 1
            case x if gs_ == "O" and pr_ != "O":
                scores_NER["Hypothesised"][pr_] += 1
            case x if gs_ != "O" and pr_ == "O":
                scores_NER["Missed"][gs_] += 1
            case _:
                scores_NER["Incorrect"][frozenset({gs_, pr_})] += 1
    # Membership
    def mem_score(gs_, pr_):
        global scores_Membership

        gs_s = {tok.strip() for tok in gs_.split(',')}
        pr_s = {tok.strip() for tok in pr_.split(',')}
        scores_Membership["Correct"] += len(gs_s.intersection(pr_s))  # Stands for TP, TN together
        scores_Membership["Missed"] += len(gs_s - pr_s)
        scores_Membership["Hypothesised"] += len(pr_s - gs_s)        
    # Primes
    def primes_score(gs_, pr_):
        global scores_Primes
        # If the token is not an Entity skip
        if gs_.strip() == "" and pr_.strip() == "":
            return
        # Skip if it's not entity related
        if gs_ == pr_ == 'O':
            return
        gs_s = {tok.strip() for tok in gs_.split(',')}
        pr_s = {tok.strip() for tok in pr_.split(',')}
        scores_Primes["Correct"] += len(gs_s.intersection(pr_s))
        scores_Primes["Missed"] += len(gs_s - pr_s)
        scores_Primes["Hypothesised"] += len(pr_s - gs_s)
    # Relations
    def split_re(tok):
        """ In some cases there is a comma in the span, hence, it's not suggested to just split by it
        """
        l = tok.split(',')
        merge = list()
        for idx, re_ in enumerate(l[1:], start=1):
            if re_.strip()[0].islower():
                merge.append((idx-1, idx))
        for m in reversed(merge):
            l.append(l[m[0]] + l[m[1]])
        for m in merge:
            del l[m[1]]
            del l[m[0]]
        return {i.strip() for i in l}

    def re_score(gs_, pr_):
        global scores_RE
        # If the token has no Entities then skip
        if gs_.strip() == "" and pr_.strip() == "":
            return
        gs_s = split_re(gs_)
        pr_s = split_re(pr_)
        # Currently is strict, has to match the
        for re_ in gs_s.intersection(pr_s):
            scores_RE["Correct"][re_.split("=")[0].strip()] += 1
        for re_ in gs_s.intersection(gs_s - pr_s):
            scores_RE["Missed"][re_.split("=")[0].strip()] += 1
        for re_ in gs_s.intersection(pr_s - gs_s):
            scores_RE["Hypothesised"][re_.split("=")[0].strip()] += 1

    for (k_gs, tok_gs), (k_pr, tok_pr) in zip(tokens_gs.items(), tokens_pr.items()):
        # Parsing Error
        if k_gs != k_pr:
            raise IndexError
        # If both are 'O' skip
        if isinstance(tok_gs, dict) and isinstance(tok_pr, dict):
            continue
        ner_score(get_ner(tok_gs), get_ner(tok_pr))
        mem_score(get_mem(tok_gs), get_mem(tok_pr))
        primes_score(get_primes(tok_gs), get_primes(tok_pr))
        if get_ner(tok_gs).lower() in static_variables.entities and get_primes(tok_gs) != get_primes(tok_pr):
            log.debug((tok_gs, tok_pr))
            # print()
        re_score(get_re(tok_gs), get_re(tok_pr))

def fname(fp, dataset_name, rnd_num="none", fold=""):
    td = datetime.date.today()
    return join(fp, dataset_name + "_" + str(td.year) + "_" + str(td.month) + "_" + str(td.day) + "_" +
                str(rnd_num) + "_f" + str(fold) + ".csv")

def generate_cross_val_predict_files(d_out, fp, rnd_num=None):
    X = [i["text_in"] for i in d_out.values()]
    Y = [i["text_out"] for i in d_out.values()]
    k = list(d_out.keys())
    kf = KFold(n_splits=5, random_state=rnd_num, shuffle=True)
    for i, (train, test) in enumerate(kf.split(X, y=Y)):
        val, test = train_test_split(test, test_size=0.66, random_state=rnd_num)
        X_train = [X[idx] for idx in train]
        k_train = [k[idx] for idx in train]
        Y_train = [Y[idx] for idx in train]
        X_val = [X[idx] for idx in val]
        k_val = [k[idx] for idx in val]
        Y_val = [Y[idx] for idx in val]
        X_test = [X[idx] for idx in test]
        k_test = [k[idx] for idx in test]
        Y_test = [Y[idx] for idx in test]
        pd.DataFrame({"id":k_train, "recipe": X_train, "out": Y_train}).to_csv(fname(fp, "train", rnd_num, i), index=False)
        pd.DataFrame({"id":k_val,"recipe": X_val, "out": Y_val}).to_csv(fname(fp, "validation", rnd_num, i), index=False)
        pd.DataFrame({"id":k_test,"recipe": X_test, "out": Y_test}).to_csv(fname(fp, "test", rnd_num, i), index=False)

def generate_predict_files(d_out, rnd_num, fp=None):
    X = [i["text_in"] for i in d_out.values()]
    Y = [i["text_out"] for i in d_out.values()]
    # 0.8 , 0.1 , 0.1
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.15, random_state=rnd_num)
    K_train, K_test = train_test_split(list(d_out.keys()), test_size=0.15, random_state=rnd_num)
    X_test, X_val, Y_test, Y_val = train_test_split(X_test, Y_test, test_size=0.40, random_state=rnd_num)
    K_test, K_val = train_test_split(K_test, test_size=0.40, random_state=rnd_num)

    # 0.7 0.1 0.2
    # Assertion Check
    for k, txt in zip(K_test, X_test):
        assert d_out[k]["text_in"] == txt
    if fp:
        # todo Print meta files with the keys
        pd.DataFrame({"recipe": X_train, "out": Y_train}).to_csv(fname(fp, "train"), index=False)
        pd.DataFrame({"recipe": X_val, "out": Y_val}).to_csv(fname(fp, "validation"), index=False)
        pd.DataFrame({"recipe": X_test, "out": Y_test}).to_csv(fname(fp, "test"), index=False)

    return K_train, K_val, K_test


def alignment(pred, gt_d):
    # r'(?<=\[)[^\s]+'
    # '(? <= \[). *?(?=\])'
    # '(? <= \[). *?(?=\])'
    tokens_label_raw = re.findall('\[.*?\]', pred)
    repl = "placeholder_x70x6c"
    temp = re.sub(r'\[.*?\]', repl, pred, 0, re.IGNORECASE | re.MULTILINE)
    # tokenise from spaCy
    nlp.tokenizer(temp)

    # Retrieve original text input
    tokens_label_raw_in = [tok.split('[')[1].split('|')[0].strip() for tok in tokens_label_raw]

    text_in_pred = ""
    par_temp = copy.copy(temp)
    for tok in tokens_label_raw_in:
        idx = par_temp.index(repl)
        par_temp = par_temp.replace(repl, tok, 1)
        text_in_pred += par_temp[:idx + len(tok)]
        par_temp = par_temp[idx + len(tok):]
    text_in_pred += par_temp
    text_in_pred = text_in_pred.rstrip()

    # Mapping dictionary between text_in_pred and tokens_label_raw_in
    # First get the tokenised labelled output
    mapping_token_labels_raw_in = dict()
    tokenised_labels_raw_in = list()
    counter = 0
    for idx, tok in enumerate(tokens_label_raw_in):
        for tok_ in nlp(tok):
            mapping_token_labels_raw_in[counter] = idx
            tokenised_labels_raw_in.append(tok_.text)
            counter += 1

    pred_mapping = dict()
    counter = 0
    for idx, tok in enumerate(nlp(text_in_pred)): # Using same tokeniser
        try:
            if tok.text == tokenised_labels_raw_in[counter]:
                pred_mapping[idx] = counter
                counter +=1
        except IndexError: # Reached end of detected labelled tokens
            break

    # Align
    seq_gt = [tok["text"] for tok in gt_d["tokens"]]
    seq_pred = [tok.text for tok in nlp(text_in_pred)]

    # Align the found items only
    mapping_gt2pred = {}
    a2b, b2a = spacy_alignments.get_alignments(seq_gt, seq_pred)
    for i in range(len(seq_gt)):
        # print(seq_gt[i])
        for j in a2b[i]:
            # print("    ", seq_pred[j])
            mapping_gt2pred[i] = j

    tokens_label_pred = {}
    for idx in range(len(seq_gt)):
        try:
            tokens_label_pred[idx] = tokens_label_raw[mapping_token_labels_raw_in[pred_mapping[mapping_gt2pred[idx]]]]
        except KeyError:
            continue

    out_dict = {"text_in": text_in_pred, "text_out": pred, "tokens": gt_d["tokens"],
                "tokens_label": tokens_label_pred}
    return out_dict

def cross_val_scoring(gs_d, in_, out_):
    global nlp
    nlp = spacy.blank("en")
    # Get predictions
    predictions = list()
    with open(out_, 'r') as f:
        for line in f:
            predictions.append(line)
    # Get the keys
    K_test = pd.read_csv(in_, header=0, index_col=False)["id"]

    pred_d = {}
    # Do the mapping and create dict
    for pred, k in zip(predictions, K_test):        
        pred_d[k] = alignment(pred, gs_d[k])

    # Get Ground Truth dict
    for k in K_test:
        if k not in gs_d:
            del gs_d[k]

    return fill_scores(gs_d, pred_d)

def predict_files_score(d_out, fp, rnd_num):
    global nlp
    nlp = spacy.blank("en")
    with open(d_out, "r") as fd:
        d_out = json.load(fd)
    predictions = list()
    with open(fp, 'r') as f:
        for line in f:
            predictions.append(line)
    # todo get keys from metadata
    # Get tokens for mapping, deprecated
    # _, _, K_test = generate_predict_files(d_out, rnd_num=rnd_num)
    # Get the keys by similarity parsing
    K_test = list()
    for idx, pred in enumerate(predictions):
        highest_score = 0
        k_highest = None
        for k, rec in d_out.items():
            score = fz.fuzz.ratio(rec["text_out"], pred)
            if score > highest_score:
                highest_score = score
                k_highest = k
        # Sanity check
        K_test.append(k_highest)
    # remove spurious entries, manually
    remove_idx = [2, 5, 6, 8]
    for idx in sorted(remove_idx, reverse=True):
        del K_test[idx]
        del predictions[idx]

    pred_d = {}
    # Do the mapping and create dict
    for pred, k in zip(predictions, K_test):
        pred_d[k] = alignment(pred, d_out[k])

    # Get Ground Truth dict
    for k in K_test:
        if k not in d_out:
            del d_out[k]

    fill_scores(d_out, pred_d)

def fill_scores(Gold_standard, Prediction):
    for k, v in Gold_standard.items():
        try:
            get_score(v, Prediction[k], key=k)
        except KeyError:
            continue
    results = {}
    # Primes
    primes = {}
    precision = scores_Primes["Correct"] / (scores_Primes["Correct"] + scores_Primes["Hypothesised"])
    recall = scores_Primes["Correct"] / (scores_Primes["Correct"] + scores_Primes["Missed"])
    f_score =  (2 * precision * recall) / (precision + recall)
    print("Primes F-Score: ", round(f_score, 3), end='\t')
    primes["fscore"] = f_score
    print("Precision: ", round(precision, 3), end='\t')
    primes["precision"] = precision
    print("Recall: ", round(recall, 3))
    primes["recall"] = recall
    results['primes'] = primes


    # Membership
    memberships = {}
    precision = scores_Membership["Correct"] / (scores_Membership["Correct"] + scores_Membership["Hypothesised"])
    recall = scores_Membership["Correct"] / (scores_Membership["Correct"] + scores_Membership["Missed"])
    f_score =  (2 * precision * recall) / (precision + recall)
    print("Membership F-Score: ", round(f_score, 3), end='\t')
    memberships["fscore"] = f_score
    print("Precision: ", round(precision, 3), end='\t')
    memberships["precision"] = precision
    print("Recall: ", round(recall, 3))
    memberships["recall"] = recall
    results['memberships'] = memberships

    # NER total. I think is micro
    ners = {}
    precision = sum(scores_NER["Correct"].values())  / (sum(scores_NER["Correct"].values()) +
                                                        sum(scores_NER["Incorrect"].values())  +
                                                        sum(scores_NER["Hypothesised"].values()) )
    recall = sum(scores_NER["Correct"].values())  / (sum(scores_NER["Correct"].values())  +
                                                     sum(scores_NER["Incorrect"].values())  + sum(scores_NER["Missed"].values()))
    f_score =  (2 * precision * recall) / (precision + recall)
    print("NER Micro F-Score: ", round(f_score, 3), end='\t')
    ners["fscore"] = f_score
    print("Precision: ", round(precision, 3), end='\t')
    ners["precision"] = precision
    print("Recall: ", round(recall, 3))
    ners["recall"] = recall
    results['ners'] = ners

    # NER for each Class/Tag
    for k in scores_NER["Correct"].keys():
        precision = scores_NER["Correct"][k] / (scores_NER["Correct"][k] +
                                                           sum([scores_NER["Incorrect"][k_] for k_ in scores_NER["Incorrect"].keys() if k in k_]) +
                                                           scores_NER["Hypothesised"][k])
        recall = scores_NER["Correct"][k] / (scores_NER["Correct"][k] +
                                                           sum([scores_NER["Incorrect"][k_] for k_ in scores_NER["Incorrect"].keys() if k in k_]) +
                                                           scores_NER["Missed"][k])
        f_score = (2 * precision * recall) / (precision + recall)
        print("\t" + k.ljust(10) + " NER F-Score: ", round(f_score, 3), end='\t')
        print("Precision: ", round(precision, 3), end='\t')
        print("Recall: ", round(recall, 3), end='\t')
        print("Support: ", scores_NER["Correct"][k] +
              sum([scores_NER["Incorrect"][k_] for k_ in scores_NER["Incorrect"].keys() if k in k_]) + scores_NER["Missed"][k])

    # RE total
    res = {}
    del scores_RE["Correct"]['O']  # I may want to remove this
    precision = sum(scores_RE["Correct"].values()) / (sum(scores_RE["Correct"].values()) + sum(scores_RE["Hypothesised"].values()))
    recall = sum(scores_RE["Correct"].values()) / (sum(scores_RE["Correct"].values()) + sum(scores_RE["Missed"].values()))
    f_score =  (2 * precision * recall) / (precision + recall)
    print("RE Micro F-Score: ", round(f_score, 3), end='\t')
    res["fscore"] = f_score
    print("Precision: ", round(precision, 3), end='\t')
    res["precision"] = precision
    print("Recall: ", round(recall, 3))
    res["recall"] = recall
    results['res'] = res
    # RE for each Class/Tag
    for k in scores_RE["Correct"].keys():
        precision = scores_RE["Correct"][k] / (scores_RE["Correct"][k] + scores_RE["Hypothesised"][k])
        recall = scores_RE["Correct"][k] / (scores_RE["Correct"][k] + scores_RE["Missed"][k])
        f_score = (2 * precision * recall) / (precision + recall)
        print("\t" + k.ljust(10)  + " RE F-Score: ", round(f_score, 3), end='\t')
        print("Precision: ", round(precision, 3), end='\t')
        print("Recall: ", round(recall, 3), end='\t')
        print("Support: ", scores_RE["Correct"][k] + scores_RE["Missed"][k])

    return results

def get_prediction():
    pass


@plac.opt('eval_type', "The evaluation to take place", choices=["inter", "cross"])
@plac.flg('gen_cross', "The evaluation to take place")
def main(eval_type=None, gen_cross=False):
    """A script for machine learning"""        
    if not eval_type and not gen_cross:
        print("Please select an evaluation or generate files for cross-validation")
    if eval_type == "inter":
        Gold_Standard_Dinner = get_recipes("P", "dinner")                
        H_Dinner = get_recipes("H", "dinner")
        M_Dinner = get_recipes("M", "dinner")
        initialise_scores()
        print("\n\n\n")
        print("Inter-annotator agreement with annotator H:")
        fill_scores(Gold_Standard_Dinner, H_Dinner)
        initialise_scores()
        print("\n\nInter-annotator agreement with annotator M:")
        fill_scores(Gold_Standard_Dinner, M_Dinner)

    elif eval_type == "cross":
        Gold_Standard_Dinner = get_recipes("P", "dinner")   
        Gold_Standard_Lunch = get_recipes("P", "lunch")   
        Gold_standard = Gold_Standard_Dinner | Gold_Standard_Lunch
        initialise_scores()
        results = list()
        print("\nPEGASUS-X Folds:")
        for i in [0, 1, 2, 3, 4]:
            in_ =  join(data_fn, "Processed", "Cross_Val", "test_2023_2_22_3245_f" + str(i) + ".csv")
            out_ = join(data_fn, "Predictions", "CrossVal", "fold" + str(i) + "_pegasus", "generated_predictions.txt")            
            print("__ Fold: " + str(i) + " results__")
            results.append(cross_val_scoring(Gold_standard, in_, out_))

        avg_res = defaultdict(list)
        for i in results:
            for k in i.keys():
                for k_, v_ in i[k].items():
                    avg_res[k,k_].append(v_)
        
        print("__PEGASUS Averaged results__")
        for k_, v_ in avg_res.items():
            print(k_[0], " - ", k_[1], round(sum(v_) / 5, 3))                    

        ## Cross Validation Results on Pegasus
        results = list()
        initialise_scores()
        print("\nLongT5 Folds:")
        for i in [0, 1, 2, 3, 4]:            
            in_ =  join(data_fn, "Processed", "Cross_Val", "test_2023_2_22_3245_f" + str(i) + ".csv")            
            out_ = join(data_fn, "Predictions", "CrossVal", "fold" + str(i) + "_t5", "generated_predictions.txt")            
            print("__ Fold: " + str(i)+ " results__")
            results.append(cross_val_scoring(Gold_standard, in_, out_))

        avg_res = defaultdict(list)
        for i in results:
            for k in i.keys():
                for k_, v_ in i[k].items():
                    avg_res[k,k_].append(v_)

        print("\n__T5 Averaged results__")
        for k_, v_ in avg_res.items():
            print(k_[0], " - ", k_[1] , round(sum(v_) / 5, 3))

    if gen_cross:
        Gold_Standard_Dinner = get_recipes("P", "dinner")   
        Gold_Standard_Lunch = get_recipes("P", "lunch")   
        Gold_standard = Gold_Standard_Dinner | Gold_Standard_Lunch
        fp = join(data_fn, "Processed", "Cross_Val")
        generate_cross_val_predict_files(Gold_standard, fp, rnd_num=432) # 3245
        


if __name__ == '__main__':
    plac.call(main)
    