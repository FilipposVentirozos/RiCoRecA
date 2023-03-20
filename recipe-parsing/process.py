import pprint
import sys
from collections import defaultdict
from os import listdir, makedirs
from os.path import isfile, join, isdir
import json
import copy
import pandas as pd
import rapidfuzz as fz
import itertools
import numpy as np
from action_set import ActionSet, NotEntityMatched
import ne
from pprint import pprint, pformat
from os import path
import random
import static_variables as stv
import logging
import csv
log_file_path = path.join(path.dirname(path.abspath(__file__)), 'log', 'log_wf_processes.txt')
# basicConfig(filename=log_file_path, level=logging.DEBUG)
logging.basicConfig(
    level=logging.DEBUG,
    format="%(message)s",
    # format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()
# Add formatter to the file handler.
logger.info("---  ----")
sys.setrecursionlimit(1000)

# d_out = defaultdict(list)
#
# directory = "/home/chroner/PhD_remote/FoodBase/data/Processed/DB/17_11_2022"
# onlydirs = [f for f in listdir(directory) if isdir(join(directory, f))]
# for dir in onlydirs:
#     onlyfiles = [f for f in listdir(join(directory, dir)) if isfile(join(directory, dir, f))]
#     for file in onlyfiles:
#         with open(join(directory, dir, file), "r") as jsonl:
#             json_list = list(jsonl)
#         for json_str in json_list:
#             d_out[json.loads(json_str)["_session_id"]].append(json.loads(json_str))


# entities = ["ingr", "tool", "cor_ingr", "cor_tool", "par_ingr", "par_tool"]
# entities_aux = ["msr", "sett"]
# states = ["stt_ingr", "stt_tool"]
# code = ["action", "if", "until", "repeat"]
# why = ["why"]


def get_sent_id(sent_psv, start, end):
    """ Get the sentence where the token belongs to. First find candidates by string index matching then use fuzzy matching
    to rectify"""
    text_len = 0
    for idx, dd in enumerate(sent_psv):
        try:
            text_len += len(dd[0]["sentence"]) + 1  # + 1 for between sentence space
        except TypeError:
            text_len += len(dd[0][0]) + 1  # + 1 for between sentence space
        if end > text_len:
            continue
        else:
            break
    else:
        raise IndexError
    # Double check and it's neighbours
    try:
        if ActionSet.recipe["text"][start: end].lower() in sent_psv[idx][0]["sentence"].lower():
            return idx
    except TypeError:
        if ActionSet.recipe["text"][start: end].lower() in sent_psv[idx][0][0].lower():
            return idx
    try:
        if ActionSet.recipe["text"][start: end].lower() in sent_psv[idx-1][0]["sentence"].lower():
            return idx - 1
    except IndexError:
        pass
    except TypeError:
        if ActionSet.recipe["text"][start: end].lower() in sent_psv[idx-1][0][0].lower():
            return idx - 1
    try:
        if ActionSet.recipe["text"][start: end].lower() in sent_psv[idx+1][0]["sentence"].lower():
            return idx + 1
    except IndexError:
        pass
    except TypeError:
        if ActionSet.recipe["text"][start: end].lower() in sent_psv[idx+1][0][0].lower():
            return idx - 1
    raise IndexError

# Note 1: Prodigy uses toke_end as id
# Note 2: Child is where the arrow ends
# Note 3: The spans come in order
# for span in reversed(ActionSet.recipe["spans"]):

def reverse_relation(relation):
    reversed_relation = copy.copy(relation)
    reversed_relation["head"] = relation["child"]
    reversed_relation["head_span"] = relation["child_span"]
    reversed_relation["child"] = relation["head"]
    reversed_relation["child_span"] = relation["head_span"]
    return reversed_relation

# This can go inside the action set, method
# Maybe not because it's used as a condition in the main process
def get_relations(span):
    # dep_relations, mem_relations = list(), list()
    belongs_to_dep_or_mod = False
    relations = defaultdict(list)
    drop = []
    for idx, relation in enumerate(ActionSet.recipe["relations"]):
        try:
            if span["token_end"] == relation["head"]:
                # It can be entity 2 code
                if relation["label"] == "Dependency":  # Dependency linkage
                    # dep_relations.append(relation)
                    relations["depend"].append(relation)
            elif span["token_end"] == relation["child"]:
                if relation["label"] == "Dependency":
                    belongs_to_dep_or_mod = True
            if span["token_end"] == relation["head"] or span["token_end"] == relation["child"]:
                # Check if code2code
                if relation["child_span"]["label"].lower() in stv.code and \
                      relation["head_span"]["label"].lower() in stv.code:
                    if relation["label"].lower() == "join":
                        # dep_relations.append(relation)
                        relations["join"].append(relation)
                        # Reverse signs to include the opposite travel
                        # dep_relations.append(reverse_relation(relation))
                        relations["join"].append(reverse_relation(relation))
                    elif relation["label"].lower() == "or":
                        # dep_relations.append(relation)
                        relations["or"].append(relation)
                        # Reverse signs to include the opposite travel
                        # dep_relations.append(reverse_relation(relation))
                        relations["or"].append(reverse_relation(relation))
                    elif relation["label"].lower() == "member":
                        relations["member"].append(relation)
                        relations["member"].append(reverse_relation(relation))
                    elif relation["label"].lower() == "modifier":
                        belongs_to_dep_or_mod = True
        except AttributeError:
            logger.error("Relation error")
            logger.error("Check '" + ActionSet.recipe["text"][relation["head_span"]["start"]: relation["head_span"]["end"]] + "' or '" + \
                         ActionSet.recipe["text"][relation["child_span"]["start"]: relation["child_span"]["end"]] + "' token, has error")
            # Drop relation to not confuse downstream
            drop.append(idx)
    # Drop corrupt relation
    for idx in reversed(drop):
        # del ActionSet.recipe["relations"][idx]
        del ActionSet.recipe["relations"][idx]
    # print(len(recipe["relations"]))
    # print(len(ActionSet.recipe["relations"]))
    # print()
    return relations, belongs_to_dep_or_mod


def recrv_find_action_set(span, paths=None): # prev_search_span_ids=None):
    """ Recursive depth-based search algorithm to find the Action-set that a span belongs to (LUs). The search is done on the
    recipe's relations. The Algorithm has a stack and it encourages to find every time a new node if not go back a level.

    :param span:
    :param prev_search_span_ids: False, not searched thoroughly, True is when it has been search thoroughly
    :return:
    """
    acceptable_relations = ["member", "or", "modifier"]
    if not paths:
        paths = defaultdict(list)
    for relation in ActionSet.recipe["relations"]:
        if relation["head"] == span["token_end"] and relation["label"].lower() in acceptable_relations:
            # Check if the other span is an AnchorCode of an ActionSet
            # If so then return it
            for i, j, action_set in ActionSet.get_idx(ActionSet.action_sets):
                if relation["child"] == action_set.anchor_code.get_id():
                    return i, j
            if relation["child"] not in paths[span["token_end"]]:
                paths[span["token_end"]].append(relation["child"])
                return recrv_find_action_set(relation["child_span"], paths=paths)
        # Do the same inverse relationship
        elif relation["child"] == span["token_end"] and relation["label"].lower() in acceptable_relations:
            # Check if the other span is an AnchorCode of an ActionSet
            # If so then return it
            for i, j, action_set in ActionSet.get_idx(ActionSet.action_sets):
                if relation["head"] == action_set.anchor_code.get_id():
                    return i, j
            if relation["head"] not in paths[span["token_end"]]:
                paths[span["token_end"]].append(relation["head"])
                return recrv_find_action_set(relation["head_span"], paths=paths)
    # if it has come again to a dead end
    # delete randomly a path and re-try
    # print(span["label"], ActionSet.recipe["text"][span["start"]: span["end"]])
    try:
        paths[span["token_end"]].pop(random.randrange(len(paths[span["token_end"]])))
    except ValueError:
        raise NotImplementedError
    return recrv_find_action_set(span, paths=paths)

def assign_ne(span):
    match span["label"].lower():
        case w if w in stv.entities:
            return ne.Entity(span)
        case w if w in stv.entities_aux:
            return ne.EntityAux(span)
        case w if w in stv.states:
            return ne.State(span)
        case w if w in stv.why:
            return ne.Why(span)
        case _:
            # Probably due to previous Annotation Guideline confusions
            if span["label"].lower() != "action":
                logger.error("Misplaced span '" + ActionSet.recipe["text"][span["start"]: span["end"]] + "' with label " + span["label"])
                # Remove from relations for safety
                drop = list()
                for idx, relation in enumerate(ActionSet.recipe['relations']):
                    if relation["head"] == span["token_end"] or relation["child"] == span["token_end"]:
                        drop.append(idx)
                for idx in reversed(drop):
                    del ActionSet.recipe['relations'][idx]
            else:
                try:
                    raise NotImplementedError(span["label"], ActionSet.recipe["text"][span["start"]: span["end"]],
                                          ActionSet.recipe["text"][span["start"]: span["end"] + 20])
                except IndexError:
                    raise NotImplementedError(span["label"], ActionSet.recipe["text"][span["start"]: span["end"]])


# noinspection SpellCheckingInspection
def process(recipe, psv_file):

    # Pass method to static
    ActionSet.recipe = recipe
    ne.NE.recipe = recipe
    del recipe # For safety
    # Check the Annotator added

    delims = [{"delimiter": '\t'}, {"delimiter": ',', "quoting": csv.QUOTE_ALL}, {"delimiter": '|'}]
    for delim in delims:
    # if annotator == "Mau":
    #     sv = pd.read_csv(psv_file, header=[0, 1], delimiter='\t')
    # elif annotator == "Haifa":
    #     sv = pd.read_csv(psv_file, header=[0, 1], delimiter=',', quoting=csv.QUOTE_ALL)
    # else:
    #     sv = pd.read_csv(psv_file, header=[0, 1], delimiter='|')
        try:
            sv = pd.read_csv(psv_file, header=[0, 1], **delim)
            logger.debug(sv.columns)
            annotator_added = defaultdict(list)
            try:
                assert "Annotator added:" in sv[('Unnamed: 0_level_0', 'Actions')].tail(1).tolist()[0]
                for col in sv.columns[1:]:
                    if sv.tail(1)[col].values[0] == 1.0:
                        if "INGR" in col:
                            annotator_added["INGR"].append(col[1])
                        elif "TOOL" in col:
                            annotator_added["TOOL"].append(col[1])
                logger.debug("Annotator added")
                logger.debug(pformat(annotator_added))
            except AssertionError:
                logger.error("Annotator added not found")
            finally:
                for col in sv.columns:
                    if "INGR" in col:
                        if col[1] not in ActionSet.recipe["text"]:
                            if col[1] not in annotator_added["INGR"]:
                                annotator_added["INGR"].append(col[1])
                    if "TOOL" in col:
                        if col[1] not in ActionSet.recipe["text"]:
                            if col[1] not in annotator_added["TOOL"]:
                                annotator_added["TOOL"].append(col[1])
            if annotator_added:
                print("Annotator added finally")
                print(pformat(annotator_added))
                # make option to exclude annotator added, if requested
            break
        except (KeyError, TypeError, pd.errors.ParserError):
            continue


    # Double Alignment, sentence then Action (if any)
    # List of sentences
    sent_psv = list()
    counter_mem = 0
    # Iterate over segmented recipe sentences
    for sentence in ActionSet.recipe["meta"]["sentences"]:
        counter = counter_mem
        sents = []
        # print("new sentence__")
        connected = False  # To break if there is another sentence in between
        while counter_mem < counter + 10:  # Iterate until next sentence
            # print(counter)
            try:
                if fz.fuzz.ratio(sv[('Unnamed: 1_level_0', 'Sentences')][counter], sentence) > 90:
                    info = defaultdict(list)
                    for col in sv.columns:
                        if sv[col][counter] == 1.0:
                            if "INGR" in col:
                                info['INGR'].append(col[1])
                            elif "TOOL" in col:
                                info['TOOL'].append(col[1])
                        elif "Actions" in col:
                            info["Action"].append(sv[col][counter])
                        elif "Action_start" in col:
                            info["Action_start"].append(sv[col][counter])
                        elif "Action_end" in col:
                            info["Action_end"].append(sv[col][counter])
                    info["sentence"] = sentence
                    sents.append(info)
                    counter_mem = counter
                    counter += 1
                    connected = True
                    # break
                else:
                    counter += 1
                    if connected:
                        break
            except TypeError:
                counter += 1
                connected = False
            except KeyError:
                break
        if not sents:  # If no sentence detected in the PSV add it from the meta
            # Try again with a smaller counter
            # match_sentence()
            # If that failed then
            sents.append([sentence])
        sent_psv.append(sents)
        # for sent_ in sent_psv:
        #     for sent in sent_:
        #         print(sent["sentence"])

    action_sets = defaultdict(list)
    second_pass_idx = dict()
    # First get all the Actions that have a PSV record
    for idx, span in enumerate(ActionSet.recipe["spans"]):  # Start ordinarily
        relations, belongs_to_dep = get_relations(span)
        if relations or belongs_to_dep:
            # Get Sentence Id
            sent_id = get_sent_id(sent_psv, span["start"], span["end"])
            # Get the Action verb if any
            try:
                for action_set_psv in sent_psv[sent_id]:
                    if fz.fuzz.partial_ratio(action_set_psv["Action"][0], ActionSet.recipe["text"][span["start"]: span["end"]]) > 90:
                        # Create new Action set and handle
                        # Link the psv
                        if span["label"].lower() in stv.code:
                            e = ne.Code(span)
                        elif span["label"].lower() in stv.entities:
                            e = ne.Entity(span)
                        try:
                            action_sets[sent_id].append(ActionSet(action_set_psv, anchor_code=e, **relations))
                        except TypeError:
                            action_sets[sent_id].append(ActionSet(action_set_psv, anchor_code=e))
                        break
                else:  # Is probably an Until
                    if span["label"].lower() in stv.code:
                        # Is it linked to another Verb with a Member, connect to its closest ones, done below
                        logger.debug("second")
                        logger.debug(ActionSet.recipe['text'][span["start"]:span["end"]])
                        second_pass_idx[idx] = sent_id

            except TypeError:  # For the sentences that were omitted in the PSV
                if fz.fuzz.partial_ratio(action_set_psv[0], ActionSet.recipe["text"][span["start"]: span["end"]]) > 90:
                    if span["label"].lower() in stv.code:
                        e = ne.Code(span)
                    elif span["label"].lower() in stv.entities:
                        e = ne.Entity(span)
                    # Create a pseudo dictionary for it
                    info = defaultdict(list)
                    info["sentence"] = action_set_psv[0]
                    # The rest entities will be empty according to defaultdict
                    try:
                        action_sets[sent_id].append(ActionSet(info, anchor_code=e, **relations))
                    except TypeError:
                        action_sets[sent_id].append(ActionSet(action_set_psv, anchor_code=e))
    logger.debug(pformat(action_sets))
    for k, v in action_sets.items():
        for s in v:
            logger.debug(s.get_word())
            for d in s.get_dependents_():
                try:
                    logger.debug("   " + d)
                except TypeError:
                    pass

    # Then fill in the rest with the sentence they belong and their closest Action PSV
    # Match if the anchor_code is there
    logger.debug(pformat(action_sets))
    if second_pass_idx:
        for idx, span in enumerate(ActionSet.recipe["spans"]):
            try:
                for action_set in action_sets[second_pass_idx[idx]]:
                    if action_set.is_parent_or_child(span):
                        if span["label"].lower() in stv.code:
                            e = ne.Code(span)
                        elif span["label"].lower() in stv.entities:
                            e = ne.Entity(span)
                        # Get all relations
                        # dep_relations, mem_relations = get_relations(span)
                        action_sets[second_pass_idx[idx]].append(ActionSet(action_set.action_set_psv, anchor_code=e,
                                                                           **get_relations(span)[0]))
                        check = False
                        break
                else:
                    # Check whether there is a neighbouring sentence which is linked to it instead
                    check = True
                    try:
                        for action_set in action_sets[second_pass_idx[idx + 1]]:
                            if action_set.is_parent_or_child(span):
                                if span["label"].lower() in stv.code:
                                    e = ne.Code(span)
                                elif span["label"].lower() in stv.entities:
                                    e = ne.Entity(span)
                                # Get all relations
                                # dep_relations, mem_relations = get_dep_mem_relations(span)
                                action_sets[second_pass_idx[idx + 1]].append(ActionSet(action_set.action_set_psv, anchor_code=e,
                                                                                       **get_relations(span)[0]))
                                check = False
                                break
                        else:
                            check = True
                    except (IndexError, KeyError):
                        check = True
                    try:
                        if check:
                            for action_set in action_sets[second_pass_idx[idx - 1]]:
                                if action_set.is_parent_or_child(span):
                                    if span["label"].lower() in stv.code:
                                        e = ne.Code(span)
                                    elif span["label"].lower() in stv.entities:
                                        e = ne.Entity(span)
                                    # Get all relations
                                    # dep_relations, mem_relations = get_dep_mem_relations(span)
                                    action_sets[second_pass_idx[idx - 1]].append(ActionSet(action_set.action_set_psv, anchor_code=e,
                                                                                           **get_relations(span)[0]))
                                    check = False
                                    break
                    except (IndexError, KeyError):
                        check = True
                # If failed to completely find a linking Code in its and neighbouring sentences then find the closest distance
                # in its sentence
                if check:
                    # if the sentence has AT's if not the neighbours
                    if  action_sets[second_pass_idx[idx]]:
                        cands = list()
                        for idx_, action_set in enumerate(action_sets[second_pass_idx[idx]]):
                            cands.append(action_set.get_distance(span))
                        # Get all relations
                        # dep_relations, mem_relations = get_dep_mem_relations(span)
                        if span["label"].lower() in stv.code:
                            e = ne.Code(span)
                        elif span["label"].lower() in stv.entities:
                            e = ne.Entity(span)
                        action_sets[second_pass_idx[idx]].append(ActionSet(
                            action_sets[second_pass_idx[idx]][np.argmin(cands)].action_set_psv, anchor_code=e,
                            **get_relations(span)[0]))
                        check = False
                        continue
                    else:
                        try:
                            for idx_, action_set in enumerate(action_sets[second_pass_idx[idx+1]]):
                                cands.append(action_set.get_distance(span))
                            # Get all relations
                            # dep_relations, mem_relations = get_dep_mem_relations(span)
                            if span["label"].lower() in stv.code:
                                e = ne.Code(span)
                            elif span["label"].lower() in stv.entities:
                                e = ne.Entity(span)
                            action_sets[second_pass_idx[idx+1]].append(ActionSet(
                                action_sets[second_pass_idx[idx+1]][np.argmin(cands)].action_set_psv, anchor_code=e,
                                **get_relations(span)[0]))
                            check = False
                            continue
                        except IndexError:
                            pass
                        try:
                            if check:
                                for idx_, action_set in enumerate(action_sets[second_pass_idx[idx-1]]):
                                    cands.append(action_set.get_distance(span))
                                # Get all relations
                                # dep_relations, mem_relations = get_dep_mem_relations(span)
                                if span["label"].lower() in stv.code:
                                    e = ne.Code(span)
                                elif span["label"].lower() in stv.entities:
                                    e = ne.Entity(span)
                                action_sets[second_pass_idx[idx-1]].append(ActionSet(
                                    action_sets[second_pass_idx[idx-1]][np.argmin(cands)].action_set_psv, anchor_code=e,
                                    **get_relations(span)[0]))
                                check = False
                                continue
                        except IndexError:
                            pass
                if check:
                    raise IndexError
            except KeyError:
                continue

    # Do the Join dictionary
    # It can be between Actions or Code, there can be one to many representations
    logger.debug(pformat(action_sets))
    for k, v in action_sets.items():
        for s in v:
            logger.debug(s.get_word())
            for d in s.get_dependents_():
                try:
                    logger.debug("   " + d)
                except TypeError:
                    pass

    ActionSet.action_sets = action_sets
    del action_sets
        # except RuntimeError:
        #     print("fefe")

    # Index relations for the ActionSets
    for action_set in list(itertools.chain.from_iterable(list(ActionSet.action_sets.values()))):
        for k, relations in action_set.relations.items():
            for relation in relations:
                if action_set.anchor_code.get_id() == relation["head"]:
                    for i, j, action_set_pair in ActionSet.get_idx(ActionSet.action_sets):
                        if action_set_pair.anchor_code.get_id() == relation["child"]:
                            relation["dest"] = (i, j)
                elif action_set.anchor_code.get_id() == relation["child"]:
                    for i, j, action_set_pair in ActionSet.get_idx(ActionSet.action_sets):
                        if action_set_pair.anchor_code.get_id() == relation["head"]:
                            relation["start"] = (i, j)

    logger.debug("\n\n\n ---- \n \n")
    # sentence_counter = 0
    for span in ActionSet.recipe["spans"]:  # Start normally
        # Exclude the already Code detected from above
        # Members may not be assigned explicitly, hence there needs to be a recursive search
        for action_set in list(itertools.chain.from_iterable(list(ActionSet.action_sets.values()))):
            if action_set.is_span(span):
                break
        else:
            # Find the ActionSet they belong to
            # Recursive with relation
            e = assign_ne(span)
            try:
                # print("Create LU")
                # print(e.out())
                # print(e.get_id())
                i, j = recrv_find_action_set(span)
                # print(e.out() + "  -->  " + ActionSet.action_sets[i][j].anchor_code.out())
                ActionSet.action_sets[i][j].add_lu(e)
            except (AttributeError, NotImplementedError, RecursionError):
                # print(span["label"], ActionSet.recipe["text"][span["start"]: span["end"]])
                # print(ActionSet.recipe["text"][span["start"]: span["end"] + 20])
                # remove this span from relations
                drop = list()
                for idx, relation in enumerate(ActionSet.recipe["relations"]):
                    if relation["head"] == span["token_end"] or relation["child"] == span["token_end"]:
                        drop.append(idx)
                for idx in reversed(drop):
                    del ActionSet.recipe["relations"][idx]

    logger.debug('\n\n\n')
    logger.debug('Fill in Entities')
    logger.debug('\n\n\n')
    # Fill all entities
    while True:
        # Iterate through all the Action Sets and ensure that their LUs have been filled
        for action_set in list(itertools.chain.from_iterable(list(ActionSet.action_sets.values()))):
            if not action_set.entity_matched:
                break
        else:
            break
        for action_set in list(itertools.chain.from_iterable(list(ActionSet.action_sets.values()))):
            logger.debug(" process... ")
            # print(" process... ")
            logger.debug(action_set.anchor_code.out())
            # print(action_set.anchor_code.out())
            try:
                action_set.fill_in_entities()
            except NotImplementedError:
                continue

    # Fill in the additional links
    for action_set in list(itertools.chain.from_iterable(list(ActionSet.action_sets.values()))):
        action_set.fill_in_links()

    # Resolve the Or/join sharing relations, update their LUs where appropriate
    for i, j, action_set in ActionSet.get_idx(ActionSet.action_sets):
        # In the join scenario all the LUs are duplicated
        for relation in action_set.relations["join"]:
            try:
                action_set_pair = ActionSet.action_sets[relation["dest"][0]][relation["dest"][1]]
                if action_set_pair == action_set:
                    continue
                for lu in action_set_pair.lu:
                    lu.add_dep(tuple((i, j)))
                    action_set.add_lu(lu)
            except KeyError:
                continue
        # We do not want to include Actuations, they will be different
        if action_set.anchor_code.get_label() == "action":
            continue
        for relation in action_set.relations["or"]:
            try:
                action_set_pair = ActionSet.action_sets[relation["dest"][0]][relation["dest"][1]]
                # We do not want to include Actuations, they will be different
                if action_set_pair == action_set or action_set_pair.anchor_code.get_label() == "action":
                    continue
                for lu in action_set_pair.lu:
                    try:
                        # If it's a modifier to code it should not be duplicated, it's unique
                        if lu.get_modifier() == action_set_pair.anchor_code.out():
                            continue
                    except AttributeError:
                        pass
                    lu.add_dep(tuple((i, j)))
                    action_set.add_lu(lu)
            except KeyError:
                continue

    # Fuse relations, for encompassing printing
    for _, _, action_set in ActionSet.get_idx(ActionSet.action_sets):
        for relation_type in ["or", "join"]:
            # Fuse the dependency links for join and `or` that don't have
            for relation in action_set.relations[relation_type]:
                try:
                    action_set_pair = ActionSet.action_sets[relation["dest"][0]][relation["dest"][1]]
                    if action_set_pair == action_set:
                        continue
                    if action_set.relations["depend"] and not action_set_pair.relations["depend"]:
                        # The `dest` should be the same
                        action_set_pair.relations["depend"] = action_set.relations["depend"]
                    elif not action_set.relations["depend"] and action_set_pair.relations["depend"]:
                        action_set.relations["depend"] = action_set_pair.relations["depend"]
                except KeyError:
                    continue
                # Fill the ActionSets that are depended upon
                for _, _, action_set_2 in ActionSet.get_idx(ActionSet.action_sets):
                    relations_to_add = list()
                    for relation_2 in action_set_2.relations["depend"]:
                        try:
                            # Check if one of them is depended on action_set
                            if ActionSet.action_sets[relation_2["dest"][0]][relation_2["dest"][1]] == action_set:
                                # If none is depended on the other
                                for _, _, action_set_pair_2 in ActionSet.get_idx(ActionSet.action_sets):
                                    for relation_pair_2 in action_set_pair_2.relations["depend"]:
                                        if ActionSet.action_sets[relation_pair_2["dest"][0]][relation_pair_2["dest"][1]] == \
                                            action_set_pair:
                                            break
                                    else:
                                        continue
                                    break
                                else:
                                    # Then copy the dependency across
                                    # Create the new relationship
                                    relation_to_add = copy.copy(relation_2)
                                    relation_to_add["child"] = action_set_pair.anchor_code.get_id()
                                    relation_to_add["dest"] = ActionSet.get_action_set_position(action_set_pair)
                                    relation_to_add["child_span"] = action_set_pair.anchor_code.span
                                    relations_to_add.append(relation_to_add)
                                break
                        except KeyError:
                            continue
                    for relation_to_add in relations_to_add:
                        action_set_2.relations["depend"].append(relation_to_add)


    # Print Output
    recipe_out = copy.copy(ActionSet.recipe["text"])
    span_label = dict()
    for span in reversed(ActionSet.recipe["spans"]):
        for action_set in list(itertools.chain.from_iterable(list(ActionSet.action_sets.values()))):
            # Check for Code
            try:
                if action_set.is_span(span):
                    recipe_out = recipe_out[:span["start"]] + action_set.get_anchor_code_out() + recipe_out[span["end"]:]
                    for token in range(span["token_start"], span["token_end"] + 1):
                        span_label[token] = action_set.get_anchor_code_out()
                    break
                # Check for LUs
                elif action_set.get_lu_by_id(span["token_end"]):
                    recipe_out = recipe_out[:span["start"]] + action_set.get_lu_out(span["token_end"]) + recipe_out[span["end"]:]
                    for token in range(span["token_start"], span["token_end"] + 1):
                        span_label[token] = action_set.get_lu_out(span["token_end"])
                    break
            except TypeError:
                continue

    logger.debug('\n\n')
    print('\n\n')
    for sent_in, sent_out in zip(ActionSet.recipe["text"].split("."), recipe_out.split(".")):
        if not sent_in:
            continue
        logger.debug(sent_in)
        logger.debug('\n\n')
        print(sent_in, end='\n\n')
        logger.debug(pformat(sent_out, width=180))
        pprint(sent_out, width=180)
        logger.debug('\n')
        print()

    out_dict = {"text_in": ActionSet.recipe["text"],"text_out": recipe_out, "tokens":ActionSet.recipe["tokens"],
                "tokens_label":span_label}
    return out_dict

if __name__ == '__main__':

    Annotator = "Phil"
    # Start with recipes
    # Prodigy Annotations
    with open("/home/chroner/PhD_remote/FoodBase/data/Processed/DB/Final/Dinner/" + Annotator + "/prodigy_3.json", "r") as fdrs:
        d_out = json.load(fdrs)
    # PSVs
    psvs = "/home/chroner/PhD_remote/FoodBase/data/Processed/PSV_2.0_" + Annotator + "/"
    onlyfiles = [f for f in listdir(psvs) if isfile(join(psvs, f))]
    # for recipe in d_out:
    #     if recipe["meta"]["id"] == "4recipe329":  # 4recipe435, 4recipe329
    #         for file in onlyfiles:
    #             if file.split(".")[0] == recipe["meta"]["id"]:
    #                 # ut = recipe["text"]
    #                 break
    #         else:
    #             continue
    #         break
    omit = ["4recipe353", "4recipe641", "4recipe338", "4recipe618", "4recipe735", "4recipe611", "4recipe15", "4recipe382", "4recipe212", "4recipe406", "4recipe285", "4recipe247"]
    for recipe in d_out:
        print(recipe["meta"]["id"])
        # Skip erroring recipes
        if recipe["meta"]["id"] in omit:
            continue  # 4recipe512, 4recipe338, 4recipe530
        for file in onlyfiles:
            if file.split(".")[0] == recipe["meta"]["id"]:
                logger.debug(recipe["meta"]["id"])
                # if recipe["meta"]["id"] == "1recipe1697":
                process(recipe, join(psvs, file))
                break
        else:
            print("not found")
