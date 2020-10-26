import re

from bs4 import BeautifulSoup
from collections import defaultdict


def xml_to_soup(xml_string):
    '''
    Create a beautiful soup object from a MetaMap XML string.

    '''
    return BeautifulSoup(xml_string[xml_string.find('<?xml'):],'lxml')


def parse_candidate(candidate, candidatematch, map_dict):
    '''
    For a given MetaMap candidate, extract relevant information
    and store in map_dict for future reference.

    Arguments:
        candidate (beautifulsoup object): A BS object containing
                                          the candidate
                                          metainformation

        candidatematch (str): The substring of the utterance that
                              triggered a MetaMap candidate match.

        map_dict (defaultdict): A dictionary providing mappings
                                from candidate matches to their
                                metainformation.

    Returns:
        map_dict (defaultdict): An updated version of the input.

    '''
    cui = candidate.find('candidatecui').get_text()
    semtype = candidate.find('semtype').get_text()
    negated = candidate.find('negated').get_text()
    preferred = candidate.find('candidatepreferred').get_text()

    map_dict[candidatematch]['CUI'] = cui
    map_dict[candidatematch]['SemanticType'] = semtype
    map_dict[candidatematch]['Negated'] = negated
    map_dict[candidatematch]['PreferredTerm'] = preferred

    return map_dict


def handle_multi_word_candidates(candidatematch, start, extra, extra_d):
    '''
    A method used to track special cases where the MetaMap returned candidate
    does not match the original text. e.g. 'heart' and 'attack' are two
    seperate lexical units, but one semantic unit 'heart attack'. As such
    they are tracked and used to reference metainformation which would
    otherwise only be assigned to a single word.

    Arguments:
        candidatematch (str): The substring of the utterance that
                              triggered a MetaMap candidate match.

        start (str): The index location of the substring. Possibly useful
                     in cases where identical substrings may have different
                     syntactical/semantic mappings.

        extra (str): A substring of `utt_text` that is being used as a key
                     to reference the full utterance and location.

        extra_d (defaultdict): A container for tracking the lexical-semantic 
                               string differences.

    Returns:
        extra_d (defaultdict): An updated version of the input.

    '''
    extra = extra.lower()
    if extra not in extra_d or start not in extra_d[extra]:
        extra_d[extra][start]['T'] = candidatematch

    return extra_d


def handle_syntax_units(inputmatch, output_d, extra_d, map_dict,
                        utt_text, lexcat):
    '''
    For each syntactical unit identified by MetaMap, aggregate the stored
    metainformation and create a new row entry in the output storage
    dictionary.

    Arguments:
        inputmatch (str): The full string that led to a MetaMap match

        output_d (defaultdict): A container for holding the final output
                                information.

        extra_d (defaultdict): A container for tracking the lexical-semantic 
                               string differences.

        map_dict (defaultdict): A dictionary providing mappings
                                from candidate matches to their
                                metainformation.

        utt_text (str): The text of the MetaMap utterance.

        lexcat (str): The part of speech as tagged by MetaMap

    Returns:
        output_d (defaultdict): An updated version of the input.

    '''
    match = inputmatch.lower()
    match_loc = utt_text.lower().find(match)
    keys = ('CUI', 'SemanticType', 'PreferredTerm', 'Negated')

    for word in inputmatch.split():
        output_d['FullLexicalUnit'].append(inputmatch)
        output_d['Word'].append(word)
        output_d['LexicalCategory'].append(lexcat)

        if inputmatch.lower() in map_dict:
            for key in keys:
                output_d[key].append(map_dict[match][key])

        elif match in extra_d and match_loc in extra_d[match]:
            map_T = map_dict[extra_d[match][match_loc]['T']]
            for key in keys:
                output_d[key].append(map_T[key])
            
        else:
            for key in keys:
                output_d[key].append('')

    return output_d


def extract_results_from_soup(soup):
    '''
    Given a beautiful soup object of a MetaMap xml output, parse the results
    and return a dictionary of ordered metainformtion.

    Arguments:
        soup (beautifulsoup object): A beautiful soup object created from 
                                     a MetaMap xml output string.

    Returns:
        output_d (defaultdict): A container for holding the final output
                                information.

        extra_d (defaultdict): A container for tracking the lexical-semantic 
                               string differences.

    '''
    utterances = soup.find_all('utterance')
    extra_d = defaultdict(lambda: defaultdict(dict))
    output_d = defaultdict(list)
    for utterance in utterances:
        utt_text = utterance.find('utttext').get_text()
        map_dict = defaultdict(dict)
        candidates = utterance.find_all('candidate')
        for candidate in candidates:
            candidatematch = candidate.find('candidatematched')
            candidatematch = candidatematch.get_text().lower()
        
            length = int(candidate.find('length').get_text())
            can_len = len(candidatematch)
            if len(candidatematch.split()) > 1 or can_len != length:
                start = int(candidate.find('startpos').get_text())

                for extra in utt_text[start:start+length].split():
                    extra_d = handle_multi_word_candidates(candidatematch,
                                                           start, extra,
                                                           extra_d)
            if candidatematch not in map_dict:
                map_dict = parse_candidate(candidate, candidatematch, map_dict)

        units = utterance.find_all('syntaxunit')
        for unit in units:
            try:
                lexcat = unit.find('lexcat').get_text()
            except AttributeError:
                lexcat = 'punc'

            inputmatch = unit.find('inputmatch').get_text()
            output_d = handle_syntax_units(inputmatch, output_d,
                                           extra_d, map_dict,
                                           utt_text, lexcat)
    return output_d, extra_d
