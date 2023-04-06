from tortus import Tortus
import pandas as pd
import os

def annotate_tweets(path, text_column = "cleaned_content", num_records=100, prev_annotations = None, additional_labels = [], tweets_longer_than_num = 10):
    df = pd.read_csv(path, index_col = "id")
    # filter for cleaned tweet greater than length 10
    df = df[df[text_column].str.count(' ').gt(tweets_longer_than_num-1)]
    df["annotate_text"] = "<b>Raw:</b> " + df["rawContent"] + "<br><b>Clean</b>: " + df[text_column]
    
    temp_path_list = os.path.dirname(path).split('/')
    temp_path_list[0] = 'first_annotation'
    outdirs = '/'.join(temp_path_list)
    basename = os.path.basename(path)
    os.makedirs(outdirs, exist_ok = True)
    output_path = os.path.join(outdirs, basename)
    
    if os.path.exists(output_path):
        print(f"Annotations already exist for: {output_path}, adding to these annotations")
        prev_annotations = pd.read_csv(output_path, index_col = "Unnamed: 0")
        
    tortus = Tortus(df, "annotate_text", num_records=num_records, annotations=prev_annotations, labels=["syntactic-english", "non-syntactic-english", "informal-english", "code-switched", "incidental-english","not-english"] + additional_labels)
    tortus.annotate()
    return tortus, output_path

def save_annotations(tortus, output_path):
    tortus.annotations.to_csv(output_path)