import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import math

# Load the tokenizer and model once, and move the model to the GPU if available
tokenizer = AutoTokenizer.from_pretrained("../Meta-Llama-3.1-8B")
model = AutoModelForCausalLM.from_pretrained("../Meta-Llama-3.1-8B", torch_dtype=torch.bfloat16).eval()
torch.cuda.empty_cache()

# Check if a GPU is available
print(f' GPU available: {torch.cuda.is_available()}')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

def compute_ppl(loss_list):
    try:
        loss_sum = sum(loss_list)
        num_tokens = len(loss_list)
        ppl = math.exp(loss_sum / num_tokens)
        return ppl
    except:
        return None

def get_tokens(text):
    inputs = tokenizer(text, return_tensors="pt")
    tokens_list = inputs['input_ids'].tolist()[0]
    return tokens_list

def cumulative_text(conv_df, which_col = 'content'):
    # Need to change the colname to make it consistent with the col where the each message piece is stored: in this example it is 'content' col 
    cum_text = []
    for i in range(len(conv_df)):
        cum_text.append(' '.join(conv_df[which_col].iloc[:i+1]))
    conv_df['cum_full_text_seg_context'] = cum_text

    return cum_text[-1]

# Function to get log probabilities using the GPU
def get_llama3(text):
    try:
        # Tokenize input and move tensors to the GPU
        inputs = tokenizer(text, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}

        # No gradient computation needed since we're just inferring
        with torch.no_grad():
            outputs = model(**inputs, return_dict=True)

            logits = outputs.logits
            labels = inputs['input_ids']

            # remove the probability distribution of the token after the end token of the sequence
            logits = logits[..., :-1, :] 
            # remove the first token which is the <start of sequence>
            labels = labels[..., 1:]
            
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

            # Reshape loss to match the number of tokens
            loss = loss.view(labels.size()).cpu().tolist()

        return loss
    except Exception as e:
        return None
    
def get_cond_probs_range(row, conv_loss_list):
    try:
        
        token_start_index = row['token_start_index_seg_context']
        token_end_index = row['token_end_index_seg_context']

        # for the first message in the conversation, now that we remove the <start of sequence>, we should slice without the first token
        if token_start_index == 0:
            cond_probs = conv_loss_list[:token_end_index]
        # for the following message, we slice the same length of the message
        else:
            cond_probs = conv_loss_list[token_start_index-1:token_end_index]
        return cond_probs
    except:
        return None

def process_conv_df(conv_df):

    if conv_df['content'].isna().any():
        return conv_df, False, 'na_content'
    full_conversation = cumulative_text(conv_df)
    conv_df['cum_token_count_seg_context'] = conv_df['cum_full_text_seg_context'].apply(lambda x: len(get_tokens(x)))
    # in df, create a new col called token_index_range: 
    # for each user_id and segment_id_unique, compute the range using cum_token_count minus previous cum_token_count as the start, and cum_token_count as the end
    conv_df['token_start_index_seg_context'] = conv_df['cum_token_count_seg_context'].shift(fill_value=0)
    conv_df['token_end_index_seg_context'] = conv_df['cum_token_count_seg_context'] - 1

    # compute loss_list for each cum_full_text
    conv_loss_list = get_llama3(full_conversation)
    if conv_loss_list is None:
        return conv_df, False, 'exceed_limit'
    conv_df['mess_loss_list_seg_context'] = conv_df.apply(lambda x: get_cond_probs_range(x, conv_loss_list[0]), axis=1)
    conv_df['ppl_llama31_seg_context'] = conv_df['mess_loss_list_seg_context'].apply(lambda x: compute_ppl(x))
    
    return conv_df, True, 'success'

def process_and_write_csv(input_file, output_file, na_content_output_csv, exceed_limit_output_csv):
    # Open the input CSV in chunks for memory efficiency
    chunk_size = 10000  # Adjust chunk size depending on memory and file size

    # Variables to keep track of the current conv
    current_conv = None
    current_conv_rows = []
    conv_counter = 0  # Conversation counter

    # Open the output CSV in write mode initially (to overwrite if it already exists)
    with open(output_file, mode='a', newline='', encoding='utf-8') as f_out, \
    open(na_content_output_csv, mode='a', newline='', encoding='utf-8') as f_out_na, \
    open(exceed_limit_output_csv, mode='a', newline='', encoding='utf-8') as f_out_exceed_limit:
        # Initialize a flag to manage writing headers just once
        write_succeed_header = True
        write_na_header = True
        write_exceed_header = True
        
        # Iterate through the input CSV file chunk by chunk
        for chunk in pd.read_csv(input_file, chunksize=chunk_size):
            # Iterate through each row in the chunk
            for _, row in chunk.iterrows():
                conv_id = row['segment_id_unique']  # Extract conv_id

                # Check if the conv_id matches the current conv
                if conv_id == current_conv:
                    # Accumulate rows for the current conv
                    current_conv_rows.append(row)
                else:
                    # If a new conv is encountered, process the previous conv
                    if current_conv is not None:
                        # Convert the current conv rows to a DataFrame
                        conv_df = pd.DataFrame(current_conv_rows)
                        
                        # Apply your processing to the conv (if needed)
                        conv_df, process_succeed, reason = process_conv_df(conv_df)
                        # Write the conv to the output file
                        if process_succeed:
                            conv_df.to_csv(f_out, header=write_succeed_header, index=False, mode='a')
                            write_succeed_header = False
                        else:
                            if reason == 'exceed_limit':
                                conv_df.to_csv(f_out_exceed_limit, header=write_exceed_header, index=False, mode='a')
                                write_exceed_header = False
                            elif reason == 'na_content':
                                conv_df.to_csv(f_out_na, header=write_na_header, index=False, mode='a')
                                write_na_header = False

                        conv_counter += 1  # Increment conversation counter
                        # Log every 100 conversations
                        if conv_counter % 100 == 0:
                            print(f"Processed {conv_counter} segements so far...")

                    # Start processing the new conv
                    current_conv = conv_id
                    current_conv_rows = [row]  # Initialize with the current row

        # After the loop, don't forget to process and write the last conv
        if current_conv is not None:
            conv_df = pd.DataFrame(current_conv_rows)
            # Apply your processing to the conv (if needed)
            conv_df, process_succeed, reason = process_conv_df(conv_df)
            # Write the conv to the output file
            if process_succeed:
                conv_df.to_csv(f_out, header=write_succeed_header, index=False, mode='a')
            else:
                if reason == 'exceed_limit':
                    conv_df.to_csv(f_out_exceed_limit, header=write_exceed_header, index=False, mode='a')
                elif reason == 'na_content':
                    conv_df.to_csv(f_out_na, header=write_na_header, index=False, mode='a')

        print('process finished')

# Example usage
file_name = "HMC_merge_corrected"
input_csv = f'{file_name}.csv'   # Path to the original CSV file
output_csv = f'{file_name}_with_llama31_ppl.csv'  # Path to the new output CSV file
na_content_output_csv = f'{file_name}_na_content.csv'
exceed_limit_output_csv = f'{file_name}_exceed_limit.csv'
process_and_write_csv(input_csv, output_csv, na_content_output_csv, exceed_limit_output_csv)
