import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForMaskedLM

# Load tokenizers for GPT-2 and BERT
gpt2_tokenizer = AutoTokenizer.from_pretrained('gpt2')
bert_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Set padding token for GPT-2 tokenizer
gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token

# Load models for GPT-2 and BERT
gpt2_model = AutoModelForCausalLM.from_pretrained('gpt2')
bert_model = AutoModelForMaskedLM.from_pretrained('bert-base-uncased')

# Sample nested list of text data
text_samples2 = [
    ["This is a sample sentence.", "Here is another example."],
    ["Text for the second batch.", "Another text sample."]
]
# Sample text data
text_samples = [
    ["As I sit down to write this letter, I am surrounded by the very technology that has been a topic of debate among people of all ages. Computers have become an integral part of our daily lives, and it's hard to imagine a world without them. However, not everyone agrees that this is a good thing. As a high school student, I believe that computers have a profound impact on our society, and I would like to share my thoughts on the matter.Firstly, computers have greatly improved our educational system. They have made it possible for us to learn about faraway places and people, cultures, and ideas that we may not have had access to otherwise. With just a few clicks, we can explore the world without leaving our classrooms. This exposure has broadened our perspectives and helped us become more empathetic and understanding individuals.Computers have also taught us valuable skills, such as problem-solving, critical thinking, and time management. These skills are essential in today's fast-paced world, and computers have helped us develop them from a young age. Moreover, computers have made it possible for people to communicate with each other instantly, regardless of their geographical location. This has broken down barriers and brought people together like never before.However, I understand that some experts are concerned about the negative effects of excessive computer use. They argue that people are spending too much time staring at screens and less time engaging in physical activities, socializing, and enjoying nature. I agree that this is a valid concern, and it's essential that we find a balance between technology and other aspects of our lives.In conclusion, while computers have their drawbacks, I firmly believe that their benefits far outweigh the negatives. They have opened up new opportunities for education, communication, and personal growth. As a society, we should strive to use technology in a way that enhances our lives, rather than controls them. I encourage my fellow students, parents, and community members to use technology responsibly and find a balance that works for everyone.Thank you for taking the time to consider my opinion. I hope that my thoughts will inspire others to think critically about the impact of computers on our society.Sincerely, Thomas"],
    ["Computers. One of the much enjoyed pieces of technology. But it is also one of the many distractions. Many people ponder if computers are really beneficial. I am one of those thinkers. I think that computers don't always benefit society. They have many, many distractions such as facebook, online games, and even inappropriate images and videos. If you really think of it, are computers as beneficial as we think? My friend, Tom was just told about facebook, and got a membership. It just started out as a 'na big deal' kind of thing. She went online often, just to check on her status and if she had any messages in her inbox, and that's It. But then, she was full-blown addicted. She applied for ALL the clubs and groups, and she started getting behind in school. She was obtaining a plethora of D's on most of her tests after not studying because she was on for almost six hours everyday. Marie would come home and not even bother to do homework, but would just immediately go online. After one month of being a member, she was failing All of her classes, and her mom took her computer priveleges away. See, she was distracted by one, little thing, and her whole life was almost destroyed. But, mom, i'm in the middle of a game! And I'm winning! Tom says vistoriously. Tom loves online games, and he wins almost 42 of the time. But, that's not the point. He plays games All the TIME. Tom's game playing gets in the way with the interaction of him and his family. He Cant't played eith his four year old sister, Richard since she was two years old. his mother also often eats dinner alone. Sometimes, he even views inappropriate pictures when his mom is out shopping. When he's playing a game, a girl Texts him, Looks at my pic. They're hot! and he looks at them. He thinks they're harmless but that's not what his mother thinks. his gaming has gotten in the way of his family and social life. most people think that computers are beneficial because you can find cool information. But, you can get ALL kinds of information from books. Instead of typing on computers, you could just search for it in a library. It takes less time, and you won't get carpal tunnel as easily. See? Computers aren't as beneficial as most think. There are MANY kinds of distraction, and these are only a few."]
]


# Function to get log probabilities from GPT-2 model
def get_log_probs_gpt2(model, input_ids, attention_mask):
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        
        # Shift tokens left for causal language model like GPT-2
        shifted_input_ids = input_ids[:, 1:].contiguous()
        shifted_logits = logits[:, :-1, :].contiguous()
        
        # Get the log probabilities for the shifted logits
        log_probs = torch.nn.functional.log_softmax(shifted_logits, dim=-1)
        
        # Gather the log probs for the actual next tokens
        next_token_log_probs = log_probs.gather(2, shifted_input_ids.unsqueeze(-1)).squeeze(-1)
        return next_token_log_probs

# Function to get log probabilities from BERT model (treated as a masked language model)
def get_log_probs_bert(model, input_ids, attention_mask):
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        
        # Get the log probabilities for the logits
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        
        # Shift tokens left for consistency
        shifted_input_ids = input_ids[:, 1:].contiguous()
        shifted_log_probs = log_probs[:, :-1, :]
        
        # Gather the log probs for the actual tokens (shifted)
        next_token_log_probs = shifted_log_probs.gather(2, shifted_input_ids.unsqueeze(-1)).squeeze(-1)
        return next_token_log_probs

# Function to compute the entropy score for each element in the nested list
def compute_entropy_score(text_samples):
    entropy_scores = []

    for batch in text_samples:
        # Tokenize the text samples
        gpt2_encoding = gpt2_tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
        bert_encoding = bert_tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
        
        # Extract the attention masks
        gpt2_attention_masks = gpt2_encoding['attention_mask']
        bert_attention_masks = bert_encoding['attention_mask']
        
        # Get log probabilities for each token
        gpt2_log_probs = get_log_probs_gpt2(gpt2_model, gpt2_encoding['input_ids'], gpt2_attention_masks)
        bert_log_probs = get_log_probs_bert(bert_model, bert_encoding['input_ids'], bert_attention_masks)
        
        # Apply weights using the attention masks
        gpt2_weights = gpt2_attention_masks[:, 1:].float()  # Ignore the first token (shifted)
        bert_weights = bert_attention_masks[:, 1:].float()  # Ignore the first token (shifted)
        
        # Ensure weights are aligned correctly
        min_length = min(gpt2_weights.size(1), bert_weights.size(1))
        gpt2_weights = gpt2_weights[:, :min_length]
        bert_weights = bert_weights[:, :min_length]
        weighted_gpt2_log_probs = gpt2_log_probs[:, :min_length] * gpt2_weights
        weighted_bert_log_probs = bert_log_probs[:, :min_length] * bert_weights
        
        # Compute cross-entropy loss for each sequence in the batch
        cross_entropy_loss = -torch.sum(weighted_gpt2_log_probs * weighted_bert_log_probs, dim=1)
        
        # Compute the final entropy score for each sequence
        batch_entropy_scores = torch.sum(cross_entropy_loss) / torch.sum(gpt2_weights)
        entropy_scores.append(batch_entropy_scores.item())
    
    return entropy_scores

# Compute the entropy scores for the nested list of text samples
entropy_scores = compute_entropy_score(text_samples)
print("Entropy Scores:\n", entropy_scores)