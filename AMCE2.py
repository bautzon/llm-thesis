import torch
from transformers import BertTokenizer, BertForMaskedLM, GPT2Tokenizer, GPT2LMHeadModel

# Load BERT model and tokenizer
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertForMaskedLM.from_pretrained('bert-base-uncased')

# Load GPT-2 model and tokenizer
gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')

def preprocess_text(text):
    return text.strip()

def tokenize_text(text, tokenizer):
    return tokenizer.encode(text, return_tensors='pt')

def top_p_sampling(logits, top_p=0.5, top_k=50):
    logits = torch.clamp(logits, min=-1e9, max=1e9)  # Clamp logits to avoid NaN
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
    
    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs > top_p
    if torch.any(sorted_indices_to_remove):
        top_k = min(top_k, torch.sum(~sorted_indices_to_remove).item())
    
    # Ensure at least one token is kept
    sorted_indices_to_remove[..., :top_k] = False
    indices_to_remove = sorted_indices[sorted_indices_to_remove]
    logits[indices_to_remove] = float('-inf')
    
    logits = torch.clamp(logits, min=-1e9, max=1e9)  # Clamp logits again to avoid NaN
    
    return torch.distributions.Categorical(logits=logits).sample()

def compute_log_probs(model, tokenizer, text, top_p=0.5, top_k=50):
    inputs = tokenize_text(text, tokenizer)
    with torch.no_grad():
        outputs = model(inputs, output_attentions=True)
    logits = outputs.logits[0, -1, :]

    print(f"Logits before sampling: {logits}")

    # Check for NaN values in logits
    if torch.isnan(logits).any():
        print("NaN detected in logits before sampling.")

    sampled_token = top_p_sampling(logits, top_p, top_k)
    log_prob = torch.log_softmax(logits, dim=-1)[sampled_token]

    # Check for NaN values in log probabilities
    if torch.isnan(log_prob).any():
        print("NaN detected in log probabilities.")
    
    attention_mask = outputs.attentions[-1].mean(dim=1).squeeze()  # Average attention masks
    return log_prob, attention_mask

def weighted_attention_sum(attention_mask, top_w=1.0):
    return attention_mask.sum() * top_w

def cross_entropy(log_probs_1, log_probs_2):
    return -(log_probs_1 * log_probs_2).sum()

def compute_cross_entropy_for_list(sample_list, top_w=1.0, top_p=0.9, top_k=100):
    results = []
    
    for text in sample_list:
        text = preprocess_text(text)
        
        log_prob_bert, attention_mask_bert = compute_log_probs(bert_model, bert_tokenizer, text, top_p, top_k)
        log_prob_gpt2, attention_mask_gpt2 = compute_log_probs(gpt2_model, gpt2_tokenizer, text, top_p, top_k)
        
        weighted_attention = weighted_attention_sum(attention_mask_bert, top_w) + weighted_attention_sum(attention_mask_gpt2, top_w)
        print(f"attention mask bert {attention_mask_bert}")
        cross_entropy_score = cross_entropy(log_prob_bert, log_prob_gpt2) + weighted_attention
        
        results.append(cross_entropy_score.item())
    
    return results

def main():
    top_k = 640
    top_p = 0.95
    top_w = 0.003
    #GPT4
    sample1 = ["Dear Editor,  I am writing to express my thoughts on the prevalent debate concerning the impact of computers on society—a topic that captivates my generation as we grow in this digital age.  Acknowledging that every coin has two sides, I agree that computers, like all other human innovations, have both positive and negative effects on individuals. However, my conviction leans towards the belief that the benefits of utilizing computers outweigh the supposed disadvantages.  The most striking attribute of computer technology is its capacity to cement gaps in knowledge and communication. Through computers, we have the extraordinary ability to traverse geographical borders and explore diverse cultures without leaving our desks. There's an infinite world of knowledge waiting to be uncovered at the click of a button, making learning an accessible and enriching experience.   Furthermore, computers have enhanced our ability to communicate. With the help of online applications, we can easily stay connected with our loved ones, irrespective of distance, and even meet new people from societies we remain physically detached from. This advancement in technology has also fortified the cause of globalization, fostering mutual understanding among nations.   On the contrary, critics argue that this increasing reliance on computers encourages sedentary behavior, lessens physical interaction, and reduces the time spent in nature. However, I believe these concerns can be addressed by advocating a balanced lifestyle instead of eschewing technology outright. Schools should implement policies that encourage students to balance their screen time with other activities such as sports, arts, and social interactions to insure well-rounded development.  It's also important to note that computers are a tool, and like all tools, their effect on individuals is largely dependent on how they are used. Just as you can use a hammer to either build a house or destroy a wall, computers can be used to either augment human potentials or foster unhealthy habits. It's our responsibility to harness the power of computers in a manner that aligns with our physical and mental wellbeing.  Thus, I urge readers to embrace computer technology while remaining mindful of maintaining a balanced lifestyle. Let's maximize the advantages while keeping potential detriments in check. After all, as the wielders of this tool, we hold the power to determine how it shapes us as individuals and a society.  Thank you for your attention.  Warm regards,  Sam"]
    #llama2
    sample2 = ["Dear Editor,I am writing to express my opinion on the effects computers have on people. As a high school student, I have seen firsthand how computers have impacted my education and daily life. I strongly believe that computers have had a positive effect on society, and I would like to share some reasons why.First and foremost, computers have improved hand-eye coordination and fine motor skills. Playing video games, using computer software, and typing on a keyboard all require manual dexterity and attention to detail. These activities help to develop important cognitive skills that can benefit people of all ages. Additionally, computers have opened up new opportunities for people to learn about faraway places and cultures. With the internet at our fingertips, we can explore different countries, historical events, and scientific discoveries with just a few clicks.Furthermore, computers have enabled people to communicate with one another in new and innovative ways. Social media platforms, messaging apps, and video conferencing software have made it easier for people to connect with others across the globe. This has fostered global understanding, collaboration, and cultural exchange.Of course, there are also concerns about the negative effects of computers on society. Some experts argue that people are spending too much time on their computers and less time exercising, enjoying nature, and interacting with family and friends. While it is true that excessive computer use can be harmful, I believe that this can be mitigated by setting limits and balancing technology use with other activities.In conclusion, I firmly believe that computers have had a positive effect on society. They have improved our cognitive skills, expanded our knowledge of the world, and enabled new forms of communication and collaboration. While it is important to be mindful of the potential negative effects, I believe that technology can be a powerful tool for good when used responsibly."]
    #Human
    sample3 = ["Dear News Paper Editor, I believe that using computers will benefit us in many ways like talking and becoming friends will others through websites like facebook and mysace. Using computers can help us find coordibates, locations, and able ourselfs to millions of information. Also computers will benefit us by helping with jobs as in planning a house plan and typing a 10 page report for one of our jobs in less than writing it. Now lets go into the wonder world of technology. Using a computer will help us in life by talking or making friends on line. Many people have myspace, facebooks, aim, these all benefit us by having conversations with one another. Many people believe computers are bad but how can you make friends if you can never talk to them? I am very fortunate for having a computer that can help with not only school work but my social life and how I make friends. Computers help us with finding our locations, coordibates and millions of information online. If we didn't go on the internet a lot we wouldn't know how to go onto websites that may help us with locations and coordinates like a place. Would you rather use a computer or be in a place. When your supposed to be vacationing in a place. Million of information is found on the internet. You can as almost every question and a computer will have it. Would you rather easily draw up a house plan on the computers or take 7 hours doing one by hand with ugly erazer marks all over it, you are garrenteed that to find a job with a drawing like that. Also when appling for a job many workers must write very long papers like a 500 word essay on why this job fits you the most, and many people I know don't like writing 200 words non-stopp for hours when it could take them I hav an a computer. That is why computers we needed a lot now adays. I hope this essay has impacted your descion on computers because they are great machines to work with. The other day I showed my mom how to use a computer and she said it was the greatest invention sense sliced bread! Now go out and buy a computer to help you chat online with friends, find locations and millions of information on one click of the button and help your self with getting a job with neat, prepared, printed work that your boss will love."]
    #GPT3 
    sample4 = ["Dear Editor,  As a high school student, I want to share my thoughts on the effects of computers on people in today's society. While it is true that computers have become an integral part of our lives and offer various benefits, I believe there are also drawbacks that need to be considered.  Undoubtedly, computers have revolutionized the way we learn, communicate, and interact with others. They provide us with opportunities to enhance our knowledge, connect with people from different backgrounds, and even improve our hand-eye coordination through various games and activities. However, it is important to acknowledge that excessive use of computers can have negative consequences on individuals and society as a whole.  One major concern is the impact of prolonged computer use on our physical health. Many individuals spend hours sitting in front of screens, leading to a sedentary lifestyle that can contribute to various health issues such as obesity, eye strain, and poor posture. Additionally, the excessive use of computers can also affect our mental well-being by isolating us from real-life interactions and experiences.  Moreover, the increasing reliance on computers for information and communication may hinder our ability to develop essential skills such as critical thinking, problem-solving, and social interaction. While computers can provide us with a wealth of knowledge, it is important to balance technology use with real-world experiences to ensure a well-rounded and fulfilling life.  In conclusion, I believe that while computers offer many advantages, it is crucial for individuals to use them in moderation and prioritize physical activity, face-to-face interactions, and time spent in nature. By finding a balance between technology and real-life experiences, we can truly benefit from the positive aspects of computers without sacrificing our health and well-being.  Sincerely, Samantha"]
    #GPT4
    sample5 = ["Dear Editor,  I hope this letter finds you well. As a high school student, born and bred in the digital age, I would like to share my thoughts on the growing trend of computer usage and its impact on people and society.  Firstly, it is undeniable that computers have transformed our lives and brought about a revolution in how we acquire knowledge, communicate, and coordinate everyday tasks. They have made it possible to learn about faraway places, to connect with people globally in an instant, and to work more efficiently. With the advantages of state-of-the-art technology, we have been able to improve our hand-eye coordination and broaden our understanding of complex issues at our fingertips.  However, as the saying goes, \"Every rose has its thorn,\" so does the extensive use of computers in our lives. Concerns have been raised that an over-reliance on computers could lead to less physical activity, limited interaction with nature, and reduced quality time with family and friends, which are all indispensable aspects of human life.  This brings me to my main point. While computers have a substantive positive effect on society, their usage must be regulated and balanced. We should take advantage of the resources technology provides without letting it consume our entire lives. It is important to set boundaries and know when to switch off our devices.    By integrating this balance, we can leverage technology to our advantage while also preserving and promoting physical health, appreciating the wonders of nature, and sharing valuable in-person experiences with our loved ones.   So, dear readers, let's ensure that we use these technological tools to augment and not replace our basic human needs. After all, computers are intended to serve us, not the other way around.  Yours sincerely,  Sam"]
    #Human
    sample6 = ["Computers a good because you can get infermation, you can play games, you can get pictures, But when you on the computer you might find something or someone that is bad or is viris. If ther is a vris you might want shut off the computers so it does not get worse. The are websites for kids, like games, there are teen games, there are adult games. Also pictures are bad for kids because most of the time they lead to inapropreit pictures. You should only look up infermation that you need not things like wepons or knifes. Also there are differnt kinds of companies like At&t Target. Target is a good place to get computers and so is At&t."]
    
  
   
    scores_sample1 = compute_cross_entropy_for_list(sample1, top_w=top_w, top_p=top_p, top_k=top_k)
    scores_sample2 = compute_cross_entropy_for_list(sample2, top_w=top_w, top_p=top_p, top_k=top_k)
    scores_sample3 = compute_cross_entropy_for_list(sample3, top_w=top_w, top_p=top_p, top_k=top_k)
    scores_sample4 = compute_cross_entropy_for_list(sample4, top_w=top_w, top_p=top_p, top_k=top_k)
    scores_sample5 = compute_cross_entropy_for_list(sample5, top_w=top_w, top_p=top_p, top_k=top_k)
    scores_sample6 = compute_cross_entropy_for_list(sample6, top_w=top_w, top_p=top_p, top_k=top_k)
    
    
    
    print(f"Cross-entropy scores for sample1: {scores_sample1}")
    print(f"Cross-entropy scores for sample2: {scores_sample2}")
    print(f"Cross-entropy scores for sample3: {scores_sample3}")
    print(f"Cross-entropy scores for sample4: {scores_sample4}")
    print(f"Cross-entropy scores for sample5: {scores_sample5}")
    print(f"Cross-entropy scores for sample5: {scores_sample6}")
  
if __name__ == "__main__":
    main()