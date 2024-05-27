import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def calculate_perplexity(text):
    # Load pre-trained model (weights)
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model.eval()

    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    # Encode text
    encodings = tokenizer(text, return_tensors='pt')

    # Calculate loss
    input_ids = encodings.input_ids
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
        perplexity = torch.exp(loss)
    
    return perplexity.item()

# Example usage
text = "Title: The Case Against Censorship in Libraries  To the editor,  In reaction to the ever-present debate regarding censorship in libraries, I am prompted to raise my voice against any form of censorship that promotes the removal of diverse content from our local libraries. These bastions of intellectual exploration are pivotal to society's growth and must remain uncensored.   The author Katherine Paterson once wrote, “All of us can think of a book that we hope none of our children or any other children have taken off the shelf. But if I have the right to remove that book from the shelf -- that work I abhor -- then you also have exactly the same right and so does everyone else. And then we have no books left on the shelf for any of us. Consequently, if we begin to remove literature based on personal bias or discomfort, we potentially deny others the opportunity to learn, grow, and formulate dissenting or diverse perspectives.  Whilst I agree that certain materials can be viewed as offensive, I believe that rather than expunge them, we should foster an environment that encourages critical thinking and thoughtful discussion. For instance, from my own experience, reading To Kill a Mockingbird in my American Literature class incited discussions about racism and class divide. This experience was pivotal in my growth as a conscious and empathetic student, an experience that would have been missed in a censored library.  Furthermore, from a more universal perspective, a censorship-free library serves as a peaceful battleground for diverse ideas to collide, interact, and coexist. These institutions promote intellectual freedom, personal growth, and societal progress. They are the democratized training grounds for children to learn about the world in all its complexity and wonder, and for adults to continue their lifelong pursuit of knowledge.  By censoring libraries, we risk creating a society with tunnel vision, stunting intellectual growth, and silencing marginal voices. Multitudes of voices, countering biases and sensitivities, are crucial for a well-rounded education and must be preserved and encouraged in our libraries. Also, remember that libraries, unlike the internet, offer a curated collection of information resources, where reputable yet controversial materials can be found, that contribute to more informed perspectives.  We should also emphasize the role of library staff in guiding young, impressionable readers towards appropriate content, assist adult patrons in locating useful resources, and facilitating respectful dialogues about challenging topics.  Therefore, whilst it is understandable to want to protect ourselves and younger generations from potentially offensive content, we must resist the allure of censorship in our libraries. Instead, we should tirelessly work towards fostering a society of critical thinkers, respectful debaters, and open-minded learners, as that is the hallmark of a thriving democracy.  For, if we start to remove books, movies, or music that some find offensive from the shelves, we risk leaving them empty not just of content, but the rich diversity of thought that shapes and defines us as individuals and as a society."
text1 = "All throughout the world there are libraries.  Librairies have all different kinds of materiel stacked upon shelves.  Most of the material is helpful and enjoyable but some is not.  I think libraries should be able to have whatever they want on their shelves.  Yes, i understand some of this material may be inappropriate for a certain person but there is a way you can fix this.     If libraries have inappropriate books, movies, music, ect.  I think they should be able to keep them.  Just because some people do not that material does not mean everyone won't like it.  Everyone has their own style, everyone likes different things and has different opinions.  Isn't that what makes the world go around?       Adults may have a problem with the offensive material because they have children who go to the library.  I understand there concern but there can be a way to stop them from getting to the inappropriate material.  If libraries are going to have this kind of stuff in their libraries i think there should be a blocked off room where kids can't go.  That way the people who like that kind of stuff can still get to it but the children can not.       We go to school for a vast amount of reasons.  Some being; to learn, to get us ready for college, and to be independent.  While we are at school not only are we learing things like math and evolution, but we are also learing whats going on in the world around us.  If you haven't noticed, all that is out there anymore is bullying, cursing songs and horrifying movies.  We, as children and teenagers see and hear about this everyday.  Since we learn and already know about all of this 'offensive' stuff, why not have it in our libraries for other people who want to study it?     There is alot of different kinds of material in libraries.  Some good for kids and teenagers and some inappropriate.  I understand the concern for parents on censorship in public libraries, but there is a way to stop it.  I think as long as its blocked off into a different room, everyone will be fine and it will cause no harm"
text2 = "Once upon a time"
text3 = "The flying union had buttocks of steel"

perplexity1 = calculate_perplexity(text1)
perplexity2 = calculate_perplexity(text2)
perplexity3 = calculate_perplexity(text3)
perplexity = calculate_perplexity(text)
print(f'Perplexity: {perplexity}')
print(f'Perplexity1: {perplexity1}')
print(f'Perplexity2: {perplexity2}')
print(f'Perplexity3: {perplexity3}')

