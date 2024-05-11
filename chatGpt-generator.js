import OpenAI from "openai";
import fs from "fs";

const openai = new OpenAI();

const prompt_1 = "More and more people use computers, but not everyone agrees that this benefits society. Those who support advances in technology believe that computers have a positive effect on people. They teach hand-eye coordination, give people the ability to learn about faraway places and people, and even allow people to talk online with other people. Others have different ideas. Some experts are concerned that people are spending too much time on their computers and less time exercising, enjoying nature, and interacting with family and friends. Write a letter to your local newspaper in which you state your opinion on the effects computers have on people. Persuade the readers to agree with you.";
const prompt_1_human_like = "Answer the following and make it sound like it was written by a human. More and more people use computers, but not everyone agrees that this benefits society. Those who support advances in technology believe that computers have a positive effect on people. They teach hand-eye coordination, give people the ability to learn about faraway places and people, and even allow people to talk online with other people. Others have different ideas. Some experts are concerned that people are spending too much time on their computers and less time exercising, enjoying nature, and interacting with family and friends. Write a letter to your local newspaper in which you state your opinion on the effects computers have on people. Persuade the readers to agree with you.";
const prompt_1_student = "I am an American high school student that received the following question. Please answer it as a student would do. More and more people use computers, but not everyone agrees that this benefits society. Those who support advances in technology believe that computers have a positive effect on people. They teach hand-eye coordination, give people the ability to learn about faraway places and people, and even allow people to talk online with other people. Others have different ideas. Some experts are concerned that people are spending too much time on their computers and less time exercising, enjoying nature, and interacting with family and friends. Write a letter to your local newspaper in which you state your opinion on the effects computers have on people. Persuade the readers to agree with you.";

const prompt_2 = "Censorship in the Libraries. ‘All of us can think of a book that we hope none of our children or any other children have taken off the shelf. But if I have the right to remove that book from the shelf -- that work I abhor -- then you also have exactly the same right and so does everyone else. And then we have no books left on the shelf for any of us.’ --Katherine Paterson, Author. Write a persuasive essay to a newspaper reflecting your views on censorship in libraries. Do you believe that certain materials, such as books, music, movies, magazines, etc., should be removed from the shelves if they are found offensive? Support your position with convincing arguments from your own experience, observations, and/or reading.";
const prompt_2_human_like = "Answer the following and make it sound like it was written by a human. Censorship in the Libraries. ‘All of us can think of a book that we hope none of our children or any other children have taken off the shelf. But if I have the right to remove that book from the shelf -- that work I abhor -- then you also have exactly the same right and so does everyone else. And then we have no books left on the shelf for any of us.’ --Katherine Paterson, Author. Write a persuasive essay to a newspaper reflecting your views on censorship in libraries. Do you believe that certain materials, such as books, music, movies, magazines, etc., should be removed from the shelves if they are found offensive? Support your position with convincing arguments from your own experience, observations, and/or reading.";
const prompt_2_student = "I am an American high school student that received the following question. Please answer it as a student would do. Censorship in the Libraries. ‘All of us can think of a book that we hope none of our children or any other children have taken off the shelf. But if I have the right to remove that book from the shelf -- that work I abhor -- then you also have exactly the same right and so does everyone else. And then we have no books left on the shelf for any of us.’ --Katherine Paterson, Author. Write a persuasive essay to a newspaper reflecting your views on censorship in libraries. Do you believe that certain materials, such as books, music, movies, magazines, etc., should be removed from the shelves if they are found offensive? Support your position with convincing arguments from your own experience, observations, and/or reading.";

async function generate() {
  const completion = await openai.chat.completions.create({
    messages: [{ role: "system", content: "You are a helpful assistant." },
        { role: "user", content: prompt_2_student}
    ],
    model: "gpt-4",
  });
  
  //console.log(completion.choices[0].message.content);
  return completion;
}

async function main() {
    let answers = {Answers: []};
    for (let i = 0; i < 100; i++) {
        let res = await generate();
            answers.Answers.push({
                id: i+1,
                creator: "ai",
                prompt: prompt_2_student,
                answer: res.choices[0].message.content.replace(/\n/g, " ")
            });
            fs.writeFileSync("prompt1_gpt4_student.json", JSON.stringify(answers, null, 4));
    }
}

main();