const fs = require('fs');

const fileContent = fs.readFileSync('output_llama2_prompt2_student.txt', 'utf-8');
const entries = fileContent.split('-------\n');
const CREATOR = "ai";
const PROMPT = "I am an American high school student that received the following question. Please answer it as a student would do. Censorship in the Libraries. ‘All of us can think of a book that we hope none of our children or any other children have taken off the shelf. But if I have the right to remove that book from the shelf -- that work I abhor -- then you also have exactly the same right and so does everyone else. And then we have no books left on the shelf for any of us.’ --Katherine Paterson, Author. Write a persuasive essay to a newspaper reflecting your views on censorship in libraries. Do you believe that certain materials, such as books, music, movies, magazines, etc., should be removed from the shelves if they are found offensive? Support your position with convincing arguments from your own experience, observations, and/or reading. Use specific examples and evidence to back up your claims.";
const parsedEntries = [];
let id = 1;
const regex = /system[\s\S]*?assistant/g;
const regex2 = /<s>[\s\S]*?experience, observations, and\/or reading./g;

entries.slice(0, 100).forEach(entry => {
    const content = entry.replace(regex2, '').replace("</s>", "").replace(/\n/g, "").trim();

    const entryObj = {
        id: id++,
        creator: CREATOR,
        prompt: PROMPT,
        answer: content
    };


    parsedEntries.push(entryObj);
});

const outputFile = "prompt2_llama2_student.json";
fs.writeFileSync(outputFile, JSON.stringify(parsedEntries, null, 2));

console.log('JSON file created successfully.');
