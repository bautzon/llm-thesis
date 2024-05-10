const fs = require('fs');

// Read the file content
const fileContent = fs.readFileSync('output_llama2_prompt2_student.txt', 'utf-8');

// Split the content into entries
const entries = fileContent.split('-------\n');

// Define constants for creator and prompt
const CREATOR = "ai";
const PROMPT = "I am an American high school student that received the following question. Please answer it as a student would do. Censorship in the Libraries. ‘All of us can think of a book that we hope none of our children or any other children have taken off the shelf. But if I have the right to remove that book from the shelf -- that work I abhor -- then you also have exactly the same right and so does everyone else. And then we have no books left on the shelf for any of us.’ --Katherine Paterson, Author. Write a persuasive essay to a newspaper reflecting your views on censorship in libraries. Do you believe that certain materials, such as books, music, movies, magazines, etc., should be removed from the shelves if they are found offensive? Support your position with convincing arguments from your own experience, observations, and/or reading. Use specific examples and evidence to back up your claims.";

// Define an array to store parsed entries
const parsedEntries = [];

// Initialize ID counter
let id = 1;

// Regular expression to match the entire sentence to exclude
const regex = /system[\s\S]*?assistant/g;

// Make a regex that removes everything between "<s>" and "readers to agree with you."
const regex2 = /<s>[\s\S]*?experience, observations, and\/or reading./g;

// Iterate through each entry but only do it 100 times
entries.slice(0, 100).forEach(entry => {
    // Replace the entire sentence and newline characters with an empty string
    const content = entry.replace(regex2, '').replace("</s>", "").replace(/\n/g, "").trim();

    // Create entry object
    const entryObj = {
        id: id++,
        creator: CREATOR,
        prompt: PROMPT,
        answer: content
    };

    // Push the entry object to the parsed entries array
    parsedEntries.push(entryObj);
});

// Write the parsed entries into a JSON file
const outputFile = "prompt2_llama2_student.json";
fs.writeFileSync(outputFile, JSON.stringify(parsedEntries, null, 2));

console.log('JSON file created successfully.');
