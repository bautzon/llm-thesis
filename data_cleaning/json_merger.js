import fs from "fs";

// List of JSON files to merge
const files = ["../Test-Data/prompt_2_human_output.json", "../Test-Data/prompt2_llama2_student.json", "prompt2_llama3_student.json", "../Test-Data/output_chatGpt_prompt2_Student.json", "../Test-Data/prompt2_gpt4_student.json"]; // Add your file names here

let mergedData = {};

files.forEach((file, index) => {
    // Read the JSON file
    const data = fs.readFileSync(file, 'utf8');

    // Parse the JSON data
    const jsonData = JSON.parse(data);


    // Iterate over the "Answers" array
    jsonData.Answers.forEach(answer => {
        // If an object with the same "id" exists in the merged data, add the "answer" with a new key
        if (mergedData[answer.id]) {
            mergedData[answer.id][`${file}`] = answer.answer;
        } else {
            // If it doesn't exist, add the object to the merged data
            mergedData[answer.id] = {
                id: answer.id,
                creator: answer.creator,
                prompt: answer.prompt,
                [`${file}`]: answer.answer
            };
        }
    });
});

// Convert the merged data to an array of objects
const mergedArray = Object.values(mergedData);

// Stringify the result
const result = JSON.stringify({ Answers: mergedArray }, null, 2);

// Write the result back to a new JSON file
fs.writeFileSync('prompt2_merged.json', result);
