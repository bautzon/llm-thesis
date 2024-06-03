import fs from "fs";
import { execSync } from "child_process";

function executeCommand(command) {
    return new Promise((resolve, reject) => {
        execSync(command, (error, stdout, stderr) => {
            if (error) {
                console.error(`exec error: ${error}`);
                console.error(`stderr: ${stderr}`);
                reject(error);
                return;
            }
            //console.log(`stdout: ${stdout}`);
            resolve(stdout);
        });
    });
}

fs.readFile('prompt2_merged.json', 'utf8', (err, data) => {
    if (err) {
        console.error('Error reading file:', err);
        return;
    }

    let llama2_answers = [];
    let llama3_answers = [];
    let chatGpt3_answers = [];
    let chatGpt4_answers = [];
    let human_answers = [];

    const jsonData = JSON.parse(data);
    const answers = jsonData.Answers;

    for (let i = 0; i < answers.length; i++) {
        let answer = answers[i];

        // Extract the answer from the JSON data
        llama2_answers.push(answer.llama2);
        llama3_answers.push(answer.llama3);
        chatGpt3_answers.push(answer.chatGpt3);
        chatGpt4_answers.push(answer.chatGpt4);
        human_answers.push(answer.human);
    }


    // write the extracted data into seperate txt files, with each answer on a new line
    fs.writeFileSync('extracted_answers/prompt2_llama2_answers.txt', llama2_answers.join('\n'));
    fs.writeFileSync('extracted_answers/prompt2_llama3_answers.txt', llama3_answers.join('\n'));
    fs.writeFileSync('extracted_answers/prompt2_chatGpt3_answers.txt', chatGpt3_answers.join('\n'));
    fs.writeFileSync('extracted_answers/prompt2_chatGpt4_answers.txt', chatGpt4_answers.join('\n'));
    fs.writeFileSync('extracted_answers/prompt2_human_answers.txt', human_answers.join('\n'));
});

//
// # Your command
// command="zippy"
//
// # Read the file line by line
// while IFS= read -r line
// do
//     # Use echo to output the string and redirect it to a temporary file
// echo $line > temp.txt
//
// # Run the command with the temporary file as input and redirect the output to a file
// $command temp.txt > output.txt
//
// # Delete the temporary file
// rm temp.txt
// done < "eli5_chatGpt3_answers.txt"




