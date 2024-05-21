import OpenAI from "openai";
import fs from "fs";
// const { exec } = require('child_process');
import { execSync } from "child_process";

const openai = new OpenAI();

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

async function generateGpt(question) {
    const completion = await openai.chat.completions.create({
        messages: [{ role: "system", content: "You are a helpful assistant." },
            { role: "user", content: question}
        ],
        model: "gpt-4",
    });
    //console.log(completion.choices[0].message.content);
    return completion;
}

fs.readFile('eli5_llama2.json', 'utf8', (err, data) => {
    if (err) {
        console.error('Error reading file:', err);
        return;
    }

    try {
        const jsonData = JSON.parse(data);
        const answers = jsonData.Answers;
        let extractedData = [];
        let startIndex = 80;
        let llama3_regex = /<\|begin_of_text\|><\|begin_of_text\|><\|start_header_id\|>system[\s\S]*?assistant<\|end_header_id\|>/g;
        (async () => {
            for (let i = startIndex; i < answers.length; i++) {
                let answer = answers[i];
                const question = answer.prompt;
                // const final_question = `<|begin_of_text|><|start_header_id|>system<|end_header_id|>{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>{${question}}<|eot_id|><|start_header_id|>assistant<|end_header_id|>`;
                // const llama2-command = `../llama.cpp/main -m ../llama.cpp/models/llama-2-13b-chat.Q5_K_M.gguf -p "${question}" --log-disable`;
                // const llama3-command = `../llama.cpp/main -m ../llama.cpp/models/Meta-Llama-3-8B-Instruct-Q8_0.gguf -p "${final_question}" --log-disable -c 2048`;
                // let response = await executeCommand(command);
                // response = response.replace(`<s> ${question}`, '').replace("</s>", '').replace(/([\u2700-\u27BF]|[\uE000-\uF8FF]|\uD83C[\uDC00-\uDFFF]|\uD83D[\uDC00-\uDFFF]|[\u2011-\u26FF]|\uD83E[\uDD10-\uDDFF])/g, '').replace(/\n/g, '').replace(/(\S)?\\\"/g, "").trim();
                // response = response.replace(eli5_regex, '').replace("<|eot_id|>", "").replace(/([\u2700-\u27BF]|[\uE000-\uF8FF]|\uD83C[\uDC00-\uDFFF]|\uD83D[\uDC00-\uDFFF]|[\u2011-\u26FF]|\uD83E[\uDD10-\uDDFF])/g, '').replace(/\n/g, '').replace(/(\S)?\\\"/g, "").trim();
                let response = await generateGpt(question)
                const entryObj = {
                    id: answer.id,
                    creator: "ai",
                    prompt: answer.prompt,
                    llama2: answer.llama2,
                    llama3: response,
                    chatGpt3: answer.chatGpt3,
                    chatGpt4: response.choices[0].message.content.replace(/\n/g, " "),
                    human: answer.human
                };

                extractedData.push(entryObj);
                console.log(entryObj);
                if (i >= startIndex + 19) break;
            }
            const outputFile = "eli5_gpt4.json";
            const dataObject = { "Answers": extractedData };
            fs.writeFileSync(outputFile, JSON.stringify(dataObject, null, 2));
        })();


    } catch (error) {
        console.error('Error parsing JSON:', error);
    }
});
