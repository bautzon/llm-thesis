const { exec } = require('child_process');
const fs = require('fs');

function executeCommand(command) {
    return new Promise((resolve, reject) => {
        exec(command, (error, stdout, stderr) => {
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

fs.readFile('open_qa_cleaned.json', 'utf8', (err, data) => {
    if (err) {
        console.error('Error reading file:', err);
        return;
    }

    try {
        const jsonData = JSON.parse(data);
        const answers = jsonData.Answers;
        let extractedData = [];
        let startIndex = 90;
        let eli5_regex = /<\|begin_of_text\|><\|begin_of_text\|><\|start_header_id\|>system[\s\S]*?assistant<\|end_header_id\|>/g;
        (async () => {
            for (let i = startIndex; i < answers.length; i++) {
                let answer = answers[i];
                const question = answer.prompt;
                // const final_question = `<|begin_of_text|><|start_header_id|>system<|end_header_id|>{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>{${question}}<|eot_id|><|start_header_id|>assistant<|end_header_id|>`;
                const command = `../llama.cpp/main -m ../llama.cpp/models/llama-2-13b-chat.Q5_K_M.gguf -p "${question}" --log-disable`;
                // const command = `../llama.cpp/main -m ../llama.cpp/models/Meta-Llama-3-8B-Instruct-Q8_0.gguf -p "${final_question}" --log-disable -c 2048`;
                let response = await executeCommand(command);
                response = response.replace(`<s> ${question}`, '').replace("</s>", '').replace(/([\u2700-\u27BF]|[\uE000-\uF8FF]|\uD83C[\uDC00-\uDFFF]|\uD83D[\uDC00-\uDFFF]|[\u2011-\u26FF]|\uD83E[\uDD10-\uDDFF])/g, '').replace(/\n/g, '').replace(/(\S)?\\\"/g, "").trim();
                // response = response.replace(regex, '').replace("<|eot_id|>", "").replace(/([\u2700-\u27BF]|[\uE000-\uF8FF]|\uD83C[\uDC00-\uDFFF]|\uD83D[\uDC00-\uDFFF]|[\u2011-\u26FF]|\uD83E[\uDD10-\uDDFF])/g, '').replace(/\n/g, '').replace(/(\S)?\\\"/g, "").trim();

                const entryObj = {
                    id: answer.id,
                    creator: "ai",
                    prompt: answer.prompt,
                    llama2: response,
                    // llama3: answer.llama3,
                    chatGpt3: answer.chatGpt3,
                    human: answer.human
                };

                extractedData.push(entryObj);
                console.log(entryObj);
                if (i >= startIndex + 9) break;
            }
            const outputFile = "openqa_llama2.json";
            const dataObject = { "Answers": extractedData };
            fs.writeFileSync(outputFile, JSON.stringify(dataObject, null, 2));
        })();


    } catch (error) {
        console.error('Error parsing JSON:', error);
    }
});
