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
            resolve(stdout);
        });
    });
}

fs.readFile('open_qa.json', 'utf8', (err, data) => {
    if (err) {
        console.error('Error reading file:', err);
        return;
    }

    try {
        const jsonData = JSON.parse(data);
        const answers = jsonData.rows;
        let extractedData = [];
        let startIndex = 0;
        (async () => {
            for (let i = startIndex; i < answers.length; i++) {
                let answer = answers[i];
                const question = answer.row.question;
                // const command = `../llama.cpp/main -m ../llama.cpp/models/llama-2-13b-chat.Q5_K_M.gguf -p "${question}" --log-disable`;
                // const command = `../llama.cpp/main -m ../llama.cpp/models/Meta-Llama-3-8B-Instruct-Q8_0.gguf -p "${final_question}" --log-disable -c 2048`;
                // let response = await executeCommand(command);
                // response = response.replace(`<s> ${question}`, '').replace("</s>", '').replace(/([\u2700-\u27BF]|[\uE000-\uF8FF]|\uD83C[\uDC00-\uDFFF]|\uD83D[\uDC00-\uDFFF]|[\u2011-\u26FF]|\uD83E[\uDD10-\uDDFF])/g, '').replace(/\n/g, '').replace(/(\S)?\\\"/g, "").trim();
                // response = response.replace(regex, '').replace("<|eot_id|>", "").replace(/([\u2700-\u27BF]|[\uE000-\uF8FF]|\uD83C[\uDC00-\uDFFF]|\uD83D[\uDC00-\uDFFF]|[\u2011-\u26FF]|\uD83E[\uDD10-\uDDFF])/g, '').replace(/\n/g, '').replace(/(\S)?\\\"/g, "").trim();

                const entryObj = {
                    id: answer.row.id,
                    creator: "ai",
                    prompt: question,
                    // llama2: response,
                    // llama3: answer.llama3,
                    chatGpt3: answer.row.chatgpt_answers[0].replace(/\\n/g, '').replace(/(\S)?\\\"/g, "").trim(),
                    human: answer.row.human_answers[0].replace(/\\n/g, '').replace(/(\S)?\\\"/g, "").trim()
                };

                extractedData.push(entryObj);
                console.log(entryObj);
                if (i >= startIndex + 99) break;
            }
            const outputFile = "open_qa.json";
            const dataObject = { "Answers": extractedData };
            fs.writeFileSync(outputFile, JSON.stringify(dataObject, null, 2));
        })();

    } catch (error) {
        console.error('Error parsing JSON:', error);
    }
});
