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

fs.readFile('eli5_100.json', 'utf8', (err, data) => {
    if (err) {
        console.error('Error reading file:', err);
        return;
    }

    try {
        const jsonData = JSON.parse(data);
        const answers = jsonData.rows;
        let extractedData = [];
        let startIndex = 20;
        (async () => {
            for (let i = startIndex; i < answers.length; i++) {
                let answer = answers[i];
                const question = answer.row.question.replace(/\"/g, '');
                const command = `../llama.cpp/main -m ../llama.cpp/models/llama-2-13b-chat.Q5_K_M.gguf -p "${question}" --log-disable`;
                let response = await executeCommand(command);
                response = response.replace(`<s> ${question}`, '').replace("</s>", '').replace(/([\u2700-\u27BF]|[\uE000-\uF8FF]|\uD83C[\uDC00-\uDFFF]|\uD83D[\uDC00-\uDFFF]|[\u2011-\u26FF]|\uD83E[\uDD10-\uDDFF])/g, '').replace(/\n/g, '').replace(/(\S)?\\\"/g, "").trim();

                const entryObj = {
                    id: answer.row.id,
                    creator: "ai",
                    prompt: question,
                    llama2: response,
                    chatGpt3: answer.row.chatgpt_answers[0].replace(/\"/g, '').replace(/\n/g, '').trim(),
                    human: answer.row.human_answers[0].replace(/\"/g, '').replace(/\n/g, '').trim()
                };

                extractedData.push(entryObj);
                console.log(entryObj);
                if (i >= startIndex + 9) break;
            }
            const outputFile = "eli5_llama2_2.json";
            const dataObject = { "Answers": extractedData };
            fs.writeFileSync(outputFile, JSON.stringify(dataObject, null, 2));
        })();


    } catch (error) {
        console.error('Error parsing JSON:', error);
    }
});