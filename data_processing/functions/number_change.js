const fs = require('fs');

// Read the JSON file
fs.readFile('prompt_2_comb.json', 'utf8', (err, data) => {
  if (err) {
    console.error('Error reading file:', err);
    return;
  }

  try {
    // Parse the JSON data
    const jsonData = JSON.parse(data);

    // Access the array of answers
    const answers = jsonData.Answers;

    // Update IDs starting from 1
    answers.forEach((answer, index) => {
      answer.id = index + 1;
    });

    // Convert the updated JSON data back to a string
    const updatedJsonString = JSON.stringify(jsonData, null, 2);

    // Write the updated JSON back to the file
    fs.writeFile('prompt_2_combined_output.json', updatedJsonString, 'utf8', (err) => {
      if (err) {
        console.error('Error writing file:', err);
        return;
      }
      console.log('File updated successfully!');
    });
  } catch (error) {
    console.error('Error parsing JSON:', error);
  }
});
