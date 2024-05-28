const fs = require('fs');
const csv = require('csv-parser');

// Define the path to your CSV file
const csvFilePath = 'human_answers_dataset.csv';
// Define the path to the JSON output file
const jsonFilePath = 'human_answers_cleaned.json';

// Define an array to store the extracted data
let extractedData = [];

csv(['essay_id', 'essay_set', 'essay']);
csv({ seperAtor: '\t'});

// Read the CSV file and extract data
fs.createReadStream(csvFilePath)
    .pipe(
            csv({
                delimiter: ";",
                columns: true,
                ltrim: true,
                from_line: 2
            })
        )    
    .on('data', (row) => {
        console.log(row);
        // Check if the essay_set is 1
        if (row.essay_set === '1') {
         
            // Push the data to the extractedData array
            extractedData.push({
                essay_id: row.essay_id,
                essay_set: row.essay_set,
                essay: row.essay
            });

            // Check if we've extracted 100 rows already
            if (extractedData.length >= 100) {
                // If we have 100 rows, close the stream to stop reading the file
                return;
            }
        }
    })
    .on('end', () => {
        // Write the extracted data to a JSON file
        fs.writeFile(jsonFilePath, JSON.stringify(extractedData, null, 2), (err) => {
            if (err) {
                console.error('Error writing JSON file:', err);
                return;
            }
            console.log('Data successfully written to', jsonFilePath);
        });
    })
    .on('error', (error) => {
        // Handle any errors that occur during parsing
        console.error('Error:', error.message);
    });
