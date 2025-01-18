function softmax(logits) {
    const expValues = logits.map(x => Math.exp(x));
    const sum = expValues.reduce((acc, val) => acc + val, 0);
    return expValues.map(x => x / sum);
}

function sampleFromDistribution(probs) {
    const rand = Math.random();
    let cumulativeProbability = 0;

    for (let i = 0; i < probs.length; i++) {
        cumulativeProbability += probs[i];
        if (rand < cumulativeProbability) {
            return i;
        }
    }
    return probs.length - 1; // Return the last index if no other index is selected
}

// async function runModel(X) {
//     try {
//         // Load the ONNX model
//         const model = await ort.InferenceSession.create('gpt_model.onnx');

//         // Run the model
//         const feeds = { input: X };
//         const results = await model.run(feeds);

//         // Display the output
//         const allLogits = results[Object.keys(results)[0]];
//         const logits = [...allLogits.data.slice(-61)]; // Get the last 61 logits
//         const probs = softmax(logits);
//         const prediction = sampleFromDistribution(probs);
    
//         const outputElement = document.getElementById('output');
//         outputElement.textContent = `Logits: ${logits}\n`;
//         outputElement.textContent += `Probabilities: ${probs}`;
//         outputElement.textContent += `\nSampled token: ${prediction}`;
//     } catch (error) {
//         console.error(error);
//         document.getElementById('output').textContent = `Error: ${error.message}`;
//     }
// }

async function runModel(X, N_vocab) {
    let model;
    try {
        // Load the ONNX model
        model = await ort.InferenceSession.create('gpt_model.onnx');
    } catch (error) {
        console.error(error);
    }

    // Run the model
    const X_obj = { input: X };
    const results = await model.run(X_obj);

    // Display the output
    const allLogits = results[Object.keys(results)[0]];
    const logits = [...allLogits.data.slice(-N_vocab)]; // Get the last 61 logits
    const probs = softmax(logits);
    const prediction = sampleFromDistribution(probs);
    
    return prediction;
}

// Create input data (adjust the shape and values as needed)
// const input = new BigInt64Array(5).fill(1n); // Example input for int64 model
const X_data = BigInt64Array.from([0n, 1n, 2n, 3n, 4n, 5n, 6n, 7n, 8n, 9n, 10n, 11n, 12n, 13n, 14n, 15n]);
const X_test = new ort.Tensor('int64', X_data, [1, 16]);        

// Attach event listener to the button
document.addEventListener('DOMContentLoaded', () => {
    document.getElementById('run-model').addEventListener('click', async () => {
        const prediction = await runModel(X_test, 61);
        const outputElement = document.getElementById('output');
        outputElement.textContent = `Sampled token: ${prediction}`;
    });
});