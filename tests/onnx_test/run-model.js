async function runModel(X_test) {
    try {
        // Load the ONNX model
        const model = await ort.InferenceSession.create('gpt_model.onnx');

        // Run the model
        const feeds = { input: X_test };
        const results = await model.run(feeds);

        // Display the output
        const outputTensor = results[Object.keys(results)[0]];
        document.getElementById('output').textContent = `Output: ${outputTensor.data}`;
    } catch (error) {
        console.error(error);
        document.getElementById('output').textContent = `Error: ${error.message}`;
    }
}

// Create input data (adjust the shape and values as needed)
// const input = new BigInt64Array(5).fill(1n); // Example input for int64 model
const X_data = BigInt64Array.from([0n, 1n, 2n, 3n, 4n, 5n, 6n, 7n, 8n, 9n]);
const X_test = new ort.Tensor('int64', X_data, [2, 5]);        

// Attach event listener to the button
document.addEventListener('DOMContentLoaded', () => {
    document.getElementById('run-model').addEventListener('click', async () => {await runModel(X_test);});
});