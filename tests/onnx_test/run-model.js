import Model from "./Model.js";

const model = await Model.load('gpt_model.onnx', 'shakespeare_tokenizer.json');
console.log(model);
  

// Attach event listener to the button
document.getElementById('run-model').addEventListener('click', async () => {
    console.log('Running model...');
    const prediction = await model.forward("Ham");
    console.log(prediction);
    const outputElement = document.getElementById('output');
    outputElement.textContent = `Sampled token: ${prediction}`;
});
