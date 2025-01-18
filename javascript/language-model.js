import Model from "./Model.js";

const model = await Model.load(
    './assets/models/gpt_model.onnx', 
    '.shakespeare_tokenizer.json');

export default async function* nextChunk(originalContext, contextSize = 8) {
    var chunk = "";

    while (true) {
        var context = (originalContext + chunk).slice(-contextSize);
        chunk += await model.forward(context);
        yield chunk; 
        // if ([" ", ".", ","].includes(chunk.at(-1))) {
        if (["\n", "."].includes(chunk.at(-1))) {
            break;
        }
    }
}