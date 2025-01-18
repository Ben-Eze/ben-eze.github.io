import Tokeniser from './Tokeniser.js';


export default class Model {
    constructor(tokeniser, model) {
        this.tokeniser = tokeniser;
        this.N_vocab = tokeniser.vocab.length;
        this.model = model;
        this.N_context = 16;
    }

    static async load(modelPath, tokenizerPath) {
        const tokeniser = Tokeniser.load(tokenizerPath);
        let model;

        try {
            // Load the ONNX model
            model = await ort.InferenceSession.create(modelPath);
        } catch (error) {
            console.error('Error loading model:', error);
        }

        return new Model(tokeniser, model);
    }

    async forward(context) {
        const encodedContext = this.tokeniser.encode(
            context.slice(-self.N_context)
        );
        console.log('Encoded context:', encodedContext);

        const X_array = BigInt64Array.from(encodedContext.map(x => BigInt(x)));
        const X_tensor = new ort.Tensor('int64', X_array, [1, context.length]);
        const X_obj = { input: X_tensor };

        // Run the model
        const results = await this.model.run(X_obj);

        // Assuming the model returns logits as an array
        const allLogits = results[Object.keys(results)[0]];
        const logits = [...allLogits.data.slice(-this.N_vocab)]; // Get the last N_vocab logits
        const probs = this.softmax(logits);
        const predToken = this.sampleFromDistribution(probs);
        const predChar = this.tokeniser.itos[predToken];
        return predChar;
    }

    softmax(logits) {
        const expValues = logits.map(x => Math.exp(x));
        const sum = expValues.reduce((acc, val) => acc + val, 0);
        return expValues.map(x => x / sum);
    }

    sampleFromDistribution(probs) {
        const rand = Math.random();
        let cumulativeProbability = 0;

        for (let i = 0; i < probs.length; i++) {
            cumulativeProbability += probs[i];
            if (rand < cumulativeProbability) {
                return i;
            }
        }
        return probs.length - 1; 
    }
}