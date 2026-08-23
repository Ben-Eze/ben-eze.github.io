import tokeniser_json from './shakespeare_tokeniser.js';

export default class Tokeniser {
    constructor(vocab, specialToken = "□") {
        this.vocab = vocab;
        this.N_vocab = vocab.length;
        this.specialToken = specialToken;

        this.stoi = {};
        this.itos = {};

        vocab.forEach((s, i) => {
            this.stoi[s] = i;
            this.itos[i] = s;
        });

        this.stoi[this.specialToken] = vocab.length;
        this.itos[vocab.length] = this.specialToken;
    }

    encode(string) {
        return string.split('').map(s => {
            return this.stoi.hasOwnProperty(s) ? this.stoi[s] : this.stoi[this.specialToken];
        });
    }

    decode(intList) {
        return intList.map(i => this.itos[i] || this.specialToken).join('');
    }

    static load(path) {
        // const data = JSON.parse(fs.readFileSync(path, 'utf8'));
        return new Tokeniser(tokeniser_json["vocab"], 
                             tokeniser_json["special_token"]);
    }
}
