async function forward(context) {
    let chars = ["a", "b", "c", "d", "e", "f", ".", " "];
    // Simulate a delay
    await new Promise(resolve => setTimeout(resolve, 100));

    return chars[Math.floor(Math.random() * chars.length)];
}

export default async function* nextChunk(originalContext, contextSize = 8) {
    var chunk = "";

    while (true) {
        var context = (originalContext + chunk).slice(-contextSize);
        chunk += await forward(context);
        yield chunk; 
        if (chunk.slice(-1) === ".") {
            break;
        }
    }
}