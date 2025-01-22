function insertText(filename, elementId) {
    fetch(filename)
        .then(response => response.text())
        .then(data => {
            document.getElementById(elementId).textContent = data;
            hljs.highlightElement(document.getElementById(elementId));
        })
        .catch(error => console.error('Error fetching the code file:', error));
}