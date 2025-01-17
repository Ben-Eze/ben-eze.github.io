const textBox = document.getElementById('textBox');
const suggestionWrapper = document.getElementById('suggestionWrapper');

let autocompleteText = ''; // Current autocomplete suggestion

// Event listener for input
textBox.addEventListener('input', () => {
    const value = textBox.value;
    const lastChar = value.slice(-1);

    // Reset suggestion if last character is '.' or ' ' or if the text is empty
    if (lastChar === '.' || lastChar === ' ' || !value) {
        suggestionWrapper.innerHTML = value; // Match the text so far
        autocompleteText = '';
    } else {
        // Show autocomplete suggestion
        autocompleteText = 'xxx.';
        suggestionWrapper.innerHTML = 
            value + `<span class="suggestion">${autocompleteText}</span>`;
    }
});

// Event listener for Tab key
textBox.addEventListener('keydown', (event) => {
    if (event.key === 'Tab' && autocompleteText) {
        event.preventDefault(); // Prevent default Tab behavior
        textBox.value += autocompleteText + ' '; // Add autocomplete and a space
        suggestionWrapper.innerHTML = textBox.value; // Update the wrapper
        autocompleteText = ''; // Reset autocomplete
    }
});

// Sync suggestion position and style
textBox.addEventListener('scroll', () => {
    suggestionWrapper.scrollTop = textBox.scrollTop;
});