function toggleDarkMode() {
    console.log("button clicked");
    const body = document.body;
    const button = document.querySelector('.dark-mode-toggle');

    // Toggle the dark mode class
    body.classList.toggle('dark-mode');
    console.log("Body class list:", body.classList);

    // Change the button's text
    if (body.classList.contains('dark-mode')) {
        button.textContent = '☀️';
    } else {
        button.textContent = '🌙';
    }
}
