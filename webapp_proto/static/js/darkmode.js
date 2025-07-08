// Check for saved theme preference or default to light mode
const currentTheme = localStorage.getItem('theme') || 'light';
document.documentElement.setAttribute('data-theme', currentTheme);
updateThemeToggle(currentTheme);

function toggleDarkMode() {
    const currentTheme = document.documentElement.getAttribute('data-theme');
    const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
    
    document.documentElement.setAttribute('data-theme', newTheme);
    localStorage.setItem('theme', newTheme);
    updateThemeToggle(newTheme);
}

function updateThemeToggle(theme) {
    const themeIcon = document.getElementById('theme-icon');
    const themeText = document.getElementById('theme-text');
    
    if (theme === 'dark') {
        themeIcon.textContent = '☀️';
        themeText.textContent = 'Light Mode';
    } else {
        themeIcon.textContent = '🌙';
        themeText.textContent = 'Dark Mode';
    }
}





function showLoading(show, which) {
    let classname = '.loader-container';

    if (which === 1)
        classname = classname + '-1';
    else if (which === 2)
        classname = classname + '-2';

    // All elements with class "loader-container"
    const loaderContainers = document.querySelectorAll(classname);
    console.log('Found ' + loaderContainers.length + ' loader containers');
    loaderContainers.forEach(container => {
        container.style.display = show ? 'flex' : 'none';
        if (show) {
            console.log('Showing loader container');
        } else {
            console.log('Hiding loader container');
        }
    });

}
