// Floating images setup and animation
window.addEventListener('DOMContentLoaded', () => {
    const container = document.getElementById('floatingImages');
    const imageNames = ['emotion_0012.jpg', 'emotion_0031.jpg', 'emotion_0090.jpg', 'emotion_0157.jpg', 'emotion_0268.jpg'];
    for (let i = 0; i < 50; i++) {
        const img = document.createElement('img');
        img.src = `/static/images/${imageNames[i % imageNames.length]}`;
        img.className = 'floating-img';
        img.style.position = 'absolute';
        img.style.top = `${Math.random() * 100}%`;
        img.style.left = `${Math.random() * 100}%`;
        img.style.transform = `translate(-50%, -50%) scale(${Math.random() * 0.5 + 3})`;
        img.style.opacity = 0; // Start transparent
        container.appendChild(img);
        setTimeout(() => img.style.opacity = 0.6, 10 + 100 * i); // Fade in effect

        // Random movement direction and speed
        const speedX = Math.random() * 0.1 - 0.05;
        const speedY = Math.random() * 0.1 - 0.05;
        setInterval(() => {
            const topVal = parseFloat(img.style.top);
            const leftVal = parseFloat(img.style.left);
            img.style.top = `${(topVal + speedY + 100) % 100}%`;
            img.style.left = `${(leftVal + speedX + 100) % 100}%`;
        }, 100);
    }
});

// Check status for processing request
function checkStatus(filename) {
    $.getJSON('/check_status/' + filename, function(data) {
        if (data.ready) {
            window.location.href = '/recordings/' + filename;
        } else {
            setTimeout(() => checkStatus(filename), 5000);
        }
    }).fail(function() {
        console.log("Error checking status");
        setTimeout(() => checkStatus(filename), 5000);
    });
}
