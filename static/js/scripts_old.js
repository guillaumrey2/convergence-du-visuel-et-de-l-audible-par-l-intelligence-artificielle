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
        img.style.transform = `translate(-50%, -50%) scale(${Math.random() * 0.5 + 0.5})`;
        img.style.filter = 'blur(8px)';
        container.appendChild(img);

        // Random movement direction and speed
        const speedX = Math.random() * 0.5 - 0.25; // Speed and direction for horizontal movement
        const speedY = Math.random() * 0.5 - 0.25; // Speed and direction for vertical movement
        setInterval(() => {
            const topVal = parseFloat(img.style.top);
            const leftVal = parseFloat(img.style.left);
            img.style.top = `${(topVal + speedY + 100) % 100}%`; // Ensure the values wrap around
            img.style.left = `${(leftVal + speedX + 100) % 100}%`; // Ensuring wrapping around
        }, 50);
    }
});
