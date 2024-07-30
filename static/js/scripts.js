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
        // Increase the scale factor for larger images
        img.style.transform = `translate(-50%, -50%) scale(${Math.random() * 0.5 + 1})`;  // Increased minimum scale to 1
        img.style.filter = 'blur(8px)';
        container.appendChild(img);

        // Reduce the speed for slower movement
        const speedX = Math.random() * 0.1 - 0.05; // Reduced speed for horizontal movement
        const speedY = Math.random() * 0.1 - 0.05; // Reduced speed for vertical movement
        setInterval(() => {
            const topVal = parseFloat(img.style.top);
            const leftVal = parseFloat(img.style.left);
            img.style.top = `${(topVal + speedY + 100) % 100}%`; // Ensure the values wrap around
            img.style.left = `${(leftVal + speedX + 100) % 100}%`; // Ensuring wrapping around
        }, 100); // Increased interval to slow down the movement
    }
});