window.addEventListener('DOMContentLoaded', () => {
    const container = document.getElementById('floatingImages');
    const imageNames = ['emotion_0012.jpg', 'emotion_0031.jpg', 'emotion_0090.jpg', 'emotion_0157.jpg', 'emotion_0268.jpg']; // Your image filenames
    for (let i = 0; i < 50; i++) { // Create more images for a denser effect
        const img = document.createElement('img');
        img.src = `/static/images/${imageNames[i % imageNames.length]}`; // Correct path to the image
        img.className = 'floating-img';
        img.style.position = 'absolute';
        img.style.top = `${Math.random() * 100}%`;
        img.style.left = `${Math.random() * 100}%`;
        img.style.transform = `translate(-50%, -50%) scale(${Math.random() * 0.5 + 0.5})`;
        img.style.filter = 'blur(8px)'; // Adding blur to the images
        container.appendChild(img);
    }

    // Mouse move effect
    document.addEventListener('mousemove', (e) => {
        const { clientX, clientY } = e;
        const centerX = window.innerWidth / 2;
        const centerY = window.innerHeight / 2;
        const deltaX = (clientX - centerX) * 0.05; // Decreased effect strength for subtlety
        const deltaY = (clientY - centerY) * 0.05; // Decreased effect strength for subtlety

        Array.from(container.children).forEach(img => {
            const initialX = parseFloat(img.style.left);
            const initialY = parseFloat(img.style.top);
            img.style.left = initialX + deltaX + 'px';
            img.style.top = initialY + deltaY + 'px';
        });
    });
});
