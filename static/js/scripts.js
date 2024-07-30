window.addEventListener('DOMContentLoaded', () => {
    const container = document.getElementById('floatingImages');
    for (let i = 0; i < 5; i++) {  // Assuming you have 5 images to cycle through
        const img = document.createElement('img');
        img.src = `/static/images/emotion_00${12 + i * 31}.jpg`;  // Update to your actual image paths
        img.className = 'floating-img';
        img.style.position = 'absolute';
        img.style.top = `${Math.random() * 100}%`;
        img.style.left = `${Math.random() * 100}%`;
        img.style.transform = `translate(-50%, -50%) scale(${Math.random() * 0.5 + 0.5})`;
        container.appendChild(img);
    }

    // Mouse move effect
    document.addEventListener('mousemove', (e) => {
        const { clientX, clientY } = e;
        const centerX = window.innerWidth / 2;
        const centerY = window.innerHeight / 2;
        const deltaX = (clientX - centerX) * 0.1;
        const deltaY = (clientY - centerY) * 0.1;

        Array.from(container.children).forEach(img => {
            const initialX = parseFloat(img.style.left, 10);
            const initialY = parseFloat(img.style.top, 10);
            img.style.left = `${initialX + deltaX}px`;
            img.style.top = `${initialY + deltaY}px`;
        });
    });
});
