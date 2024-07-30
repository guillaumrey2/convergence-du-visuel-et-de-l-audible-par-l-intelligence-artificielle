window.addEventListener('DOMContentLoaded', () => {
    const container = document.getElementById('floatingImages');
    if (!container) {
        console.error("Container for floating images not found!");
        return;
    }

    // Clear previous images if any
    container.innerHTML = '';

    // Ensure images are loaded and positioned correctly
    const imageNames = ['0012', '0031', '0090', '0157', '0268']; // Specific image identifiers
    for (let i = 0; i < imageNames.length; i++) {
        const img = document.createElement('img');
        img.src = `/static/images/emotion_${imageNames[i]}.jpg`; // Correct path based on your image names
        img.className = 'floating-img';
        img.style.position = 'absolute';
        img.style.top = `${Math.random() * 100}%`;
        img.style.left = `${Math.random() * 100}%`;
        img.onload = () => {
            img.style.transform = `translate(-50%, -50%) scale(${Math.random() * 0.5 + 0.5})`;
        };
        img.onerror = () => {
            console.error("Failed to load image:", img.src);
        };
        container.appendChild(img);
    }

    // Mouse move effect
    document.addEventListener('mousemove', (e) => {
        const { clientX, clientY } = e;
        const centerX = window.innerWidth / 2;
        const centerY = window.innerHeight / 2;
        const deltaX = (clientX - centerX) * 0.1; // Adjust for more subtle movement
        const deltaY = (clientY - centerY) * 0.1;

        Array.from(container.children).forEach(img => {
            const initialX = parseFloat(img.style.left, 10);
            const initialY = parseFloat(img.style.top, 10);
            img.style.left = `${initialX + deltaX}px`;
            img.style.top = `${initialY + deltaY}px`;
        });
    });
});
